from pathlib import Path
import sys
sys.path.append(str(Path('.').absolute()))

from itertools import product
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from Utils import *
from tqdm import tqdm
from record_processors import *
import dill
from tensorboardX import SummaryWriter

def run(config, device=torch.device('cuda')):
    cpu = torch.device('cpu')

    task = config['task']
    id_base = config['id_base']
    processors = config['processors']
    savers_init = config['savers_init']
    optimizer_init = config['optimizer_init']
    scheduler_init = config['scheduler_init']

    train = config['train']
    batch_size = config['batch_size']
    num_train_epochs = config['num_train_epochs']
    parallel = config['parallel']
    dataset_class = config['dataset_class']
    model_class = config['model_class']
    
    task_path = f'{task}/exp-{id_base}'
    task_file_path = f'output/{task_path}/tasks/data.csv'
    pkl_dump(config, f'output/{task_path}/tasks/config.pkl')

    label_info = json_load(f'output/{task_path}/tasks/label_info.json')
    config = {**config, **label_info}

    n_classes = config['n_classes']
    label_cols = config['label_cols']

    df = pd.read_csv(task_file_path, low_memory=False)
    results = defaultdict(list)


    print(f'task={task}, id_base={id_base}')
    name = f"{task}-exp-{id_base}"

    train_df = df[df['dataset'].isin([0, 1, 2])].copy()
    valid_df = df[df['dataset'] == 3].copy()
    # train_df = df[df['dataset'] == 'train'].copy()
    # valid_df = df[df['dataset'] == 'valid'].copy()
    if config['external_test']:
        df_ex = pd.read_csv(f'output/0124-ex/d56_external/emb_pretrain_resnet50_raw/exp-pid/tasks/data.csv', low_memory=False)
        test_df = df_ex.copy()
    else:
        test_df = df[df['dataset'] == 4].copy()
        # test_df = df[df['dataset'] == 'test'].copy()

    train_ds = dataset_class(train_df, 'train', config)
    valid_ds = dataset_class(valid_df, 'valid', config)
    test_ds = dataset_class(test_df, 'test', config)
#

#
    train_dl = DataLoader(train_ds, sampler=RandomSampler(train_ds), batch_size=batch_size, num_workers=32)
    valid_dl = DataLoader(valid_ds, sampler=SequentialSampler(valid_ds), batch_size=batch_size, num_workers=32)
    test_dl = DataLoader(test_ds, sampler=SequentialSampler(test_ds), batch_size=batch_size, num_workers=32)

    model = model_class(config)

    model = model.to(device)

    if parallel:
        model_for_train = nn.DataParallel(model)
    else:
        model_for_train = model
    records = defaultdict(list)
    # records['fold'] = fold

    savers = []
    for saver_init in savers_init:
        savers.append(ModelSaver(model, f'models/{name}', records, saver_init[0], saver_init[1]))
    savers.append(ModelSaverOneEpoch(model, f'models/{name}', records, epoch=num_train_epochs - 1))
    if train:

        optimizer = optimizer_init(model)
        scheduler = scheduler_init(optimizer)


        for epoch in range(num_train_epochs):
            with Benchmark(f'Epoch {epoch}'):
                records['epoch'] = epoch
                clear_records_epoch(records)

                model_for_train.train()
                with tqdm(train_dl, leave=False, file=sys.stdout) as t:
                    for batch in t:
                        img = batch['img'].to(device)
                        label = [l.to(device) for l in batch['label']]
                        optimizer.zero_grad()
                        y, loss, _ = model_for_train(img, label)
                        loss_bp = torch.mean(loss)
                        loss_bp.backward()
                        optimizer.step()
                        t.set_postfix(loss=float(loss_bp))
                        label = [l[~torch.isnan(l)] for l in label]
                        y = [yy[~torch.isnan(l)] for yy, l in zip(y, label)]

                        records['train-loss-list'].append([float(x) for x in loss])
                        records['train-y_true-list'].append([x.detach().cpu() for x in label])
                        records['train-y_pred-list'].append([x.detach().cpu() for x in y])
                        
                model_for_train.eval()
                with torch.no_grad():
                    with tqdm(valid_dl, leave=False, file=sys.stdout) as t:
                        for batch in t:
                            img = batch['img'].to(device)
                            label = [l.to(device) for l in batch['label']]
                            optimizer.zero_grad()
                            y, loss, _ = model_for_train(img, label)
                            label = [l[~torch.isnan(l)] for l in label]
                            y = [yy[~torch.isnan(l)] for yy, l in zip(y, label)]

                            records['valid-loss-list'].append([float(x) for x in loss])
                            records['valid-y_true-list'].append([x.cpu() for x in label])
                            records['valid-y_pred-list'].append([x.cpu() for x in y])
                to_print = []
                for processor in processors:
                    key, value = processor(records)
                    records[key] = value
                    to_print.append(f'{key}={value:.4f}')

                print(f'Epoch {epoch}: ' + ', '.join(to_print))

                scheduler.step()
                for saver in savers:
                    saver.step()


    model.predict = True
    print('test')
    for saver in savers:
        saver.load()
    
        test_df_tmp = test_df.copy()
        preds = [[] for _ in range(len(n_classes))]
        model_for_train.eval()

        with torch.no_grad():
            with tqdm(test_dl, leave=False, file=sys.stdout) as t:
                for batch in t:
                    img = batch['img'].to(device)
                    y = model_for_train(img)
                    for i in range(len(preds)):
                        if n_classes[i] != 1:
                            y[i] = torch.softmax(y[i], dim=-1)
                        preds[i].append(y[i].cpu())
        for i in range(len(preds)):
            preds[i] = torch.cat(preds[i], dim=0).numpy()
            if n_classes[i] == 1:
                test_df_tmp[f'{label_cols[i]}_prob_0'] = preds[i]
            else:
                for j in range(1, n_classes[i]):
                    test_df_tmp[f'{label_cols[i]}_prob_{j}'] = preds[i][:, j]
        results[saver.key].append(test_df_tmp)
        ensure_path(f'output/{task_path}/results/{saver.key}')
        if config['external_test']:
            test_df_tmp.to_csv(f'output/{task_path}/results/{saver.key}/mt_fold_ex_d56_external.csv', index=False)
        else:
            test_df_tmp.to_csv(f'output/{task_path}/results/{saver.key}/mt_fold_internal_test.csv', index=False)
    model.predict = False
    model_for_train.to(cpu)
    model.to(cpu)

