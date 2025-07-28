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
import os
import copy

def run(round, idx, config, model_for_train, global_model_for_train, device=torch.device('cuda')):
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
    dataset_class = config['dataset_class']
    
    task_path = f'{task}/exp-{id_base}'
    task_file_path = f'output/{task_path}/tasks/client{idx}/data.csv'
    pkl_dump(config, f'output/{task_path}/tasks/client{idx}/config.pkl')

    n_classes = config['n_classes']
    label_cols = config['label_cols']

    df = pd.read_csv(task_file_path, low_memory=False)
    
    print(f'task={task}, id_base={id_base}, client={idx}')
    name = f"{task}-exp-{id_base}-client{idx}"

    # train_df = df[df['dataset'].isin([0, 1, 2, 3, 4])].copy()
    # valid_df = df[df['dataset'] == 5].copy()
    # test_df = df[df['dataset'] == 6].copy()
    # train_df = df[df['dataset'] == 'train'].copy()
    # valid_df = df[df['dataset'] == 'valid'].copy()
    # test_df = df[df['dataset'] == 'test'].copy()

    train_df = df[df['dataset'].isin([0, 1, 2])].copy()
    valid_df = df[df['dataset'] == 3].copy()
    test_df = df[df['dataset'] == 4].copy()

    train_ds = dataset_class(train_df, 'train', config)
    valid_ds = dataset_class(valid_df, 'valid', config)
    test_ds = dataset_class(test_df, 'test', config)

    train_dl = DataLoader(train_ds, sampler=RandomSampler(train_ds), batch_size=batch_size, num_workers=32)
    valid_dl = DataLoader(valid_ds, sampler=SequentialSampler(valid_ds), batch_size=batch_size, num_workers=32)
    test_dl = DataLoader(test_ds, sampler=SequentialSampler(test_ds), batch_size=batch_size, num_workers=32)

    records = defaultdict(list)

    if train:

        optimizer = optimizer_init(model_for_train)
        scheduler = scheduler_init(optimizer)

        for epoch in range(num_train_epochs):
            with Benchmark(f'Round {round}, Epoch {epoch}'):
                records['epoch'] = round
                clear_records_epoch(records)

                model_for_train.train()
                with tqdm(train_dl, leave=False, file=sys.stdout) as t:
                    for batch in t:
                        # print('img-shape', batch['img'].shape)
                        img = batch['img'].to(device)
                        label = [l.to(device) for l in batch['label']]
                        optimizer.zero_grad()
                        y, loss, _ = model_for_train(img, label)
                        # fedprox
                        proximal_term = 0.0
                        for w, w_t in zip(model_for_train.parameters(), global_model_for_train.parameters()):
                            proximal_term += (w - w_t).norm(2)
                        loss_bp = torch.mean(loss) + (0.1 / 2) * proximal_term
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
                print(f'Round {round}, Epoch {epoch}: ' + ', '.join(to_print))

                scheduler.step()
                
                if os.path.exists(f'models/{name}/best_valid-loss.pkl'):
                    best_valid_loss = pkl_load(f'models/{name}/best_valid-loss.pkl')
                    records[f'best_valid-loss'] = best_valid_loss
                else:
                    records[f'best_valid-loss'] = float('inf')

                savers = []
                for saver_init in savers_init:
                    savers.append(FedModelSaver(model_for_train, f'models/{name}', records, saver_init[0], saver_init[1]))
                savers.append(ModelSaverOneEpoch(model_for_train, f'models/{name}', records, epoch=config['rounds'] - 1))
                for saver in savers:
                    saver.step()
        
        return model_for_train.state_dict(), len(train_df)
    
    else:
        print('test')
        model_for_train.module.predict = True
        for pth in os.listdir(f'models/0124-ex/unseen_external/fed_emb_pretrain_prox_resnet50_raw-exp-pid'):
            print(pth)
            if '.pkl' in pth:
                continue
            model_for_train.load_state_dict(torch.load(f'models/0124-ex/unseen_external/fed_emb_pretrain_prox_resnet50_raw-exp-pid/{pth}', torch.device('cpu')))
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
            name_without_extension = os.path.splitext(os.path.basename(pth))[0]
            ensure_path(f'output/{task_path}/results/{name_without_extension}')
            test_df_tmp.to_csv(f'output/{task_path}/results/{name_without_extension}/internal_client{idx}.csv', index=False)


def run_val(round, idx, config, model_for_train, device=torch.device('cuda')):

    label_info_test = json_load(f'output/0124-ex/unseen_external/fed_emb_pretrain_prox_resnet50_raw/exp-pid/tasks/label_info.json')
    config_val = copy.deepcopy(config)
    config_val['task'] = '0124-ex/unseen_external/fed_emb_pretrain_prox_resnet50_raw'
    config_val['processors'] = [valid_loss] + [partial(valid_auc, n_class=n_class, record_index=i) for i, n_class in enumerate(label_info_test['n_classes'])]
    config_val = {**config_val, **label_info_test}
    task = config_val['task']
    id_base = config_val['id_base']
    processors = config_val['processors']
    savers_init = config_val['savers_init']

    train = config_val['train']
    batch_size = config_val['batch_size']
    dataset_class = config_val['dataset_class']
    
    task_path = f'{task}/exp-{id_base}'
    task_file_path = f'output/{task_path}/tasks/data.csv'
    pkl_dump(config_val, f'output/{task_path}/tasks/config.pkl')

    df = pd.read_csv(task_file_path, low_memory=False)
    
    print(f'task={task}, id_base={id_base}')
    name = f"{task}-exp-{id_base}"

    valid_df = df.copy()
    valid_ds = dataset_class(valid_df, 'valid', config_val)
    valid_dl = DataLoader(valid_ds, sampler=SequentialSampler(valid_ds), batch_size=batch_size, num_workers=32)

    records = defaultdict(list)

    if train:

        records['epoch'] = round
        model_for_train.eval()
        with torch.no_grad():
            with tqdm(valid_dl, leave=False, file=sys.stdout) as t:
                for batch in t:
                    img = batch['img'].to(device)
                    label = [l.to(device) for l in batch['label']]
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
        print(f'Round {round}, '.join(to_print))
        
        if os.path.exists(f'models/{name}/best_valid-loss.pkl'):
            best_valid_loss = pkl_load(f'models/{name}/best_valid-loss.pkl')
            records[f'best_valid-loss'] = best_valid_loss
        else:
            records[f'best_valid-loss'] = float('inf')

        savers = []
        for saver_init in savers_init:
            savers.append(FedModelSaver(model_for_train, f'models/{name}', records, saver_init[0], saver_init[1]))
        savers.append(ModelSaverOneEpoch(model_for_train, f'models/{name}', records, epoch=config_val['rounds'] - 1))
        for saver in savers:
            saver.step()
        
        return model_for_train.state_dict()
    

def run_ex_test(file_name, round, idx, config, model_for_train, device=torch.device('cuda')):

    task = config['task']
    id_base = config['id_base']

    batch_size = config['batch_size']
    dataset_class = config['dataset_class']
    
    task_path = f'{task}/exp-{id_base}'
    task_file_path = f'output/{task_path}/tasks/data.csv'
    pkl_dump(config, f'output/{task_path}/tasks/config.pkl')

    n_classes = config['n_classes']
    label_cols = config['label_cols']

    df = pd.read_csv(task_file_path, low_memory=False)
    
    print(f'task={task}, id_base={id_base}, client={idx}')
    name = f"{task}-exp-{id_base}"

    test_df = df.copy()
    test_ds = dataset_class(test_df, 'test', config)
    test_dl = DataLoader(test_ds, sampler=SequentialSampler(test_ds), batch_size=batch_size, num_workers=32)

    model_for_train.module.predict = True
    for pth in os.listdir(f'models/0124-ex/unseen_external/fed_emb_pretrain_prox_resnet50_raw-exp-pid'):
        print(pth)
        if '.pkl' in pth:
            continue
        model_for_train.load_state_dict(torch.load(f'models/0124-ex/unseen_external/fed_emb_pretrain_prox_resnet50_raw-exp-pid/{pth}', torch.device('cpu')))
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
        name_without_extension = os.path.splitext(os.path.basename(pth))[0]

        if config['external_test']:
            ensure_path(f'output/0124/fed_emb_pretrain_prox_resnet50_raw/exp-pid/results/{name_without_extension}')
            test_df_tmp.to_csv(f'output/0124/fed_emb_pretrain_prox_resnet50_raw/exp-pid/results/{name_without_extension}/client_ex_{file_name}_{idx}.csv', index=False)
        else:
            ensure_path(f'output/{task_path}/results/{name_without_extension}')
            test_df_tmp.to_csv(f'output/{task_path}/results/{name_without_extension}/client{idx}.csv', index=False)