import os
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'


from pathlib import Path
import sys
sys.path.append(str(Path('.').absolute()))
import copy
from mains.embryo_pretrain.embryo_image_model import EmbryoImageModel
from mains.embryo_pretrain.embryo_image_dataset import EmbryoImageDataset
from mains.embryo_pretrain.embryo_image_dataset_aug import EmbryoImageDatasetAug
from record_processors import *
from mains.embryo_pretrain.federated_runner import *

def optimizer_init(model):
    return Adam([
        {'params': model.module.backbone_parameters(), 'lr': 1e-4},
        {'params': model.module.head_parameters(), 'lr': 1e-3}
    ], weight_decay=1e-5)

def scheduler_init(optimizer):
    return lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)

def average_weights(w):
    w_avg = copy.deepcopy(w)
    for key in w[0].keys():
        for i in range(1, len(w)):
            w_avg[0][key] += w[i][key]
        w_avg[0][key] = torch.div(w_avg[0][key], len(w))
        for i in range(1, len(w)):
            w_avg[i][key] = w_avg[0][key]
    return w_avg

def average_dynamic_weights(w, lambda_weight, index):
    print(f'dynamic averaging: round{index}')
    w_avg = copy.deepcopy(w)
    for key in w[0].keys():
        w_avg[0][key] = w_avg[0][key] * lambda_weight[0, index]
        for i in range(1, len(w)):
            w_avg[0][key] += w[i][key] * lambda_weight[i, index]
        for i in range(1, len(w)):
            w_avg[i][key] = w_avg[0][key]
    return w_avg


def average_weights_by_sample(w, nums):
    w_avg = copy.deepcopy(w)
    for key in w[0].keys():
        w_avg[0][key] = w_avg[0][key] * nums[0]
        for i in range(1, len(w)):
            w_avg[0][key] += w[i][key] * nums[i]
        w_avg[0][key] = torch.div(w_avg[0][key], sum(nums))
        for i in range(1, len(w)):
            w_avg[i][key] = w_avg[0][key]
    return w_avg

if __name__ == '__main__':
    #if run live-birth outcome, find ‘fed_emb_pretrain_resnet50_raw’, then, replace 'emb_pretrain' with 'outcomes'

    # clients
    clients = []
    len_client = len(os.listdir(f'output/0124/fed_emb_pretrain_resnet50_raw/exp-pid/tasks/'))
    for idx in range(len_client):
        label_info = json_load(f'output/0124/fed_emb_pretrain_resnet50_raw/exp-pid/tasks/client{idx}/label_info.json')
        config = {
            'task': '0124/fed_emb_pretrain_resnet50_raw',
            'id_base': 'pid',
            'processors': [train_loss, valid_loss] + [partial(valid_auc, n_class=n_class, record_index=i) for i, n_class in enumerate(label_info['n_classes'])],
            'savers_init': [('valid-loss', min)],

            'train': False,
            'batch_size': 160,
            'num_train_epochs': 1,
            'parallel': True,
            'dataset_class': EmbryoImageDatasetAug,
            'model_class': EmbryoImageModel,
            'optimizer_init': optimizer_init,
            'scheduler_init': scheduler_init,
            
            'image_root': Path('/home/gtr21/Embryo_/data5/embryousr/share/embryo/data/embryo课题整理/crop_unzip'),
            'reshape_size': 400,
            'crop_size': 400,
            'arch': 'resnet50',
            'rounds': 15,
            'multimodal': False,
            'inter_dwa':False,
            'external_test': False
        }
        config = {**config, **label_info}
        clients.append(config)
    
    # rounds
    ROUNDS = config['rounds']

    # models
    local_model_list = []
    for idx in range(len_client):

        model_class = clients[idx]['model_class']
        model = model_class(clients[idx])
        model = model.to(device=torch.device('cuda'))
        if clients[idx]['parallel']:
            model_for_train = nn.DataParallel(model)
        else:
            model_for_train = model
        
        local_model_list.append(model_for_train)
    
    if clients[0]['train']:
        print('starting training')

        # inter—dwa
        avg_cost = np.zeros([ROUNDS, len_client], dtype=np.float32)
        lambda_weight = np.ones([len_client, ROUNDS])
        task_weight = [None for _ in range(len_client)] 
        # federated training
        for i in range(ROUNDS):
            if config['inter_dwa']:
                print('inter-dwa mode')
                # inter-dwa
                if i == 0 or i ==1:
                    lambda_weight[:, i] = 1.0
                else:
                    for j in range(len(task_weight)):
                        print(f'dy loss: round{i} client{j} last avg_cost is {avg_cost[i - 1, j]}, last last avg_cost is {avg_cost[i - 2, j]}')
                        task_weight[j] = avg_cost[i - 1, j] / avg_cost[i - 2, j]
                        print(f'dy loss: round{i} client{j} task_weight is{task_weight[j]}')
                    for j in range(len(task_weight)):
                        lambda_weight[j, i] = np.exp(task_weight[j] / 4) / sum([np.exp(task_weight[k] / 4) for k in range(len(task_weight))])
                        print(f'dy loss: round{i} client{j} lambda_weight is{lambda_weight[j, i]}')
                # Train models
                local_weights = []
                local_loss = []
                for idx in range(len_client):
                    w, loss = run(i, idx, clients[idx], model_for_train=copy.deepcopy(local_model_list[idx]))
                    local_weights.append(copy.deepcopy(w))
                    local_loss.append(loss)
                print('collect local loss', local_loss)
                # Aggregate the model
                if i == 0 or i ==1:
                    local_weights_list = average_weights(local_weights)
                else:
                    local_weights_list = average_dynamic_weights(local_weights, lambda_weight, i)
                avg_cost[i, :len_client] += np.array(local_loss)

                if clients[0]['external_test']:
                    print('external dataset validating')
                    model_val = copy.deepcopy(local_model_list[0])
                    model_val.load_state_dict(local_weights_list[0], strict=True)
                    run_val(i, 0, clients[0], model_for_train=copy.deepcopy(model_val))
                    
                for idx in range(len_client):
                    local_model = copy.deepcopy(local_model_list[idx])
                    local_model.load_state_dict(local_weights_list[idx], strict=True)
                    local_model_list[idx] = local_model
                    
                print("<Round {} finished>".format(i))
            else:
                print('normal mode')
                # Train models
                local_weights = []
                nums = []
                for idx in range(len_client):
                    w, num = run(i, idx, clients[idx], model_for_train=copy.deepcopy(local_model_list[idx]))
                    local_weights.append(copy.deepcopy(w))
                    nums.append(copy.deepcopy(num))
                # Aggregate the model
                local_weights_list = average_weights_by_sample(local_weights, nums)
                
                if clients[0]['external_test']:
                    print('external dataset validating')
                    model_val = copy.deepcopy(local_model_list[0])
                    model_val.load_state_dict(local_weights_list[0], strict=True)
                    run_val(i, 0, clients[0], model_for_train=copy.deepcopy(model_val))

                for idx in range(len_client):
                    local_model = copy.deepcopy(local_model_list[idx])
                    local_model.load_state_dict(local_weights_list[idx], strict=True)
                    local_model_list[idx] = local_model
                    
                print("<Round {} finished>".format(i))
    
    else:
        print('starting testing')
        # federated test
        if clients[0]['external_test']:
            print('external dataset testing')
            for file in os.listdir(f'output/0124-ex'):
                print(f'testing {file}')
                # if file == 'unseen_external':
                #     continue
                label_info_test = json_load(f'output/0124-ex/{file}/fed_emb_pretrain_resnet50_raw/exp-pid/tasks/label_info.json')
                for idx in range(len_client):
                    clients[idx]['task'] = f'0124-ex/{file}/fed_emb_pretrain_resnet50_raw'
                    clients[idx] = {**clients[idx], **label_info_test}
                    run_ex_test(file, 1, idx, clients[idx], model_for_train=copy.deepcopy(local_model_list[idx]))
        else:
            for idx in range(len_client):
                run(1, idx, clients[idx], model_for_train=copy.deepcopy(local_model_list[idx]))
        
        print('finished')

