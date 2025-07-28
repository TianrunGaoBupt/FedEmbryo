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
from mains.embryo_pretrain.federated_mlp_multimodal_runner import *

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

if __name__ == '__main__':

    # clients
    clients = []
    len_client = len(os.listdir(f'output/0124/fed_outcomes_mlp_reverse_multimodal_resnet50_raw/exp-pid/tasks/'))
    for idx in range(len_client):
        label_info = json_load(f'output/0124/fed_outcomes_mlp_reverse_multimodal_resnet50_raw/exp-pid/tasks/client{idx}/label_info.json')
        config = {
            'task': '0124/fed_outcomes_mlp_reverse_multimodal_resnet50_raw',
            'id_base': 'pid',
            'processors': [train_loss, valid_loss] + [partial(valid_auc, n_class=n_class, record_index=i) for i, n_class in enumerate(label_info['n_classes'])],
            'savers_init': [('valid-loss', min)],

            'train': False,
            'batch_size': 64,
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
            'rounds': 13,
            'multimodal': True,
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
        # federated training
        for i in range(ROUNDS):
            # Train models
            local_weights = []
            for idx in range(len_client):
                w,_ = run(i, idx, clients[idx], model_for_train=copy.deepcopy(local_model_list[idx]).cuda())
                local_weights.append(copy.deepcopy(w))
            # Aggregate the model
            local_weights_list = average_weights(local_weights)

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
                label_info_test = json_load(f'output/0124-ex/{file}/fed_outcomes_mlp_reverse_multimodal_resnet50_raw/exp-pid/tasks/label_info.json')
                for idx in range(len_client):
                    clients[idx]['task'] = f'0124-ex/{file}/fed_outcomes_mlp_reverse_multimodal_resnet50_raw'
                    clients[idx] = {**clients[idx], **label_info_test}
                    run_ex_test(file, 1, idx, clients[idx], model_for_train=copy.deepcopy(local_model_list[idx]))
        else:
            for idx in range(len_client):
                run(1, idx, clients[idx], model_for_train=copy.deepcopy(local_model_list[idx]).cuda())
        
        print('finished')

