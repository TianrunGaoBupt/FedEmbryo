import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"


from pathlib import Path
import sys
sys.path.append(str(Path('.').absolute()))

from mains.embryo_pretrain.embryo_image_model import EmbryoImageModel
from mains.embryo_pretrain.embryo_image_dataset import EmbryoImageDataset
from mains.embryo_pretrain.embryo_image_dataset_aug import EmbryoImageDatasetAug

# from mains.embryo_pretrain.federated_local_runner_multimodal import *
from mains.embryo_pretrain.federated_local_runner import *

def optimizer_init(model):
    return Adam([
        {'params': model.backbone_parameters(), 'lr': 1e-4},
        {'params': model.head_parameters(), 'lr': 1e-3}
    ], weight_decay=1e-5)


def scheduler_init(optimizer):
    return lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)


if __name__ == '__main__':
    #if run live-birth outcome, find ‘fed_emb_pretrain_resnet50_raw’, then, replace 'emb_pretrain' with 'outcomes'
    label_info = json_load('output/0124/fed_emb_pretrain_resnet50_raw/exp-pid/tasks/client0/label_info.json')
    config = {
        'task': '0124/fed_emb_pretrain_resnet50_raw',
        'id_base': 'pid',
        'processors': [train_loss, valid_loss] + [partial(train_auc, n_class=n_class, record_index=i) for i, n_class in enumerate(label_info['n_classes'])],
        'savers_init': [('valid-loss', min)],

        'train': False,
        'batch_size': 160,
        'num_train_epochs': 15,
        'parallel': True,
        'dataset_class': EmbryoImageDatasetAug,
        'model_class': EmbryoImageModel,
        'optimizer_init': optimizer_init,
        'scheduler_init': scheduler_init,

        'image_root': Path('/home/gtr21/Embryo_/data5/embryousr/share/embryo/data/embryo课题整理/crop_unzip'),
        'reshape_size': 400,
        'crop_size': 400,
        'arch': 'resnet50',
        'multimodal': False,
        'external_test': True
    }
    run(config)
    
    
