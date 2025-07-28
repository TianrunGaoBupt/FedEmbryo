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
from albumentations import *
from albumentations.pytorch import ToTensorV2
from utils_vis import GradCAM, show_cam_on_image, center_crop_img

def read_image(path):
    assert os.path.exists(path), "file: '{}' dose not exist.".format(path)
    img = np.array(Image.open(path).convert('RGB'))
    print(img.shape)
    data_transform = Compose([
                SmallestMaxSize(400, p=1.),
                CenterCrop(400, 400, p=1.),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ], p=1.)
    img_tensor = data_transform(image=img)['image']
    return img_tensor, cv2.resize(img, (400, 400))


def run(config, device=torch.device('cuda')):
    cpu = torch.device('cpu')
    task = config['task']
    id_base = config['id_base']

    savers_init = config['savers_init']


    train = config['train']

    num_train_epochs = config['num_train_epochs']
    parallel = config['parallel']

    model_class = config['model_class']
    task_path = f'{task}/exp-{id_base}'


    label_info = json_load(f'/home/gtr21/Embryo_/data5/wkai/projects/embryo_init/output/{task_path}/tasks/client0/label_info.json')
    config = {**config, **label_info}

    n_classes = config['n_classes']



    print(f'task={task}, id_base={id_base}, client0')


    model = model_class(config)
    model = model.to(device)

    if parallel:
        model_for_train = nn.DataParallel(model)
    else:
        model_for_train = model
    records = defaultdict(list)

    savers = []
    for saver_init in savers_init:
        savers.append(ModelSaver(model, f'/home/gtr21/Embryo_/data5/wkai/projects/embryo_init/models/0628/loc_emb_pretrain_resnet50_raw-exp-pid-client0', records, saver_init[0], saver_init[1]))
    savers.append(ModelSaverOneEpoch(model, f'/home/gtr21/Embryo_/data5/wkai/projects/embryo_init/models/0628/loc_emb_pretrain_resnet50_raw-exp-pid-client0', records, epoch=num_train_epochs - 1))
    if train:
        print('train')

    model.predict = True
    print('test')
    
        
    for saver in savers:
        saver.load() 
        # test_df_tmp = test_df.copy()
        preds = [[] for _ in range(len(n_classes))]
        model_for_train.eval()
        
        target_layers = [model_for_train.module.net.layer4]
        # img_tensor, img = read_image(str('/home/gtr21/Embryo_/data5/embryousr/share/embryo/data/embryo课题整理/crop_unzip/20200804-广州-PGT/images/PGD104688.1 D1 1-8.jpg'))
        img_tensor, img = read_image(str('/home/gtr21/Embryo_/data5/embryousr/share/embryo/data/embryo课题整理/crop_unzip/20200804-广州-PGT/images/PGS105999.1 D1 1-8.jpg'))
        # print('PGS105999.1 D1 1-4.jpg')
        print('PGS105999.1 D1 1-8.jpg')
        input_tensor = torch.unsqueeze(img_tensor.to(torch.device('cuda')), dim=0)

        cam = GradCAM(model=model_for_train, target_layers=target_layers, use_cuda=True)
        target_category =  1   # 281   tabby, tabby cat    21 kite    662 modem
        # target_category = 254  # pug, pug-dog

        grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                        grayscale_cam,
                                        use_rgb=True)
        plt.style.use('default')
        plt.imshow(visualization)
        plt.savefig('test111')
        plt.show()

    model.predict = False
    model_for_train.to(cpu)
    model.to(cpu) 

    print('done')
