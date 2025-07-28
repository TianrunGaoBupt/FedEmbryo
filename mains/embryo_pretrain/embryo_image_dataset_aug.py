from dataclasses import dataclass
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import torch
import pandas as pd
import torch
import numpy as np
from PIL import Image
from albumentations import *
from albumentations.pytorch import ToTensorV2


@dataclass
class EmbryoImageDatasetAug(Dataset):
    data: pd.DataFrame
    mode: str
    config: dict

    def __post_init__(self):
        self.image_root = self.config['image_root']
        self.reshape_size = self.config['reshape_size']
        self.crop_size = self.config['crop_size']
        self.label_cols = self.config['label_cols']

        self.trans = {
            'train': Compose([
                RandomResizedCrop(self.crop_size, self.crop_size, p=1.),
                Transpose(p=0.5),
                HorizontalFlip(p=0.5),
                VerticalFlip(p=0.5),
                ShiftScaleRotate(p=0.5),
                HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
                RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ], p=1.),
            'valid': Compose([
                SmallestMaxSize(self.reshape_size, p=1.),
                CenterCrop(self.crop_size, self.crop_size, p=1.),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ], p=1.),
            'test': Compose([
                SmallestMaxSize(self.reshape_size, p=1.),
                CenterCrop(self.crop_size, self.crop_size, p=1.),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ], p=1.),
        }

    def clinical_factor(self, idx):
        input_data1 = self.data.loc[self.data.index[idx], '女方年龄'].reshape((-1, 1))
        input_data2 = self.data.loc[self.data.index[idx], '男方年龄'].reshape((-1, 1))
        input_data3 = self.data.loc[self.data.index[idx], '不孕年限'].reshape((-1, 1))
        input_data4 = self.data.loc[self.data.index[idx], 'BMI'].reshape((-1, 1))
        input_data5 = self.data.loc[self.data.index[idx], 'FSH'].reshape((-1, 1))
        input_data6 = self.data.loc[self.data.index[idx], '自然流产'].reshape((-1, 1))
        input_data7 = self.data.loc[self.data.index[idx], '人工流产'].reshape((-1, 1))
        # input_data8 = self.data.loc[self.data.index[idx], 'primary infertility'].reshape((-1, 1))
        # input_data9 = self.data.loc[self.data.index[idx], 'Secondary infertility'].reshape((-1, 1))     				
        input_data10=self.data.loc[self.data.index[idx], 'AMH'].reshape((-1, 1))
        input_data11=self.data.loc[self.data.index[idx], '内膜厚度'].reshape((-1, 1))
        input_data12=self.data.loc[self.data.index[idx], '药物流产'].reshape((-1, 1))
        input_data13=self.data.loc[self.data.index[idx], '获卵数'].reshape((-1, 1))
        input_data14=self.data.loc[self.data.index[idx], '原发/继发不孕_原发'].reshape((-1, 1))
        input_data15=self.data.loc[self.data.index[idx], '原发/继发不孕_继发'].reshape((-1, 1))
        input_data16=self.data.loc[self.data.index[idx], '内膜形态_三线'].reshape((-1, 1))
        input_data17=self.data.loc[self.data.index[idx], '内膜形态_弱三线'].reshape((-1, 1))
        input_data18=self.data.loc[self.data.index[idx], '内膜形态_强回声'].reshape((-1, 1))
                        
        input_tensor = torch.tensor(np.concatenate((input_data1, input_data2, input_data3, input_data4,
        input_data5, input_data10, input_data11, input_data6, input_data12, input_data7, input_data13, input_data14, input_data15, input_data16, 
        input_data17, input_data18), axis=1), dtype=torch.float32)
        return input_tensor

    def read_image(self, path):
        img = np.array(Image.open(path).convert('RGB'))
        img = self.trans[self.mode](image=img)['image']
        return img

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.read_image(str(self.image_root / self.data.loc[self.data.index[idx], 'image_path']))
        if self.config['multimodal']:
            result = {
                'img': img,
                'img_path': str(self.image_root / self.data.loc[self.data.index[idx], 'image_path']),
                'label': [torch.tensor(self.data.loc[self.data.index[idx], col], dtype=torch.float) for col in self.label_cols],
                'clinical_factor': self.clinical_factor(idx)
            }
        else:
            result = {
                'img': img,
                'img_path': str(self.image_root / self.data.loc[self.data.index[idx], 'image_path']),
                'label': [torch.tensor(self.data.loc[self.data.index[idx], col], dtype=torch.float) for col in self.label_cols]
            }

        return result
