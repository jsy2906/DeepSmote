import cv2
import numpy as np

from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize, SmallestMaxSize
)
from albumentations.pytorch import ToTensorV2

from catalyst.data.sampler import BalanceClassSampler

cfg = {
    'train_bs': 25,
    'valid_bs': 25,
    'num_workers' : 0,
    
    'enc_bs': 10,
    }


def get_img(path, sub_path=None):
    im_bgr = cv2.imread(path)
    im_rgb = im_bgr[:, :, ::-1]

    return im_rgb

class AEDataset(Dataset):
    def __init__(self, df, data_root, 
                 transforms=None, 
                 output_label=True, 
                 one_hot_label=False):
        
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.data_root = data_root
        self.transforms = transforms
        
        self.output_label = output_label
        self.one_hot_label = one_hot_label
        
        if output_label == True:
            self.labels = self.df['label'].values
            
            if one_hot_label is True:
                self.labels = np.eye(self.df['label'].max()+1)[self.labels]
                
            
    def __len__(self):
        return self.df.shape[0]
    
    
    def __getitem__(self, index: int):
        
        # get labels
        if self.output_label:
            target = self.labels[index]
        
        img  = get_img("{}/{}".format(self.data_root[index], self.df.loc[index]['image_id']))
        
        
        if self.transforms:
            img = self.transforms(image=img)['image']
        
                            
        # do label smoothing
        if self.output_label == True:
            return img, target
        else:
            return img
          
 
######################## get transforms #######################


def get_train_transforms(image_size):
    return Compose([
            Resize(image_size, image_size),   # (h, w)
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            RandomRotate90(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)
   
def get_valid_transforms(image_size):
    return Compose([
            Resize(image_size, image_size),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)
  
  
######################### get dataloader #########################
 

def prepare_dataloader(image_size, train, valid, trn_batch=cfg['train_bs'], val_batch=cfg['valid_bs'],
                       trn_root=train.dir.values, val_root=valid.dir.values, num_workers=cfg['num_workers']):
    

    train_ds = AEDataset(train, trn_root, transforms=get_train_transforms(image_size), output_label=True, one_hot_label=False)
    valid_ds = AEDataset(valid, val_root, transforms=get_valid_transforms(image_size), output_label=True)
    
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=trn_batch,
        pin_memory=False,
        drop_last=False,
        shuffle=False,        
        num_workers=num_workers,
        sampler=BalanceClassSampler(labels=train['label'].values, mode="upsampling")
    )
    val_loader = torch.utils.data.DataLoader(
        valid_ds, 
        batch_size=val_batch,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=False,
    )
    return train_loader, val_loader



######################## get test dataloader ########################

def get_test_transforms(image_size):
    return Compose([
            Resize(image_size, image_size),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)


def prepare_test_dataloader(image_size, test, tst_batch=cfg['enc_bs'], num_workers=cfg['num_workers'], tst_root=test.dir.values,):
    
    test_ds = AEDataset(test, tst_root, transforms=get_test_transforms(image_size), output_label=True)
    
    tst_loader = torch.utils.data.DataLoader(
        test_ds, 
        batch_size=tst_batch,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=False,
    )
    return tst_loader
