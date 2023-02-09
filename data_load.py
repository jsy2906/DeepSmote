import cv2
import numpy as np



def get_img(path, sub_path=None):
    im_bgr = cv2.imread(path)
    im_rgb = im_bgr[:, :, ::-1]

    return im_rgb

class AEDataset(Dataset):
    def __init__(self, df, data_root, 
                 transforms=None, 
                 noise = False, prob=0.15,
                 output_label=True, 
                 one_hot_label=False):
        
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.data_root = data_root
        self.transforms = transforms
        self.noise = noise
        self.prob = prob
        
        self.output_label = output_label
        self.one_hot_label = one_hot_label
        
        if output_label == True:
            self.labels = self.df['label'].values
            #print(self.labels)
            
            if one_hot_label is True:
                self.labels = np.eye(self.df['label'].max()+1)[self.labels]
                #print(self.labels)
            
    def __len__(self):
        return self.df.shape[0]
    
    def plus_noise(self, image):
        output = image.copy()
        if len(image.shape) == 2:
            black = 0
            white = 255            
        else:
            colorspace = image.shape[2]
            if colorspace == 3:  # RGB
                black = np.array([0, 0, 0], dtype='uint8')
                white = np.array([255, 255, 255], dtype='uint8')
            else:  # RGBA
                black = np.array([0, 0, 0, 255], dtype='uint8')
                white = np.array([255, 255, 255, 255], dtype='uint8')
        probs = np.random.random(image.shape[:2])
        output[probs < (self.prob / 2)] = black
        output[probs > 1 - (self.prob / 2)] = white
        return output
    
    def __getitem__(self, index: int):
        
        # get labels
        if self.output_label:
            target = self.labels[index]
        
        img  = get_img("{}/{}".format(self.data_root[index], self.df.loc[index]['image_id']))
        
        
        if self.transforms:
            if self.noise:
                noise_img = self.plus_noise(img)
                noise_img = self.transforms(image=noise_img)
                img = self.transforms(image=img)['image']
            else:
                img = self.transforms(image=img)['image']
        
                            
        # do label smoothing
        if self.output_label == True:
            if self.noise:
                return img, noise_img, target
            else:
                return img, target
        else:
            return img
          
 
######################## get transforms #######################


from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize, SmallestMaxSize
)
from albumentations.pytorch import ToTensorV2

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
                       trn_root=train.dir.values, val_root=valid.dir.values,):
    
    from catalyst.data.sampler import BalanceClassSampler

    train_ds = AEDataset(train, trn_root, transforms=get_train_transforms(image_size), noise=False, output_label=True, )
    valid_ds = AEDataset(valid, val_root, transforms=get_valid_transforms(image_size), output_label=True)
    
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=trn_batch,
        pin_memory=False,
        drop_last=False,
        shuffle=False,        
        num_workers=0,
        sampler=BalanceClassSampler(labels=train['label'].values, mode="upsampling")
    )
    val_loader = torch.utils.data.DataLoader(
        valid_ds, 
        batch_size=val_batch,
        num_workers=0,
        shuffle=False,
        pin_memory=False,
    )
    return train_loader, val_loader
