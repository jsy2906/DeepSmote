import random
import numpy as np
import wandb
import matplotlib.pyplot as plt


def wandb_im(class_, epoch, type_, img_list, label_list, pred_list, similarity, loss):
    
    cls = class_  # class_ is dict type
    idx = random.randint(0, len(img_list)-1)
    imgs = img_list[idx][:3, ...]
    labels = label_list[idx][:3, ...]
    preds = pred_list[idx][:3, ...]
    
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    dis_image=[]
    for i in range(imgs.shape[0]):
        or_im = imgs[i].transpose(1, 2, 0)
        or_im = (np.clip((std * or_im + mean), 0, 1)*255).astype(np.uint8)
        label = labels[i]
        
        pred = preds[i].transpose(1, 2, 0)
        pred = (np.clip((std * pred + mean), 0, 1)*255).astype(np.uint8)
        
        concat_ = np.concatenate((or_im, pred), axis=1)
        dis_image.append(concat_)

    wandb.log({'epoch':epoch,
            f'{type_} Truth vs Prediction' : [wandb.Image(np.vstack(dis_image), 
                                                caption=f'{type_} | Loss {loss:.5f} | Similarity {similarity:.5f}')]
        })
    
    
def vis_img(class_, img_list, label_list, pred_list):
    
    cls = class_  # class_ is dict type
    idx = random.randint(0, len(img_list)-1)
    imgs = img_list[idx][:3, ...]
    labels = label_list[idx][:3, ...]
    preds = pred_list[idx][:3, ...]
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
#     mean = np.array([0.5, 0.5, 0.5])
#     std = np.array([0.5, 0.5, 0.5])
    
    for i in range(imgs.shape[0]):
        or_im = imgs[i].transpose(1, 2, 0)
        or_im = (np.clip((std * or_im + mean), 0, 1)*255).astype(np.uint8)
        
        pred = preds[i].transpose(1, 2, 0)
        pred = (np.clip((std * pred + mean), 0, 1)*255).astype(np.uint8)
        
        label = labels[i]
        
        fig1 = plt.subplot(1, 2, 1, )
        fig1.imshow(or_im)
        fig1.set_title(f'Original {cls[label]}')
        
        fig2 = plt.subplot(1, 2, 2)
        fig2.imshow(pred)
        fig2.set_title('Prediction')
        
        plt.show()
