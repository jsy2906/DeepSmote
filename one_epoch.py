import torch
from torch.cuda.amp import autocast, GradScaler

import wandb
from tqdm import tqdm
import numpy as np


cfg = {
    'accum_iter': 2, # suppoprt to do batch accumulation for backprop with effectively larger batch size
    'verbose_step': 1,
    }


def add_noise(img, prob=0.1):
    noise = torch.randn(img.size()) * prob
    noisy_img = img + noise
    return noisy_img
  

def train_one_epoch(wandb_use, epoch, encoder, decoder, loss_fn, ssim,
                    enc_optimizer, dec_optimizer, scaler,
                    train_loader, device, scheduler=None, schd_batch_update=False, noise=None):
    
    encoder.train()
    decoder.train()
    torch.autograd.set_detect_anomaly(True)
    
    if wandb_use: wandb.watch(encoder, criterion=loss_fn, log='all')
    if wandb_use: wandb.watch(decoder, criterion=loss_fn, log='all') 
    
    t = time.time()
    running_loss = 0.0
    loss_sum = 0
    ssim_score = 0
    sample_num = 0
    
    img_list = []
    label_list = []
    pred_list = []

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for step, (imgs, labels) in pbar:
        if noise is not None: 
            imgs = imgs.float()
            noises = add_noise(imgs, prob=noise).to(device).float()
        else:
            imgs = imgs.float().to(device)
        img_list.append(imgs.detach().cpu().numpy())
        label_list.append(labels.numpy())
        
        with autocast():
            if noise:
                en_out = encoder(noises)
            else:
                en_out = encoder(imgs)
            de_out = decoder(en_out)
            pred_list.append(de_out.detach().cpu().numpy().astype(np.float32))


            de_out = de_out.type(torch.float32)
            if noise: imgs = imgs.to(device)
            loss = loss_fn(de_out, imgs)

            similarity = ssim(de_out, imgs)
            ssim_score += similarity.item()

            scaler.scale(loss).backward()

            running_loss += (loss.item() * imgs.shape[0])
            loss_sum += loss.item()
            sample_num += imgs.shape[0]

            if ((step + 1) %  cfg['accum_iter'] == 0) or ((step + 1) == len(train_loader)):
                # may unscale_ here if desired (e.g., to allow clipping unscaled gradients)
                # torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5) 
                # torch.nn.utils.clip_grad_norm_(decoder.parameters(), 5)

                scaler.step(enc_optimizer)
                scaler.step(dec_optimizer)

                scaler.update()

                enc_optimizer.zero_grad()
                dec_optimizer.zero_grad()
                
                if scheduler is not None and schd_batch_update:
                    scheduler[0].step()
                    scheduler[1].step()

            if ((step + 1) % cfg['verbose_step'] == 0) or ((step + 1) == len(train_loader)):
                description = f'Train epoch {epoch} | Loss: {running_loss/(sample_num+1e-100):.5f}'
                pbar.set_description(description)
                
            del imgs, loss
            
    train_loss =  loss_sum / (len(train_loader)+1e-100)
    train_ssim = ssim_score / (len(train_loader)+1e-100)
    
    if scheduler is not None and not schd_batch_update:
        scheduler[0].step()
        scheduler[1].step()
    
    return train_loss, train_ssim, img_list, label_list, pred_list
  

  
def valid_one_epoch(wandb_use, epoch, encoder, decoder, loss_fn, ssim,
                    val_loader, device, scheduler=None, schd_loss_update=False):
    
    
    encoder.eval()
    decoder.eval()
    torch.autograd.set_detect_anomaly(True)
    
    if wandb_use: wandb.watch(encoder, criterion=loss_fn, log='all')
    if wandb_use: wandb.watch(decoder, criterion=loss_fn, log='all') 
    
    t = time.time()
    running_loss = 0.0
    loss_sum = 0
    ssim_score = 0
    sample_num = 0
    
    img_list = []
    label_list = []
    pred_list = []
    
    pbar = tqdm(enumerate(val_loader), total=len(val_loader))
    for step, (imgs, labels) in pbar:
        imgs = imgs.to(device).float()

        img_list.append(imgs.detach().cpu().numpy())
        label_list.append(labels.numpy())
        
        en_out = encoder(imgs)
        de_out = decoder(en_out)
        pred_list.append(de_out.detach().cpu().numpy().astype(np.float32))
        
        
        de_out = de_out.type(torch.float32)
        loss = loss_fn(de_out, imgs)
        
        similarity = ssim(de_out, imgs)
        ssim_score += similarity.item()

        running_loss += loss.item()*imgs.shape[0]
        loss_sum += loss.item()
        sample_num += imgs.shape[0]  
        
        
        if ((step + 1) % cfg['verbose_step'] == 0) or ((step + 1) == len(val_loader)):
            description = f'Valid epoch {epoch} | Loss: {running_loss/(sample_num+1e-100):.5f}'
            pbar.set_description(description)

    valid_loss = loss_sum/(len(val_loader)+1e-100)
    valid_ssim = ssim_score/(len(val_loader)+1e-100)
    
    if scheduler is not None:
        if schd_loss_update:
            scheduler[0].step(valid_loss)
            scheduler[1].step(valid_loss)
        else:
            scheduler[0].step()
            scheduler[1].step()
    
    del imgs, loss
    
    return valid_loss, valid_ssim, img_list, label_list, pred_list
