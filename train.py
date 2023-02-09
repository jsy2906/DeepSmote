import random
import os
import time, datetime
import numpy as np
import wandb
import gc
import matplotlib.pyplot as plt

import torch
import torchmetrics
import kornia
from torch.cuda.amp import GradScaler

from data_loader import prepare_dataloader
from model import Encoder, Decoder
from one_epoch import train_one_epoch, valid_one_epoch
from visualize import wandb_im, vis_img

cfg = {
    'seed': 0,
    'img_size': 224,
    'latent_size' : 1000,
    'dim' : 64,
    'epochs': 500,
    'train_bs': 25,
    'valid_bs': 25,
    'lr': 1e-4,
    'weight_decay':1e-5,
    'num_workers' : 0
    'device': 'cuda:0'
}


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
    
def wandb_train(cfg, enc_dir, dec_dir, project, name, 
                train_df, valid_df, noise_prob=0.2, use = False):
    
    if not os.path.isdir(enc_dir): os.makedirs(enc_dir)
    if not os.path.isdir(dec_dir): os.makedirs(dec_dir)


    if use:
        wandb.login()
        wandb.init(project=project)
        wandb.config.update(cfg)
        wandb.run.name = name
        wandb.define_metric('Train Similarity', step_metric='epoch')
        wandb.define_metric('Train Loss', step_metric='epoch')
        wandb.define_metric('Valid Similarity', step_metric='epoch')
        wandb.define_metric('Valid Loss', step_metric='epoch')
        wandb.define_metric('Train Truth vs Prediction', step_metric='epoch')
        wandb.define_metric('Valid Truth vs Prediction', step_metric='epoch')


    seed_everything(cfg['seed'])

    print(f'Training start with Epochs {cfg["epochs"]} \n')

    print('Train:', len(train_df), '| Valid:', len(valid_df))
    train_loader, valid_loader = prepare_dataloader(cfg['img_size'], train_df, valid_df,
                                                trn_root=train_df.dir.values, val_root=valid_df.dir.values, num_workers=cfg['num_workers'])

    device = cfg['device']

    encoder = Encoder(cfg['latent_size'], cfg['dim'], cfg['img_size']).to(device)
    decoder = Decoder(cfg['latent_size'], cfg['dim'], cfg['img_size']).to(device)

    
    loss_tr = kornia.losses.MS_SSIMLoss(compensation=10, reduction='mean', data_range = cfg['train_bs']).to(device)
    loss_fn = kornia.losses.MS_SSIMLoss(compensation=10, reduction='mean', data_range = cfg['valid_bs']).to(device)

    ssim = torchmetrics.functional.structural_similarity_index_measure
    scaler = GradScaler()   
    enc_optimizer = torch.optim.Adam(encoder.parameters(), lr=cfg['lr'], 
                                         weight_decay=cfg['weight_decay'] 
                                )
    dec_optimizer = torch.optim.Adam(decoder.parameters(), lr=cfg['lr'], 
                                         weight_decay=cfg['weight_decay'] 
                                )

    enc_scheduler = torch.optim.lr_scheduler.StepLR(enc_optimizer, gamma=0.1, step_size=25)
    dec_schduler = torch.optim.lr_scheduler.StepLR(dec_optimizer, gamma=0.1, step_size=25)


    best_ssim = 0
    train_loss_list = []
    train_ssim_list = []
    valid_loss_list = []
    valid_ssim_list = []

    start = time.time()
    for epoch in range(cfg['epochs']):

        train_loss, train_ssim, train_img_list, train_label_list, train_pred_list \
            = train_one_epoch(use, epoch, encoder, decoder, loss_tr, ssim, enc_optimizer, dec_optimizer, 
                                     scaler, train_loader, device, 
#                                          scheduler=[enc_scheduler, dec_schduler],
                                                                  schd_batch_update=False, noise=noise_prob)

        if epoch%10==0: 
            vis_img(train_img_list, train_label_list, train_pred_list)
        elif epoch%50 == 0 and use==True:
            wandb_im(epoch, 'Train', train_img_list, train_label_list, train_pred_list, train_ssim, train_loss)
        elif epoch == cfg['epochs']-1: 
            vis_img(train_img_list, train_label_list, train_pred_list)
            if use: wandb_im(epoch, 'Train', train_img_list, train_label_list, train_pred_list, train_ssim, train_loss)

        with torch.no_grad():
            valid_loss, valid_ssim, valid_img_list, valid_label_list, valid_pred_list \
                = valid_one_epoch(use, epoch, encoder, decoder, loss_fn, ssim, valid_loader, device, 
#                                              scheduler=[enc_scheduler, dec_schduler], 
                                                                          schd_loss_update=False)
            if use:
                wandb.log({'epoch':epoch, 'Train Loss':train_loss, 'Train Similarity':train_ssim})
                wandb.log({'epoch':epoch, 'Valid Loss':valid_loss, 'Valid Similarity':valid_ssim})
            if epoch%10==0: 
                vis_img(valid_img_list, valid_label_list, valid_pred_list)
            elif epoch%50 == 0 & use==True:
                wandb_im(epoch, 'Valid', valid_img_list, valid_label_list, valid_pred_list, valid_ssim, valid_loss)
            elif epoch == cfg['epochs']-1 : 
                vis_img(valid_img_list, valid_label_list, valid_pred_list)
                if use: wandb_im(epoch, 'Valid', valid_img_list, valid_label_list, valid_pred_list, valid_ssim, valid_loss)

        torch.cuda.empty_cache()
        gc.collect()

        print(f'Train Loss : {train_loss:.5f} | Similarity : {train_ssim:.5f}')
        print(f'Valid Loss : {valid_loss:.5f} | Similarity : {valid_ssim:.5f}')
        train_loss_list.append(train_loss)
        train_ssim_list.append(train_ssim)
        valid_loss_list.append(valid_loss)
        valid_ssim_list.append(valid_ssim)

        if valid_ssim > best_ssim:
            print('Save the Best Model')
            path_enc = f'{enc_dir}best_enc.pth'
            path_dec = f'{dec_dir}best_dec.pth'

            torch.save(encoder.state_dict(), path_enc)
            torch.save(decoder.state_dict(), path_dec)

            best_ssim = valid_ssim

        plt.plot(train_loss_list, label='Train loss')
        plt.plot(train_ssim_list, label = 'Train Similarity')
        plt.plot(valid_loss_list, label='Valid loss')
        plt.plot(valid_ssim_list, label = 'Valid Similarity')
        plt.title('Train-Valid AE Loss & Similarity')
        plt.ylim(-0.2, 1.2)
        plt.legend()
        plt.show()
        

    end = time.time() - start
    time_ = str(datetime.timedelta(seconds=end)).split(".")[0]
    print("time :", time_)
    best_idx = valid_loss_list.index(min(valid_loss_list))
    print(f'Best Eopch : {best_idx}')
    print(f'Train Best Loss : {train_loss_list[best_idx]:.5f} | Best Similarity : {train_ssim_list[best_idx]:.5f}' )
    print(f'Valid Best Loss : {valid_loss_list[best_idx]:.5f} | Best Similarity : {valid_ssim_list[best_idx]:.5f}' )

    path_enc = enc_dir+'final_enc.pth'
    path_dec = dec_dir+'final_dec.pth'
    torch.save(encoder.state_dict(), path_enc)
    torch.save(decoder.state_dict(), path_dec)

    torch.cuda.empty_cache()
    gc.collect()
    del encoder, decoder, loss_tr, loss_fn, enc_optimizer, dec_optimizer, train_loader, valid_loader, scaler
    #     del encoder, decoder, loss_tr, enc_optimizer, dec_optimizer, train_loader, valid_loader, scaler, enc_scheduler, dec_schduler
    
    return train_loss, train_ssim, train_loss_list, train_ssim_list, \
            valid_loss, valid_ssim, valid_loss_list, valid_ssim_list, time_, \
            train_img_list, train_label_list, train_pred_list, \
            valid_img_list, valid_label_list, valid_pred_list
