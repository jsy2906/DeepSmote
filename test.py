from tqdm import tqdm
import time, datetime
import gc
import numpy as np

import torch
import imblearn

from model import Encoder, Decoder
from data_load import prepare_test_dataloader

cfg = {
    'img_size': 224,
    'latent_size' : 1000,
    'dim' : 64,
    'epochs': 500,
    'enc_bs': 10,
    'dec_bs':10,
    'n_neigh' : 10,
    'num_workers': 0,
    'accum_iter': 2, # suppoprt to do batch accumulation for backprop with effectively larger batch size
    'verbose_step': 1,
    'device': 'cuda:0'
}


def generate_data(data_df, enc_dir, dec_dir):
  
  device = torch.device(cfg['device'])

  encoder = Encoder(cfg['latent_size'], cfg['dim'])
  encoder.load_state_dict(torch.load(f'{enc_dir}final_enc.pth', map_location=cfg['device']), strict=False)
  encoder = encoder.to(device)
  encoder.eval()

  t0 = time.time()

  test_loader = prepare_test_dataloader(cfg['img_size'], data_df, tst_batch=cfg['enc_bs'], num_workers=cfg['num_workers'], tst_root=data_df.dir)
  
  original = []
  en_output = []
  en_label = []

  pbar = tqdm(enumerate(test_loader), total=len(test_loader))
  for step, (images, labels) in pbar:

      images = images.to(device)

      encoder_x = encoder(images)
      en_output.append(encoder_x.detach().cpu().numpy())
      en_label.append(labels)
      original.append(images.detach().cpu().numpy())

      del images, encoder_x

      torch.cuda.empty_cache()
      gc.collect()
    
  del encoder
  
  
  en_output = np.vstack(en_output)
  en_label = np.hstack(en_label)
  
  sm = imblearn.over_sampling.SMOTE(random_state=0, k_neighbors=cfg['n_neigh'])
  xsm, ysm = sm.fit_resample(en_output, en_label)
  
  
  start = 0
  end = cfg['dec_bs']
  xsm_output = []
  ysm_output = []
  for idx in range(xsm.shape[0]//end+1):
      if idx == xsm.shape[0]//end+1:
          xsm_output.append(xsm[start:, ...])
          ysm_output.append(ysm[start:, ...])
      xsm_output.append(xsm[start:end, :])
      ysm_output.append(ysm[start:end, ...])
      start += cfg['dec_bs']
      end += cfg['dec_bs']
      
      

  decoder = Decoder(cfg['latent_size'], cfg['dim'])
  decoder.load_state_dict(torch.load(f'{dec_dir}final_dec.pth', map_location=cfg['device']), strict=False)
  decoder = decoder.to(device)
  decoder.eval()

  dec_output=[]
  dec_label = []
  for idx in tqdm(range(len(xsm))):

      images = torch.Tensor(xsm[idx]).to(device)
      decoder_x = decoder(images)
      dec_output.append(decoder_x.detach().cpu().numpy())
      dec_label.append(ysm[idx])

      del images, decoder_x

      torch.cuda.empty_cache()
      gc.collect()

  del decoder
  
  
  result_x = np.vstack(dec_output)
  result_y = np.hstack(dec_label)
  
  return result_x, result_y

