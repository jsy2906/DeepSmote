import os, cv2
import numpy as np

from test import generate_data


def save_result(class_, data_df, enc_dir, dec_dir, save_dir):
  
  result_x, result_y = generate_data(data_df, enc_dir, dec_dir)
  
  mean = np.array([0.485, 0.456, 0.406])
  std = np.array([0.229, 0.224, 0.225])

  if not os.path.isdir(save_dir): os.makedirs(save_dir)

  for i in tqdm(range(result_x.shape[0])):
      a = result_x[i].transpose(1, 2, 0)
      input_ = (np.clip((std * a + mean), 0, 1)*255).astype(np.uint8)
      chg_clr = cv2.cvtColor(input_, cv2.COLOR_BGR2RGB)
      
      
      if i >= len(data_df):
          cv2.imwrite(f'{save_dir}{class_[result_y[i]]}_{i}.jpg', chg_clr)
          
          
