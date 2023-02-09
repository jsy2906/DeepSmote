import torch.nn as nn


cfg = {
    'img_size': 224,
    'latent_size' : 1000,
    'dim' : 64,
    }


class Encoder(nn.Module):
    def __init__(self, latent, dim, img_size):
        super(Encoder, self).__init__()

        self.latent = cfg['latent_size']
        self.dim = cfg['dim']
        self.img_size = cfg['img_size']
        
        # convolutional filters, work excellent with image data
        self.conv1 = nn.Conv2d(3, self.dim, 4, 2, 2, bias=False)
        self.conv2 = nn.Conv2d(self.dim, self.dim * 2, 4, 2, 2, bias=False)
        self.conv3 = nn.Conv2d(self.dim * 2, self.dim * 4, 4, 2, 2, bias=False)
        self.conv4 = nn.Conv2d(self.dim * 4, self.dim * 8, 4, 2, 2, bias=False)
        
        self.batch = nn.BatchNorm2d(self.dim)
        self.batch1 = nn.BatchNorm2d(self.dim * 2)
        self.batch2 = nn.BatchNorm2d(self.dim * 4)
        self.batch3 = nn.BatchNorm2d(self.dim * 8)
        
#         self.flatten = layer.SelectAdaptivePool2d(pool_type='avg', flatten=nn.Flatten(start_dim=1, end_dim=-1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(self.dim * 8*(15**2), self.latent) if self.img_size == 224 \
                  else nn.Linear(self.dim * 8*(15**3), self.latent)   # if 224 else 384
        
        

    def forward(self, x):
        x = nn.LeakyReLU(0.1)(self.batch(self.conv1(x)))
        x = nn.LeakyReLU(0.1)(self.batch1(self.conv2(x)))
        x = nn.LeakyReLU(0.1)(self.batch2(self.conv3(x)))
        x = nn.LeakyReLU(0.1)(self.batch3(self.conv4(x)))
        x = self.flatten(x)
        x = nn.LeakyReLU(0.1)(self.fc(x))
        return x
    
    
class Decoder(nn.Module):
    def __init__(self, latent, dim, img_size):
        super(Decoder, self).__init__()

        self.latent = cfg['latent_size']
        self.dim = cfg['dim']
        self.img_size = cfg['img_size']

        # first layer is fully connected
        self.fc = nn.Linear(self.latent, self.dim * 8*(15**2)) if self.img_size == 224 \
                  else nn.Linear(self.latent, self.dim * 8*(15**3))   # if 224 else 384

        self.upsample1 = nn.ConvTranspose2d(self.dim * 8, self.dim * 4, 5, 2, 2)
        self.upsample2 = nn.ConvTranspose2d(self.dim * 4, self.dim * 2, 5, 2, 2)
        self.upsample3 = nn.ConvTranspose2d(self.dim * 2, self.dim, 5, 2, 2)
        self.upsample4 = nn.ConvTranspose2d(self.dim, 3, 4, 2, 2)

        self.batch1 = nn.BatchNorm2d(self.dim * 4)
        self.batch2 = nn.BatchNorm2d(self.dim * 2)
        self.batch3 = nn.BatchNorm2d(self.dim)

        
    def forward(self, x):
        x = nn.LeakyReLU(0.1)(self.fc(x))
        layer_size = int((x.shape[-1]/(self.dim*8))**(1/2))
        x = x.view(-1, self.dim * 8, layer_size, layer_size)
        x = nn.LeakyReLU(0.1)(self.batch1(self.upsample1(x)))
        x = nn.LeakyReLU(0.1)(self.batch2(self.upsample2(x)))
        x = nn.LeakyReLU(0.1)(self.batch3(self.upsample3(x)))
        x = nn.LeakyReLU(0.1)(self.upsample4(x))
        return x
