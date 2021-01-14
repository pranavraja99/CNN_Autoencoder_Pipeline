#%%
import torch
import torchvision
import torch.nn as nn

class auto_encoder(nn.Module):
    def __init__(self):
        super(auto_encoder, self).__init__()

        self.encoder=nn.Sequential(nn.Conv2d(3,64,3, padding=1), nn.ReLU(), nn.MaxPool2d(2,2), nn.Conv2d(64,256,3, padding=1), nn.ReLU(), nn.MaxPool2d(2,2), nn.Conv2d(256,512,3, padding=1), nn.ReLU(), nn.MaxPool2d(2,2))
        self.decoder=nn.Sequential(nn.ConvTranspose2d(512,256,2, stride=2), nn.ReLU(), nn.ConvTranspose2d(256,64,2,stride=2), nn.ReLU(), nn.ConvTranspose2d(64,3,2, stride=2), nn.Tanh())

    def forward(self, x):
        x=self.encoder(x)
        return self.decoder(x)


class small_auto_encoder(nn.Module):
    def __init__(self):
        super(auto_encoder, self).__init__()

        self.encoder=nn.Sequential(nn.Conv2d(3,16,3, padding=1), nn.ReLU(), nn.MaxPool2d(2,2), nn.Conv2d(16,32,3, padding=1), nn.ReLU(), nn.MaxPool2d(2,2))#, nn.Conv2d(32,64,3, padding=1), nn.ReLU(), nn.MaxPool2d(2,2))
        self.decoder=nn.Sequential(nn.ConvTranspose2d(32,16,2, stride=2), nn.ReLU(), nn.ConvTranspose2d(16,3,2,stride=2), nn.Tanh())#, nn.ReLU(), nn.ConvTranspose2d(16,3,2, stride=2), nn.Softmax())

    def forward(self, x):
        x=self.encoder(x)
        return self.decoder(x)


# %%
