import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self,width=112):
        super().__init__()
        self.layers=nn.Sequential(nn.Flatten(),nn.Linear(3072,width),nn.ReLU(),nn.Linear(width,width),nn.ReLU())
        self.head=nn.Linear(width,10)
    def features(self,x): return self.layers(x)
    def forward(self,x): return self.head(self.features(x))

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        layers=[]
        for a,b in [(3,32),(32,64),(64,128)]:
            layers.extend([nn.Conv2d(a,b,3,padding=1),nn.ReLU(),nn.MaxPool2d(2)])
        self.conv=nn.Sequential(*layers)
        self.fc=nn.Sequential(nn.Flatten(),nn.Linear(2048,128),nn.ReLU())
        self.head=nn.Linear(128,10)
    def features(self,x): return self.fc(self.conv(x))
    def forward(self,x): return self.head(self.features(x))

class Autoencoder(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.conv=nn.Sequential(nn.Conv2d(3,16,3,2,1),nn.ReLU(),nn.Conv2d(16,32,3,2,1),nn.ReLU(),nn.Conv2d(32,64,3,2,1),nn.ReLU())
        self.encoder=nn.Linear(1024,dim)
        self.expand=nn.Linear(dim,1024)
        self.decoder=nn.Sequential(nn.ConvTranspose2d(64,32,4,2,1),nn.ReLU(),nn.ConvTranspose2d(32,16,4,2,1),nn.ReLU(),nn.ConvTranspose2d(16,3,4,2,1),nn.Sigmoid())
    def features(self,x): return self.encoder(self.conv(x).flatten(1))
    def forward(self,x):
        z=self.features(x)
        return self.decoder(torch.relu(self.expand(z)).reshape(-1,64,4,4))

def count(model): return sum(p.numel() for p in model.parameters())
