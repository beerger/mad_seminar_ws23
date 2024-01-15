import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch import Tensor


# need to concatenate features into a 256d vector before
class DADHead(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        # Input feature is a 256d vector
        self.fc0 = nn.Linear(256, 128)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.fc1 = nn.Linear(128, 2)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.fc0(x)
        x = self.leaky_relu(x)
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.softmax(x)
        return x
