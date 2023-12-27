import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch import Tensor
from iad import iad_head

class DADHead(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        # Input feature is a 256d vector
        self.fc0 = nn.Linear(256, 128)
        self.leaky_relu = nn.LeakyReLU(negative_slope=self.config['leaky_relu_slope'])
        self.fc1 = nn.Linear(128, 2)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.fc0(x)
        x = self.leaky_relu(x)
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.softmax(x)
        return x

    # FIXME: Training loss should use IAD loss too (I think)
    #def loss_function(self, predictions, targets):
    #    l_iad = iad_head()
    #    loss_func_dad = nn.BCEWithLogitsLoss()
    #    l_dad = loss_func_dad(predictions, targets)
    #    return l_iad + self.config['loss_weight_t'] * l_dad

    def loss_function(self, predictions, targets):
        loss_func_dad = nn.BCEWithLogitsLoss()
        return loss_func_dad(predictions, targets)
        

    def training_step(self, batch: Tensor, batch_idx):
        inputs, targets = batch
        predictions = self.forward(inputs)
        loss = self.loss_function(predictions, targets)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch: Tensor, batch_idx):
        inputs, targets = batch
        predictions = self.forward(inputs)
        loss = self.loss_function(predictions, targets)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def configure_optimizers(self):
        print(self.config)
        return optim.Adam(self.parameters(), betas=(self.config['beta_1'], self.config['beta_2']), lr=self.config['lr'])