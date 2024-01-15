import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from iad import iad_head

class JointGlobalDADTrainingModule(pl.LightningModule):
    def __init__(self, config, local_net, global_net, dad_head):
        super().__init__()
        self.config = config
        self.local_net = local_net
        self.global_net = global_net
        self.dad_head = dad_head
        self.automatic_optimization = False

        # Make sure to freeze the Local-Net if it's not supposed to be finetuned
        self.local_net.eval()
        for param in self.local_net.parameters():
            param.requires_grad = False

    def forward(self, batch):
        I, patch, binary_mask, _ = batch
        # Get local and global features
        local_features = self.local_net(patch)
        global_features = self.global_net(I, binary_mask)
        # Concatenate features for DAD-head input
        combined_features = torch.cat((local_features, global_features), dim=1)
        # Pass concatenated features through DAD-head
        dad_classification = self.dad_head(combined_features)
        return local_features, global_features, dad_classification

    def training_step(self, batch, batch_idx):

        optimizer_global, optimizer_dad = self.optimizers()
        # Zero gradients for both optimizers
        optimizer_global.zero_grad()
        optimizer_dad.zero_grad()
        # Split your data into inputs and targets
        local_features, global_features, dad_output = self.forward(batch)
        _, _ ,_ , targets = batch
        loss = self.joint_loss(local_features, global_features, dad_output, targets)
        # Backward pass for both optimizers
        self.manual_backward(loss)
        # Optimizer steps
        optimizer_global.step()
        optimizer_dad.step()
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Split your data into inputs and targets
        local_features, global_features, dad_output = self.forward(batch)
        _, _ ,_ , targets = batch
        loss = self.joint_loss(local_features, global_features, dad_output, targets)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def joint_loss(self, local_features, global_features, dad_output, targets):
        # Compute IAD loss (no targets needed for this part)
        iad_loss = iad_head(local_features, global_features)
        # Compute DAD loss with DAD-head output and targets
        dad_loss = self.compute_dad_loss(dad_output, targets)
        # Compute total loss
        total_loss = iad_loss + self.config['loss_weight_t'] * dad_loss
        return total_loss

    def compute_dad_loss(self, dad_output, targets):
        # Create an instance of the BCEWithLogitsLoss
        loss_fn = nn.BCEWithLogitsLoss()
        # Compute the loss
        dad_loss = loss_fn(dad_output, targets)
        return dad_loss

    def configure_optimizers(self):
        # Optimizers for both Global-Net and DAD-Head
        optimizer_global = optim.Adam(self.global_net.parameters(), betas=(self.config['beta_1'], self.config['beta_2']), lr=self.config['lr_global'])
        optimizer_dad = optim.Adam(self.dad_head.parameters(), betas=(self.config['beta_1'], self.config['beta_2']), lr=self.config['lr_dad'])
        return [optimizer_global, optimizer_dad], []  # No LR schedulers

