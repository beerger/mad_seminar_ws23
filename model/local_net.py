import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torchvision import models
import os
import requests
from torch import Tensor

class LocalNet(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.device = torch.device(config['device']) # Set device as defined in yaml-file
        self.checkpoint_path = 'resnet18-5c106cde.pth'  # Path to the checkpoint file

        # TODO: Should bias be true or false?
        # Image size is 33 x 33 x 3, outputs feature map size 1 x 1 x 128
        self.model = nn.Sequential(
            nn.Conv2d(3, 128, 3, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=self.config['leaky_relu_slope']),
            nn.MaxPool2d(2, stride=2, padding=0),
            nn.Conv2d(128, 256, 5, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=self.config['leaky_relu_slope']),
            nn.MaxPool2d(2, stride=2, padding=0),
            nn.Conv2d(256, 256, 2, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=self.config['leaky_relu_slope']),
            nn.Conv2d(256, 128, 4, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=self.config['leaky_relu_slope'])
        )

        self.model.to(self.device) # Move the model to the specified device

        # TODO: Should it only be one (linaer?) layer?
        # Define the decoder as a fully connected network
        # Feature map from Local-Net has size 1 x 1 x 128
        # Feature map from ResNet-18 has size 1 x 1 x 512
        self.decoder = nn.Linear(in_features=128, out_features=512)
        self.decoder.to(self.device) # Move the model to the specified device

        self.teacher_model = self.load_teacher_model() # Load teacher model (ResNet-18 with its fully connected layer removed)

    def forward(self, x):
        x = self.model(x)
        return x
    
    def load_teacher_model(self):
        # Check if the checkpoint file already exists
        if not os.path.isfile(self.checkpoint_path):
            # Download the file
            # Checkpoint used in paper is mentioned in Supplementary Material, found at 
            # https://openaccess.thecvf.com/content/CVPR2021/supplemental/Wang_Glancing_at_the_CVPR_2021_supplemental.pdf
            response = requests.get('https://download.pytorch.org/models/resnet18-5c106cde.pth', stream=True)
            with open(self.checkpoint_path, 'wb') as f:
                f.write(response.content)
        
        model = models.resnet18(pretrained=False) # Load the ResNet-18 model

        # Load the downloaded weights
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint)

        # Remove the fully connected layer
        model = torch.nn.Sequential(*(list(model.children())[:-1]))
        model.eval().to(self.device)  # Set the model to evaluation mode and move the model to the specified device
        return model

    def loss_function(self, output, teacher_output):
        # Compute knowledge and compactness loss
        lk = self.knowledge_dist_loss(output, teacher_output)
        lc = self.compactness_loss(output)
        return self.config['loss_weight_k'] * lk + self.config['loss_weight_c'] * lc

    def knowledge_dist_loss(self, output, teacher_output):
        output_flat = output.view(output.size(0), -1) # Flatten the student output to match the linear layer's input expectations
        decoded_output = self.decoder(output_flat) # Ensures same output dimensions as ResNet-18
        loss = torch.norm(decoded_output - teacher_output, p=2)**2
        return loss
        
    def compactness_loss(self, output):
        n = output.size(0)  # Batch size
        output_flat = output.view(n, -1)  # Flatten the feature maps

        # Normalize the features to zero mean and unit variance
        mean = output_flat.mean(dim=1, keepdim=True)
        std = output_flat.std(dim=1, keepdim=True) + 1e-8  # Add epsilon to avoid division by zero
        output_norm = (output_flat - mean) / std

        # Compute the correlation matrix using normalized features
        correlation_matrix = torch.matmul(output_norm, output_norm.t())

        # Compute compactness loss as the sum of off-diagonal elements
        loss_c = correlation_matrix.sum() - correlation_matrix.diag().sum()
        return loss_c

    def training_step(self, batch: Tensor, batch_idx):
        x = batch
        output = self.forward(x)
        with torch.no_grad():  # No need to track gradients for the teacher model
            teacher_output = self.teacher_model(x)
            teacher_output = torch.flatten(teacher_output, start_dim=1)
        loss = self.loss_function(output, teacher_output)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch: Tensor, batch_idx):
        x = batch
        output = self.forward(x)
        with torch.no_grad():  # No need to track gradients for the teacher model
            teacher_output = self.teacher_model(x)
            teacher_output = torch.flatten(teacher_output, start_dim=1)
        loss = self.loss_function(output, teacher_output)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def configure_optimizers(self):
        print(self.config)
        return optim.Adam(self.parameters(), betas=(self.config['beta_1'], self.config['beta_2']), lr=self.config['lr'])