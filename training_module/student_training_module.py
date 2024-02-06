import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch import Tensor

class StudentTrainingModule(pl.LightningModule):
    """
    A training module for both distillation and fine-tuning phases of a student model.

    Attributes:
        config (dict): Configuration parameters for the training module.
        student_model (nn.Module): The student model to be trained.
        teacher_model (nn.Module): The pre-trained teacher model used for knowledge distillation.
        decoder (nn.Module): The decoder used to align the dimensions of the student's output with the teacher's.
        mode (str): Indicates the mode of training - either 'distillation' or 'finetuning'.
    """
    def __init__(self, config, student_model, teacher_model, decoder, mode):
        super().__init__()
        self.config = config
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.decoder = decoder
        self.mode = mode # 'distillation' or 'finetuning

        # Ensure the teacher model is in evaluation mode and gradients are not calculated
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        # Set up the decoder based on the mode
        if self.mode == 'finetuning':
            # In fine-tuning mode, the decoder should not be updated
            self.decoder.eval()
            for param in self.decoder.parameters():
                param.requires_grad = False
        elif self.mode == 'distillation':
            # In distillation mode, the decoder is part of the training process
            self.decoder.train()
            for param in self.decoder.parameters():
                param.requires_grad = True
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")
            
    def loss_function(self, student_output, teacher_output):
        # Compute knowledge and compactness loss
        lk = self.knowledge_dist_loss(student_output, teacher_output)
        lc = self.compactness_loss(student_output)
        return self.config['loss_weight_k'] * lk + self.config['loss_weight_c'] * lc

    def knowledge_dist_loss(self, student_output, teacher_output):
        # Flatten the student output to match the linear layer's input expectations
        student_output_flat = torch.flatten(student_output, 1) 
        # Ensures same output dimensions as teacher model
        decoded_output = self.decoder(student_output_flat) 
        loss = torch.norm(decoded_output - teacher_output, p=2)**2
        return loss
    
    def compactness_loss(self, student_output):
        # Flatten the feature maps and center them
        h = student_output.view(student_output.size(0), -1)
        h = h - h.mean(dim=0)

        # Compute the correlation matrix
        r = torch.mm(h.T, h) / (student_output.size(0) - 1)

        # Normalize the correlation matrix by the standard deviations
        std = torch.sqrt(torch.diag(r))
        outer_std = torch.outer(std, std)
        corr_matrix = r / outer_std

        # Sum the off-diagonal elements of the correlation matrix
        # Use torch.ones_like to mask the diagonal elements
        off_diagonal_mask = torch.ones_like(corr_matrix) - torch.eye(corr_matrix.size(0), device=corr_matrix.device)
        compactness = (corr_matrix * off_diagonal_mask).sum()

        return compactness
        
    def training_step(self, batch: Tensor, batch_idx):
        local_patch, resnet_patch = batch
        student_output = self.student_model(local_patch)
        with torch.no_grad():  # No need to track gradients for the teacher model
            teacher_output = self.teacher_model(resnet_patch)
            teacher_output = torch.flatten(teacher_output, start_dim=1)
        loss = self.loss_function(student_output, teacher_output)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch: Tensor, batch_idx):
        local_patch, resnet_patch = batch
        student_output = self.student_model(local_patch)
        with torch.no_grad():  # No need to track gradients for the teacher model
            teacher_output = self.teacher_model(resnet_patch)
            teacher_output = torch.flatten(teacher_output, start_dim=1)
        loss = self.loss_function(student_output, teacher_output)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def configure_optimizers(self):
        if self.mode == 'distillation':
            # If in distillation mode, we want to train both the student and decoder
            optimizer = optim.Adam(self.parameters(), betas=(self.config['beta_1'], self.config['beta_2']), lr=self.config['lr'])
        elif self.mode == 'finetuning':
            # If in finetuning mode, we only want to train the student model
            optimizer = optim.Adam(self.student_model.parameters(), betas=(self.config['beta_1'], self.config['beta_2']), lr=self.config['lr'])
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        return optimizer