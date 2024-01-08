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

        # TODO: Do I need to freeze weights of decoder? Not to be optimized during training/validation steps
            
    def loss_function(self, student_output, teacher_output):
        # Compute knowledge and compactness loss
        lk = self.knowledge_dist_loss(student_output, teacher_output)
        lc = self.compactness_loss(student_output)
        return self.config['loss_weight_k'] * lk + self.config['loss_weight_c'] * lc

    def knowledge_dist_loss(self, student_output, teacher_output):
        output_flat = student_output.view(student_output.size(0), -1) # Flatten the student output to match the linear layer's input expectations

        # TODO: torch.no_grad() required?
        decoded_output = self.decoder(output_flat) # Ensures same output dimensions as teacher model
        loss = torch.norm(decoded_output - teacher_output, p=2)**2
        return loss
        
    def compactness_loss(self, student_output):
        n = student_output.size(0)  # Batch size
        output_flat = student_output.view(n, -1)  # Flatten the feature maps

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
        print(self.config)
        return optim.Adam(self.student_model.parameters(), betas=(self.config['beta_1'], self.config['beta_2']), lr=self.config['lr'])