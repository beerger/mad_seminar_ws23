import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torchvision import models
import os
import requests
from torch import Tensor

# 'resnet18-5c106cde.pth'

def load_resnet_18_teacher_model(checkpoint_path, device):
    # Check if the checkpoint file already exists
    if not os.path.isfile(checkpoint_path):
        # Download the file
        # Checkpoint used in paper is mentioned in Supplementary Material, found at 
        # https://openaccess.thecvf.com/content/CVPR2021/supplemental/Wang_Glancing_at_the_CVPR_2021_supplemental.pdf
        response = requests.get('https://download.pytorch.org/models/resnet18-5c106cde.pth', stream=True)
        with open(checkpoint_path, 'wb') as f:
            f.write(response.content)
    
    model = models.resnet18() # Load the ResNet-18 model
    # Load the downloaded weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    # Remove the fully connected layer (teacher model is only used to extract features, not classification)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model.eval().to(device)  # Set the model to evaluation mode and move the model to the specified device
    return model