import os
from typing import List, Tuple

import pandas as pd
import pytorch_lightning as pl
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ImageNetPatchesDataset(Dataset):
    def __init__(self, image_paths, is_train=True):
        self.image_paths = image_paths
        self.is_train = is_train
        self.transform_local = self._build_transforms_local()
        self.transform_resnet = self._build_transforms_resnet()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')

        # Resize image to 256x256
        image = transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)(image)

        patch = transforms.RandomCrop(33)(image) if self.is_train else transforms.CenterCrop(33)(image)

        # Apply transformations
        patch_local = self.transform_local(patch)
        patch_resnet = self.transform_resnet(patch)

        return patch_local, patch_resnet

    def _build_transforms_local(self):
        # Define transforms for Local-Net
        return transforms.Compose([
            transforms.ToTensor(),
            # Mean and std from ImageNet
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _build_transforms_resnet(self):
        # Define transforms for ResNet-18
        return transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.ToTensor(),
            # Mean and std from ImageNet
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

class ImageNetDataModule(pl.LightningDataModule):
    def __init__(self, train_image_paths, val_image_paths , batch_size=64, num_workers=4):
        super().__init__()
        self.train_image_paths = train_image_paths
        self.val_image_paths = val_image_paths
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        train_dataset = ImageNetPatchesDataset(self.train_image_paths, is_train=True)
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        val_dataset = ImageNetPatchesDataset(self.val_image_paths, is_train=False)
        return DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

