import os
from typing import List, Tuple

import pandas as pd
import pytorch_lightning as pl
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import UnidentifiedImageError

class LocalNetPatchesDataset(Dataset):
    def __init__(self, image_paths, is_train=True, caching_strategy='none'):
        self.image_paths = image_paths
        self.is_train = is_train
        self.transform_local = self._build_transforms_local()
        self.transform_resnet = self._build_transforms_resnet()
        
        if caching_strategy == 'at-init':
            self.cache = self._preload_images()
        elif caching_strategy == 'on-the-fly':
            self.cache = {}
        else:
            self.cache = None

    def _preload_images(self):
        cache = {}
        for idx in range(len(self.image_paths)):
            image = Image.open(self.image_paths[idx]).convert('RGB')
            image = transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)(image)
            cache[idx] = image
        return cache

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load the full-size image from cache or disk
        if self.cache is not None:
            if idx in self.cache:
                image = self.cache[idx]
            else:
                image = self._load_image(idx)
                self.cache[idx] = image
        else:
            image = self._load_image(idx)

        # Generate a new patch every time
        patch = self._generate_patch(image)

        # Apply local and resnet transformations to the patch
        patch_local = self.transform_local(patch)
        patch_resnet = self.transform_resnet(patch)

        return patch_local, patch_resnet

    def _generate_patch(self, image):
        crop_transform = transforms.RandomCrop(33) if self.is_train else transforms.CenterCrop(33)
        return crop_transform(image)
        
    def _load_image(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert('RGB')
            image = transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)(image)
            return image
        except UnidentifiedImageError:
            print(f"Error loading image: {image_path}")
            return None

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

class LocalNetDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning data module for training of LocalNet

    This module handles the loading and batching of images for training and validation. 
    It provides flexibility in data caching strategies to optimize memory usage and data access speed.

    Attributes:
        train_image_paths (list): List of paths to the training images.
        val_image_paths (list): List of paths to the validation images.
        batch_size (int): Batch size for training and validation data loaders.
        num_workers (int): Number of workers to use for data loading.
        caching_strategy (str): Determines the image caching strategy. 
            'none' - No caching, images are loaded from disk on each access.
            'on-the-fly' - Images are cached as they are accessed.
            'at-init' - All images are preloaded into memory at initialization (requires substantial memory).

    Warnings:
        'at-init' caching strategy requires significant memory and should be used with caution. 
        It's suitable when there is enough RAM to hold the entire dataset. 
        'on-the-fly' caching will gradually consume more memory as more unique images are accessed.
    """
    def __init__(self, train_image_paths, val_image_paths , batch_size=16, num_workers=4, caching_strategy='none'):
        super().__init__()
        self.train_image_paths = train_image_paths
        self.val_image_paths = val_image_paths
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.caching_strategy = caching_strategy

        if caching_strategy == 'at-init':
            print("Warning: `caching_strategy` is set to 'at-init'. Ensure you have enough memory.")
        elif caching_strategy == 'on-the-fly':
            print("Warning: `caching_strategy` is set to 'on-the-fly'. Ensure you have enough memory.")

    def train_dataloader(self):
        train_dataset = LocalNetPatchesDataset(
            self.train_image_paths,
            is_train=True, 
            caching_strategy=self.caching_strategy
        )
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        val_dataset = LocalNetPatchesDataset(
            self.val_image_paths, 
            is_train=False, 
            caching_strategy=self.caching_strategy
        )
        return DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

