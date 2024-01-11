import os
from typing import List, Tuple

import pandas as pd
import pytorch_lightning as pl
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ImageNetPatchesDataset(Dataset):
    def __init__(self, image_paths, is_train=True, cache_images=False):
        self.image_paths = image_paths
        self.is_train = is_train
        self.transform_local = self._build_transforms_local()
        self.transform_resnet = self._build_transforms_resnet()
        self.cache = {} if cache_images else None # Initialize an empty cache if caching is enabled

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if self.cache is not None:
            # If caching is enabled and the image is cached, return it from the cache
            if idx in self.cache:
                patch_local, patch_resnet = self.cache[idx]
            else:
                # If the image is not cached, load and transform it, then add it to the cache
                patch_local, patch_resnet = self.load_and_transform_image(idx)
                self.cache[idx] = (patch_local, patch_resnet)
        else:
            # If caching is not enabled, simply load and transform the image without caching
            patch_local, patch_resnet = self.load_and_transform_image(idx)

        return patch_local, patch_resnet

    def load_and_transform_image(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')

        # Resize image to 256x256
        image = transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)(image)

        # Create a patch for local and resnet transformations
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
    """
    PyTorch Lightning data module for ImageNet dataset.

    Attributes:
        train_image_paths (list): List of paths to the training images.
        val_image_paths (list): List of paths to the validation images.
        batch_size (int): Batch size for training and validation data loaders.
        num_workers (int): Number of workers to use for data loading.
        cache_images (bool): If True, caches images in memory. Use with caution.
    """
    def __init__(self, train_image_paths, val_image_paths , batch_size=64, num_workers=4, cache_images=False):
        super().__init__()
        self.train_image_paths = train_image_paths
        self.val_image_paths = val_image_paths
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_images = cache_images

        if cache_images:
            # Warning about memory usage
            print("Warning: `cache_images` is set to True. Ensure you have enough memory to cache all images.")

    def train_dataloader(self):
        train_dataset = ImageNetPatchesDataset(
            self.train_image_paths,
            is_train=True, 
            cache_images=self.cache_images
        )
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        val_dataset = ImageNetPatchesDataset(
            self.val_image_paths, 
            is_train=False, 
            cache_images=self.cache_images
        )
        return DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

