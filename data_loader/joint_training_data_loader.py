import os
from typing import List, Tuple
import torch
import pandas as pd
import pytorch_lightning as pl
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import UnidentifiedImageError
import random

class JointTrainingDataset(Dataset):
    def __init__(self, image_paths, is_train=True, caching_strategy='none'):
        self.image_paths = image_paths
        self.is_train = is_train
        self.transform_local = self._build_transforms_local()
        self.transform_global = self._build_transforms_global()
        
        if caching_strategy == 'at-init':
            self.cache = self._preload_images()
        elif caching_strategy == 'on-the-fly':
            self.cache = {}
        else:
            self.cache = None

    def _preload_images(self):
        cache = {}
        for idx in range(len(self.image_paths)):
            cache[idx] = self.load_and_transform_image(idx)
        return cache

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if self.cache is not None:
            # If caching is enabled and the image is cached, return it from the cache
            if idx in self.cache:
                I, patch, binary_mask, target_label = self.cache[idx]
            else:
                # If the image is not cached, load and transform it, then add it to the cache
                I, patch, binary_mask, target_label = self.load_and_transform_image(idx)
                self.cache[idx] = (I, patch, binary_mask, target_label)
        else:
            # If caching is not enabled, simply load and transform the image without caching
            I, patch, binary_mask, target_label = self.load_and_transform_image(idx)

        return I, patch, binary_mask, target_label

    def load_and_transform_image(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert('RGB')

            # Image for Global-Net
            I = transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)(image)

            # Create a patch with or without modifications
            if self.is_train:
                crop_transform = transforms.RandomCrop(33, padding=None, pad_if_needed=False, fill=0, padding_mode='constant')
            else:
                crop_transform = transforms.CenterCrop(33)
            
            positive_patch, i, j, h, w = self.random_crop_with_coords(I, crop_transform)
            positive_patch = self.transform_local(positive_patch)

            patch, target_label = None, None
            # Determine whether to use positive or negative patch for this index
            use_negative = random.choice([True, False])
            if use_negative:
                patch = self.add_stain(positive_patch)
                target_label = 1  # Label for negative patch (abnormal)
            else:
                patch = positive_patch
                target_label = 0  # Label for positive patch (normal)

            binary_mask = self.generate_mask((i, j, h, w))

            I = transforms.ToTensor()(I)
            return I, patch, binary_mask, target_label
        except UnidentifiedImageError:
            print(f"Error loading image: {image_path}")
            return None, None, None, None
        
    def add_stain(self, patch):
        # Randomly choose top-left corner of the stain
        x = random.randint(0, 23)  # 33 - 10 = 23 to ensure the stain fits in the patch
        y = random.randint(0, 23)

        # Create the stain (black patch)
        stain = torch.zeros((3, 10, 10))

        # Apply the stain to the patch
        patch[:, y:y+10, x:x+10] = stain

        return patch

    def generate_mask(self, crop_coordinates):
        mask_size = (256, 256)
        mask = torch.ones(mask_size)
        x, y, h, w = crop_coordinates  # Unpack the crop coordinates
        mask[x:x+h, y:y+w] = 0
        return mask
    
    def random_crop_with_coords(self, img, crop_transform):
        i, j, h, w = crop_transform.get_params(img, crop_transform.size)
        cropped_img = transforms.functional.crop(img, i, j, h, w)
        return cropped_img, i, j, h, w

    def _build_transforms_local(self):
        # Define transforms for Local-Net
        return transforms.Compose([
            transforms.ToTensor(),
            # Mean and std from ImageNet
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _build_transforms_global(self):
        return transforms.Compose([
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),  # Resize to the expected input size for the Global-Net
            transforms.ToTensor(),
        ])


class JointTrainingDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning data module for joint training on MVTec AD dataset.

    This module handles the loading and batching of MVTec AD data for training and validation. 
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
        train_dataset = JointTrainingDataset(
            self.train_image_paths,
            is_train=True, 
            caching_strategy=self.caching_strategy
        )
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        val_dataset = JointTrainingDataset(
            self.val_image_paths, 
            is_train=False, 
            caching_strategy=self.caching_strategy
        )
        return DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

