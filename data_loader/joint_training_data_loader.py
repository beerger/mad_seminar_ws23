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
import numpy as np
from random import randint, uniform
from numpy.random import normal, uniform
from skimage.util import random_noise
from skimage.filters import threshold_otsu
from skimage.draw import ellipse_perimeter
from scipy.interpolate import interp1d 
from scipy.ndimage import gaussian_filter
import math
import cv2

class JointTrainingDataset(Dataset):
    def __init__(self, image_paths, is_train=True, caching_strategy='none'):
        self.image_paths = image_paths
        self.is_train = is_train
        self.transform_local = self._build_transforms_local()
        
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
            I = transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)(image)
            cache[idx] = I  # Cache the full-size image only
        return cache
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if self.cache is not None:
            if idx in self.cache:
                image = self.cache[idx]  # Retrieve the full-size image from cache
            else:
                image = self.load_and_transform_image(idx)
                self.cache[idx] = image  # Cache the full-size image only
        else:
            image = self.load_and_transform_image(idx)

        # Generate a new patch, binary mask, and target label every time
        patch, binary_mask, target_label = self.create_patch_binary_mask_and_target_label(image)

        image = transforms.ToTensor()(image)

        return image, patch, binary_mask, target_label
    
    def load_and_transform_image(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        image = transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)(image)
        return image  # Return the full-size image only

    def create_patch_binary_mask_and_target_label(self, image):

            # Create a patch with or without modifications
            if self.is_train:
                crop_transform = transforms.RandomCrop(33, padding=None, pad_if_needed=False, fill=0, padding_mode='constant')
            else:
                crop_transform = transforms.CenterCrop(33)
            
            positive_patch, i, j, h, w = self.random_crop_with_coords(image, crop_transform)
            positive_patch = transforms.ToTensor()(positive_patch)

            patch, target_label = None, None
            # Determine whether to use positive or negative patch for this index
            use_negative = random.choice([True, False])
            if use_negative:
                size = "1-12"  # 1% to 12% of the patch size
                color = "0-100"  # Full range of grayscale intensities
                # Randomize irregularity and blur within specified ranges
                irregularity_range = (0.3, 0.7)  # Example range for irregularity
                blur_range = (0.0, 0.0)  # Example range for blur

                irregularity = random.uniform(*irregularity_range)
                blur = random.uniform(*blur_range)
                
                patch = self.add_stain(positive_patch, size, color, irregularity, blur)
                target_label = 1  # Label for negative patch (abnormal)
            else:
                patch = positive_patch
                target_label = 0  # Label for positive patch (normal)

            patch = self.transform_local(patch)
            binary_mask = self.generate_mask((i, j, h, w))

            binary_mask = binary_mask.unsqueeze(0) # Now binary_mask has shape: [1, height, width]

            return patch, binary_mask, target_label
    
    def _build_transforms_local(self):
        # Define transforms for Local-Net
        return transforms.Compose([
            #transforms.ToTensor(),
            # Mean and std from ImageNet
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def add_stain(self, img, size, color, irregularity, blur):

        img = img.permute(1, 2, 0).numpy()
    
        if '-' not in color: 
            color = int(color)
        else: 
            min_color, max_color = int(color.split('-')[0]), int(color.split('-')[1])
            color                = randint(min_color, max_color)
        row, col, _ = img.shape
        min_range, max_range = float(size.split('-')[0]), float(size.split('-')[1])
        rotation = uniform(0, 2*np.pi)
        a = max(1, randint(int(min_range/100.0 * col), min(int(max_range/100.0 * col), col//2)))
        b = max(1, randint(int(min_range/100.0 * row), min(int(max_range/100.0 * row), row//2)))

        # Ensure the ellipse fits within the image
        cx = randint(a, col - a)
        cy = randint(b, row - b)

        x,y      = ellipse_perimeter(cy, cx, a, b, rotation)
        
        contour  = np.array([[i,j] for i,j in zip(x,y)])

        # Change the shape of the ellipse 
        if irregularity > 0: 
            contour = self.perturbate_ellipse(contour, cx, cy, (a+b)/2, irregularity)

        mask = np.zeros((row, col)) 
        mask = cv2.drawContours(mask, [contour], -1, 1, -1)

        if blur != 0 : 
            mask = gaussian_filter(mask, max(a,b)*blur)

        if img.shape[2] == 1: # Grayscale image
            rgb_mask     = np.expand_dims(mask, axis=-1)
        else: # Color image
            rgb_mask = np.repeat(mask[:, :, np.newaxis], img.shape[2], axis=2)

        # Generate a constant-color stain
        stain_intensity = (color / 255.) if isinstance(color, int) else (randint(min_color, max_color) / 255.)
        stain = np.full(img.shape, stain_intensity, dtype=np.float32)
        
        # Apply the mask to the stain
        stain_masked = stain * rgb_mask.astype(np.float32)

        # Apply the stain to the image
        result = np.where(rgb_mask.astype(bool), stain_masked, img)

        # Convert result back to PyTorch tensor and permute back to [C, H, W]
        result = torch.from_numpy(result).permute(2, 0, 1)
        return result

    def perturbate_ellipse(self, contour, cx, cy, diag, irregularity):
        # Keep only some points
        if len(contour) < 20: 
            pts = contour
        else: 
            pts = contour[0::int(len(contour)/20)]

        # Perturbate coordinates
        for idx,pt in enumerate(pts): 
            pts[idx] = [pt[0]+randint(-int(diag*irregularity),int(diag*irregularity)), pt[1]+randint(-int(diag*irregularity),int(diag*irregularity))]
        pts = sorted(pts, key=lambda p: self.clockwiseangle(p, cx, cy))
        pts.append([pts[0][0], pts[0][1]])

        # Interpolate between remaining points
        i = np.arange(len(pts))
        interp_i = np.linspace(0, i.max(), 10 * i.max())
        xi = interp1d(i, np.array(pts)[:,0], kind='cubic')(interp_i)
        yi = interp1d(i, np.array(pts)[:,1], kind='cubic')(interp_i) 

        return np.array([[int(i),int(j)] for i,j in zip(yi,xi)])

    def clockwiseangle(self, point, cx, cy):
        refvec = [0 , 1]
        vector = [point[0]-cy, point[1]-cx]
        norm   = math.hypot(vector[0], vector[1])
        # If length is zero there is no angle
        if norm == 0:
            return -math.pi
        normalized = [vector[0]/norm, vector[1]/norm]
        dotprod    = normalized[0]*refvec[0] + normalized[1]*refvec[1] 
        diffprod   = refvec[1]*normalized[0] - refvec[0]*normalized[1] 
        angle      = math.atan2(diffprod, dotprod)
        if angle < 0:
            return 2*math.pi+angle
        return angle

    def generate_mask(self, crop_coordinates):
        mask_size = (256, 256)
        mask = torch.ones(mask_size)
        y, x, h, w = crop_coordinates  # Unpack the crop coordinates
        mask[y:y+h, x:x+w] = 0
        return mask
    
    def random_crop_with_coords(self, img, crop_transform):
        if isinstance(crop_transform, transforms.RandomCrop):
            i, j, h, w = crop_transform.get_params(img, crop_transform.size)
        else:  # For deterministic crops like CenterCrop
            img_size = img.size
            th, tw = crop_transform.size  # Target height and width
            i = (img_size[0] - th) // 2
            j = (img_size[1] - tw) // 2
            h, w = th, tw

        cropped_img = transforms.functional.crop(img, i, j, h, w)
        return cropped_img, i, j, h, w


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

