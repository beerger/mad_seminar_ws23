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
                size = "1-12"  # 1% to 12% of the patch size
                color = "0-10"  # Full range of grayscale intensities
                # Randomize irregularity and blur within specified ranges
                irregularity_range = (0.3, 0.7)  # Example range for irregularity
                blur_range = (0, 0)  # Example range for blur

                irregularity = random.uniform(*irregularity_range)
                blur = random.uniform(*blur_range)
                
                patch = self.add_stain(positive_patch, size, color, irregularity, blur)
                target_label = 1  # Label for negative patch (abnormal)
            else:
                patch = positive_patch
                target_label = 0  # Label for positive patch (normal)

            binary_mask = self.generate_mask((i, j, h, w))


            # TODO: See if necessary
            binary_mask = binary_mask.unsqueeze(0) # Now binary_mask has shape: [1, height, width]

            I = transforms.ToTensor()(I)
            return I, patch, binary_mask, target_label
        except UnidentifiedImageError:
            print(f"Error loading image: {image_path}")
            return None, None, None, None

    def add_stain(self, img, size, color, irregularity, blur):

        img = img.permute(1, 2, 0).numpy()

        row, col, _ = img.shape  # row and col are now correctly assigned

    
        if '-' not in color: 
            color = int(color)
        else: 
            min_color, max_color = int(color.split('-')[0]), int(color.split('-')[1])
            color                = randint(min_color, max_color)
        col, row             = img.shape[1], img.shape[0]
        min_range, max_range = float(size.split('-')[0]), float(size.split('-')[1])
        rotation = uniform(0, 2*np.pi)
        a = max(1, randint(int(min_range/100.0 * col), min(int(max_range/100.0 * col), col//2)))
        b = max(1, randint(int(min_range/100.0 * row), min(int(max_range/100.0 * row), row//2)))

        # Ensure the ellipse fits within the image
        cx = randint(a, col - a)
        cy = randint(b, row - b)

        #print(f"cy: {cy}")
        #print(f"cx: {cx}")
        #print(f"a: {a}")
        #print(f"b: {b}")
        #print(f"rotation: {rotation}")
        x,y      = ellipse_perimeter(cy, cx, a, b, rotation)

        #print(f"x: {x}")
        #print(f"y: {y}")

        
        contour  = np.array([[i,j] for i,j in zip(x,y)])

        # Change the shape of the ellipse 
        if irregularity > 0: 
            contour = self.perturbate_ellipse(contour, cx, cy, (a+b)/2, irregularity)

        #print(f"row: {row}")
        #print(f"col: {col}")
        mask = np.zeros((row, col)) 
        mask = cv2.drawContours(mask, [contour], -1, 1, -1)

        if blur != 0 : 
            mask = gaussian_filter(mask, max(a,b)*blur)

        if img.shape[2] == 1: # Grayscale image
            rgb_mask     = np.expand_dims(mask, axis=-1)
        else: # Color image
            rgb_mask = np.repeat(mask[:, :, np.newaxis], img.shape[2], axis=2)
        #not_modified = np.subtract(np.ones_like(img), rgb_mask)
        #noise_channel = random_noise(np.zeros((row, col)), mode='gaussian', mean=color/255., var=0.05/255.)
        #stain = np.stack([255 * noise_channel] * 3, axis=-1)  # Replicate noise across all channels
        #result       = np.add( np.multiply(img,not_modified), np.multiply(stain,rgb_mask) ) 

        # Generate a constant-color stain
        stain_intensity = color if isinstance(color, int) else randint(min_color, max_color)
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
        x, y, h, w = crop_coordinates  # Unpack the crop coordinates
        mask[x:x+h, y:y+w] = 0
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

