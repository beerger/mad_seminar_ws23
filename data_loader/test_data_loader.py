import os
from typing import List, Tuple

import pandas as pd
import pytorch_lightning as pl
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class TestDataset(Dataset):

    def __init__(self, img_paths: str, pos_mask_paths):
        """
        Loads anomalous images, and their positive masks

        @param img_paths: list
            List of paths to the images.
        @param pos_mask_paths: list
            List of paths to the positive masks.
        """
        super(TestDataset, self).__init__()
        self.img_paths = img_paths
        self.pos_mask_paths = pos_mask_paths
        self.transform_global = self._build_transforms_global()
        assert len(self.img_paths) == len(self.pos_mask_paths)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Load image
        img = Image.open(self.img_paths[idx]).convert('RGB')
        img = self.transform_global(img)

        # Load positive mask
        pos_mask = Image.open(self.pos_mask_paths[idx]).convert('L')
        pos_mask = pos_mask.resize((256, 256), Image.NEAREST)
        pos_mask = transforms.ToTensor()(pos_mask)

        return img, pos_mask
    
    def _build_transforms_global(self):
        return transforms.Compose([
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),  # Resize to the expected input size for the Global-Net
            transforms.ToTensor(),
        ])


class TestDataModule(pl.LightningDataModule):

    def __init__(self, split_dir: str, batch_size=None):
        super().__init__()
        self.split_dir = split_dir
        self.batch_size = batch_size

    def test_dataloader(self, pathology: str):
        """
        Loads test data from split_dir

        @param pathology: str
            pathology to load
        """
        img_csv = os.path.join(self.split_dir, f'{pathology}.csv')
        pos_mask_csv = os.path.join(self.split_dir, f'{pathology}_ann.csv')

        img_paths = pd.read_csv(img_csv)['filename'].tolist()
        pos_mask_paths = pd.read_csv(pos_mask_csv)['filename'].tolist()

        if not(self.batch_size):
          batch_size = len(img_paths)
        else:
          batch_size = self.batch_size
        return DataLoader(TestDataset(img_paths, pos_mask_paths),
                          batch_size=batch_size,
                          shuffle=False,
                          drop_last=False)

    def test_dataloader_all(self):
        """
        Loads all test data from split_dir, including all pathologies.

        @param split_dir: str
            Path to directory containing the split files.
        @param batch_size: int
            Batch size.
        """
        pathologies = [
            'absent_septum',
            'artefacts',
            'craniatomy',
            'dural',
            'ea_mass',
            'edema',
            'encephalomalacia',
            'enlarged_ventricles',
            'intraventricular',
            'lesions',
            'mass',
            'posttreatment',
            'resection',
            'sinus',
            'wml',
            'other'
        ]

        all_images = []
        all_pos_masks = []

        for pathology in pathologies:
            img_csv = os.path.join(self.split_dir, f'{pathology}.csv')
            pos_mask_csv = os.path.join(self.split_dir, f'{pathology}_ann.csv')

            all_images.extend(pd.read_csv(img_csv)['filename'].tolist())
            all_pos_masks.extend(pd.read_csv(pos_mask_csv)['filename'].tolist())

        combined_dataset = TestDataset(all_images, all_pos_masks)
        if not(self.batch_size):
          batch_size = len(all_images)
        else:
          batch_size = self.batch_size
        return DataLoader(combined_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

