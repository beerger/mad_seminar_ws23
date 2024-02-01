import os
from typing import List, Tuple

import pandas as pd
import pytorch_lightning as pl
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class TestDataset(Dataset):

    def __init__(self, img_csv: str, pos_mask_csv: str):
        """
        Loads anomalous images, and their positive masks

        @param img_csv: str
            path to csv file containing filenames to the images
        @param img_csv: str
            path to csv file containing filenames to the positive masks
        """
        super(TestDataset, self).__init__()
        self.img_paths = pd.read_csv(img_csv)['filename'].tolist()
        self.pos_mask_paths = pd.read_csv(pos_mask_csv)['filename'].tolist()
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
        pos_mask = pos_mask.resize(self.target_size, Image.NEAREST)
        pos_mask = transforms.ToTensor()(pos_mask)

        return img, pos_mask
    
    def _build_transforms_global(self):
        return transforms.Compose([
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),  # Resize to the expected input size for the Global-Net
            transforms.ToTensor(),
        ])


def get_all_test_dataloader(split_dir: str, batch_size: int):
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
        img_csv = os.path.join(split_dir, f'{pathology}.csv')
        pos_mask_csv = os.path.join(split_dir, f'{pathology}_ann.csv')

        all_images.extend(pd.read_csv(img_csv)['filename'].tolist())
        all_pos_masks.extend(pd.read_csv(pos_mask_csv)['filename'].tolist())

    combined_dataset = TestDataset(all_images, all_pos_masks)
    return DataLoader(combined_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

