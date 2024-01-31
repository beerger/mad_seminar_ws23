import pytorch_lightning as pl
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import UnidentifiedImageError

class InferenceDataSet(Dataset):
    def __init__(self, image_paths, caching_strategy='none'):
        self.image_paths = image_paths
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
                I = self.cache[idx]
            else:
                # If the image is not cached, load and transform it, then add it to the cache
                I = self.load_and_transform_image(idx)
                self.cache[idx] = (I)
        else:
            # If caching is not enabled, simply load and transform the image without caching
            I = self.load_and_transform_image(idx)

        return I

    def load_and_transform_image(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert('RGB')
            I = self.transform_global(image)
            return I
        except UnidentifiedImageError:
            print(f"Error loading image: {image_path}")
            return None, None, None

    def _build_transforms_global(self):
        return transforms.Compose([
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),  # Resize to the expected input size for the Global-Net
            transforms.ToTensor(),
        ])


class InferenceDataModule(pl.LightningDataModule):

    def __init__(self, test_image_paths, batch_size=16, num_workers=4, caching_strategy='none'):
        super().__init__()
        self.test_image_paths = test_image_paths
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.caching_strategy = caching_strategy

        if caching_strategy == 'at-init':
            print("Warning: `caching_strategy` is set to 'at-init'. Ensure you have enough memory.")
        elif caching_strategy == 'on-the-fly':
            print("Warning: `caching_strategy` is set to 'on-the-fly'. Ensure you have enough memory.")

    def test_dataloader(self):
        test_dataset = InferenceDataSet(
            self.test_image_paths,
            caching_strategy=self.caching_strategy
        )
        return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
