import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
from torchvision.transforms import Resize, Compose, ToTensor, Normalize


import gdown
import zipfile
import os.path as osp

from .transforms_utils import ResizePad224

class BeesDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, data_dir: str = '', zip_path: str = '', resize_pad_224: bool = False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        image_transformations = []
        if resize_pad_224:
            image_transformations.append(
                ResizePad224()
            )
        image_transformations.extend(
            [ ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) ]
        )

        self.imagenet_transform = Compose(image_transformations)
        self.num_classes = 5
        self.zip_name = zip_path

    def prepare_data(self):
        if not osp.isfile(self.zip_name):
            gdown.download('https://drive.google.com/uc?id=1S7xrUA1npvJqhsi0qqlyzubeStmp32Hi', output=self.zip_name, quiet=False)

        if not osp.isdir(self.data_dir):
            with zipfile.ZipFile(self.zip_name, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)

    def setup(self, stage=None):
        full_dataset = ImageFolder(self.data_dir, transform=self.imagenet_transform)

        train_dataset_size = int(len(full_dataset) * 0.7)
        val_dataset_size = int(len(full_dataset) * 0.2)
        test_dataset_size = len(full_dataset) - train_dataset_size - val_dataset_size

        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, [train_dataset_size, val_dataset_size, test_dataset_size]
        )

        if stage == 'fit' or stage is None:
          self.train_dataset = train_dataset
          self.val_dataset = val_dataset

        if stage == 'test' or stage is None:
          self.test_dataset = test_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
