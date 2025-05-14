import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import pytorch_lightning as pl
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

import gdown
import zipfile
import os.path as osp

from .transforms_utils import ResizePad224

class BeesSingleBalancedDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers, data_dir: str = '', zip_path: str = '', resize_pad_224: bool = False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

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

        # losowy podział, z równowagą, na train i temp
        indices = list(range(len(full_dataset)))
        labels  = [y for _, y in full_dataset.imgs]

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=10)
        train_idx, tmp_idx = next(sss.split(indices, labels))

        # podział temp na val i test, 2:1
        tmp_labels = [labels[i] for i in tmp_idx]
        sss_val = StratifiedShuffleSplit(n_splits=1, test_size=1/3, random_state=42)
        val_idx, test_idx = next(sss_val.split(tmp_idx, tmp_labels))

        self.train_dataset = Subset(full_dataset, train_idx)
        self.val_dataset = Subset(full_dataset, [tmp_idx[i] for i in val_idx])
        self.test_dataset = Subset(full_dataset, [tmp_idx[i] for i in test_idx])

        # sampler z wagami
        if stage in ("fit", None):
            targets = [full_dataset.imgs[i][1] for i in train_idx]
            class_counts = np.bincount(targets)
            weights = 1.0 / class_counts
            sample_weights = [weights[t] for t in targets]
            self.train_sampler = WeightedRandomSampler(
                sample_weights, num_samples=len(sample_weights), replacement=True
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=self.train_sampler, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
