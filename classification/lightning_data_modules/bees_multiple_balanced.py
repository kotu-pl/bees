import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import pytorch_lightning as pl
import tensorflow_datasets as tfds
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
from torchvision.transforms import RandomResizedCrop, RandomHorizontalFlip, RandomRotation, ColorJitter, Resize, Compose, ToTensor, Normalize
from sklearn.model_selection import StratifiedShuffleSplit
from torchvision.datasets.folder import default_loader
import numpy as np

import gdown
import zipfile
import os.path as osp

from .transforms_utils import ResizePad224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

class MultiLabelDataset(Dataset):
    def __init__(self, samples, labels, transform=None):
        self.samples = samples
        self.labels = labels
        self.transform = transform
        self.loader = default_loader

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = self.loader(self.samples[idx])
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(self.labels[idx]).float()

class BeesMultipleBalancedDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers, data_dir: str = '', zip_path: str = '',  resize_pad_224: bool = False, augmentation: bool = False):
        super().__init__()
        self.data_dir = data_dir
        self.zip_name = zip_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transform = []
        self.eval_transform = []

        aug_transform = [
            RandomResizedCrop(224, scale=(0.8, 1.0)),
            RandomHorizontalFlip(p=0.5),
            RandomRotation(degrees=15),
            ColorJitter(0.2, 0.2, 0.2, 0.1)
        ]
        base_transform = [
            ToTensor(), Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ]

        if resize_pad_224:
            self.train_transform.append(ResizePad224())
            self.eval_transform.append(ResizePad224())

        if augmentation:
            self.train_transform.extend(aug_transform)

        self.train_transform.extend(base_transform)
        self.eval_transform.extend(base_transform)

        self.train_transform = Compose(self.train_transform)
        self.eval_transform  = Compose(self.eval_transform)

    def prepare_data(self):
        if not osp.isfile(self.zip_name):
            gdown.download('https://drive.google.com/uc?id=1mhpNh6t741xbVciUOPz0xznh_DwZWoK3', output=self.zip_name, quiet=False)

        if not osp.isdir(self.data_dir):
            with zipfile.ZipFile(self.zip_name, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)

    def setup(self, stage=None):
        img_paths, labels = self.load_multilabel_data()

        # losowy podział, z równowagą, na train i temp
        indices = list(range(len(img_paths)))
        strat_labels = [tuple(label) for label in labels]

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=10)
        train_idx, tmp_idx = next(sss.split(indices, strat_labels))

        # podział temp na val i test, 2:1
        tmp_labels = [strat_labels[i] for i in tmp_idx]
        sss_val = StratifiedShuffleSplit(n_splits=1, test_size=1/3, random_state=42)
        val_idx, test_idx = next(sss_val.split(tmp_idx, tmp_labels))

        self.train_dataset = MultiLabelDataset(
            [img_paths[i] for i in train_idx],
            [labels[i] for i in train_idx],
            transform=self.train_transform
        )

        self.val_dataset = MultiLabelDataset(
            [img_paths[tmp_idx[i]] for i in val_idx],
            [labels[tmp_idx[i]] for i in val_idx],
            transform=self.eval_transform
        )

        self.test_dataset = MultiLabelDataset(
            [img_paths[tmp_idx[i]] for i in test_idx],
            [labels[tmp_idx[i]] for i in test_idx],
            transform=self.eval_transform
        )

        self.train_dataset.dataset.transform = self.train_transform
        self.val_dataset.dataset.transform = self.eval_transform
        self.test_dataset.dataset.transform = self.eval_transform

        # sampler z wagami
        if stage in ("fit", None):
            class_counts = np.sum([labels[i] for i in train_idx], axis=0)
            weights = 1.0 / (class_counts + 1e-6)
            sample_weights = np.array([np.sum(labels[i] * weights) for i in train_idx])
            self.train_sampler = WeightedRandomSampler(
                sample_weights, len(sample_weights), replacement=True
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def load_multilabel_data(self):
        class_names = sorted(os.listdir(self.data_dir))
        class_to_idx = {cls: i for i, cls in enumerate(class_names)}

        img_to_labels = {}
        for cls in class_names:
            folder = os.path.join(self.data_dir, cls)
            for img_path in glob(os.path.join(folder, '*')):
                if img_path not in img_to_labels:
                    img_to_labels[img_path] = np.zeros(len(class_names), dtype=float)
                img_to_labels[img_path][class_to_idx[cls]] = 1

        img_paths = list(img_to_labels.keys())
        labels    = list(img_to_labels.values())
        return img_paths, labels
