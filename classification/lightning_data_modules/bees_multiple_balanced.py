import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import pytorch_lightning as pl
import tensorflow_datasets as tfds
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
from torchvision.transforms import RandomResizedCrop, RandomHorizontalFlip, RandomRotation, ColorJitter, Resize, Compose, ToTensor, Normalize
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

import gdown
import zipfile
import os.path as osp

from .transforms_utils import ResizePad224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

class BeesMultipleBalancedDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers, data_dir: str = '', resize_pad_224: bool = False, augmentation: bool = False):
        super().__init__()
        self.data_dir = data_dir
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
        self.builder = tfds.builder('bee_dataset', data_dir=self.data_dir)
        self.builder.download_and_prepare()

    def setup(self, stage=None):
        full_dataset = self.builder.as_dataset(
            split=['train'], as_supervised=False, shuffle_files=True
        ).map(add_bee_output).map(rename_labels)

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

        self.train_dataset.dataset.transform = self.train_transform
        self.val_dataset.dataset.transform = self.eval_transform
        self.test_dataset.dataset.transform = self.eval_transform

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

    def add_bee_output(example):
        outputs = example['output']
        sum_other = (
            outputs['cooling_output'] +
            outputs['pollen_output'] +
            outputs['varroa_output'] +
            outputs['wasps_output']
        )
        bee = tf.cast(tf.equal(sum_other, 0.0), tf.float32)
        outputs['bee_output'] = bee
        example['output'] = outputs
        return example

    def rename_labels(example):
        outputs = example['output']
        renamed = {}
        for key, value in outputs.items():
            new_key = key.replace('_output', '')
            renamed[new_key] = value
        example['output'] = renamed
        return example

