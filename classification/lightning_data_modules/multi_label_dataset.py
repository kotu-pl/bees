import torch
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset

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
