from omegaconf import DictConfig
import torch
from torch import nn
import torch.optim as optim
import pytorch_lightning as pl
from torchvision import models
from torch.nn import functional as F

import torchmetrics

from torchvision import models

class GenericLitModel(pl.LightningModule):
    def __init__(self, model, num_classes, learning_rate=1e-3, training_type="transfer"):
        super().__init__()
        self.save_hyperparameters()

        self.model = model
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.training_type = training_type.lower()

        # stosowne zakończenie sieci
        self.adjust_fc_layer()

        # zamrożenie modeli
        if self.training_type == "freeze_model":
            self.freeze_model()

    def freeze_model(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def adjust_fc_layer(self):
        num_in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_in_features, self.num_classes)

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, x, y):
        return F.cross_entropy(x, y)

    def common_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = self.compute_loss(outputs, y)
        return loss, outputs, y

    def common_test_valid_step(self, batch, batch_idx):
        loss, outputs, y = self.common_step(batch, batch_idx)
        preds = torch.argmax(outputs, dim=1)
        acc = torchmetrics.functional.accuracy(
            preds, y, num_classes=self.num_classes, task="multiclass"
        )
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.common_test_valid_step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.common_test_valid_step(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, acc = self.common_test_valid_step(batch, batch_idx)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # tylko trenowalne parametry (czyli fc po zamrożeniu reszty)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
