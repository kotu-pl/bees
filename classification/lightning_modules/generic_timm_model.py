import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim

import pytorch_lightning as pl
import torchmetrics

class GenericTimmLitModel(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-3, freeze_backbone: bool = True):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.learning_rate = learning_rate

        # zamrożenie modeli
        if freeze_backbone:
            self.freeze_backbone()

        # Zakończenie sieci ogarnia `timm` w trakcie tworzenia modelu
        # self.adjust_fc_layer()

    def freeze_backbone(self):
        # zamrożenie wszystkiego
        for param in self.model.parameters():
            param.requires_grad = False
        # odmrożenie classifier/fc
        for p in self.model.get_classifier().parameters():
            p.requires_grad = True

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
        # liczbę klas wyciągam z wektora logitów
        num_classes = outputs.size(1)
        acc = torchmetrics.functional.accuracy(
            preds, y, num_classes=num_classes, task="multiclass"
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
