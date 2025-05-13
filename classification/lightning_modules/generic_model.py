import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim

import pytorch_lightning as pl
import torchmetrics

from timm.models.resnet import ResNet
from timm.models.densenet import DenseNet
from timm.models.regnet import RegNet
from timm.models.efficientnet import EfficientNet
from timm.models.mobilenetv3 import MobileNetV3
from timm.models.convnext import ConvNeXt
from timm.models.swin_transformer import SwinTransformer
from timm.models.vision_transformer import VisionTransformer

class GenericLitModel(pl.LightningModule):
    def __init__(self, model, num_classes, learning_rate=1e-3, freeze_backbone: bool = True):
        super().__init__()
        self.save_hyperparameters()

        self.model = model
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.training_type = training_type.lower()

        # stosowne zakończenie sieci
        self.adjust_fc_layer()

        # zamrożenie modeli
        if freeze_backbone:
            self.freeze_model()

    def freeze_model(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def adjust_fc_layer(self):
        if isinstance(self.model, (ResNet, RegNet)):
            in_f = self.model.fc.in_features
            self.model.fc = nn.Linear(in_f, self.num_classes)
        elif isinstance(self.model, DenseNet):
            in_f = self.model.classifier.in_features
            self.model.classifier = nn.Linear(in_f, self.num_classes)
        elif isinstance(self.model, EfficientNet):
            in_f = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(in_f, self.num_classes)
        elif isinstance(self.model, MobileNetV3):
            in_f = self.model.classifier[3].in_features
            self.model.classifier[3] = nn.Linear(in_f, self.num_classes)
        elif isinstance(self.model, ConvNeXt):
            in_f = self.model.classifier[2].in_features
            self.model.classifier[2] = nn.Linear(in_f, self.num_classes)
        elif isinstance(self.model, (SwinTransformer, VisionTransformer)):
            in_f = self.model.head.in_features
            self.model.head = nn.Linear(in_f, self.num_classes)
        else:
            raise ValueError("Nieznana struktura klasyfikatora")

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
