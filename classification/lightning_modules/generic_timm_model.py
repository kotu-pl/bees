import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim

import pytorch_lightning as pl
import torchmetrics

class GenericTimmLitModel(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-3, freeze_backbone: bool = True, loss_fn: str = "bce"):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.loss_fn = loss_fn.lower()

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
        y = y.float()

        if self.loss_fn == "bce":
            return F.binary_cross_entropy_with_logits(
                x, y, pos_weight=getattr(self, "pos_weight", None)
            )

        elif self.loss_fn == "dice":
            probs = torch.sigmoid(x)
            smooth = 1e-6
            intersection = (probs * y).sum()
            dice = (2. * intersection + smooth) / (probs.sum() + y.sum() + smooth)
            return 1 - dice

        elif self.loss_fn == "focal":
            gamma = 2.0
            alpha = 0.25
            bce = F.binary_cross_entropy_with_logits(
                x, y, reduction='none', pos_weight=getattr(self, "pos_weight", None)
            )
            pt = torch.exp(-bce)
            focal = alpha * (1 - pt) ** gamma * bce
            return focal.mean()

        elif self.loss_fn == "hinge":
            y_hinge = y * 2 - 1  # z {0,1} do {-1,1}
            return F.multi_label_margin_loss(torch.sigmoid(x), y_hinge.long())

        else:
            raise ValueError(f"Nieznana funkcja straty: {self.loss_fn}")

    def common_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = self.compute_loss(outputs, y)
        return loss, outputs, y

    def common_test_valid_step(self, batch, batch_idx):
        loss, outputs, y = self.common_step(batch, batch_idx)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).int()
        acc = torchmetrics.functional.accuracy(
            preds, y.int(), task="multilabel", num_labels=outputs.size(1)
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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate,
                                      weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.3, patience=1, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val_loss",   # <-- kluczowe!
            },
        }

#    def configure_optimizers(self):
#        # tylko trenowalne parametry (czyli fc po zamrożeniu reszty)
#        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
#        return optimizer

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        logits = self(x)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).int()
        return preds

    def on_fit_start(self):
        pos_w = getattr(self.trainer.datamodule, "pos_weight", None)
        if pos_w is not None:
            self.register_buffer("pos_weight", pos_w.to(self.device))
        else:
            self.pos_weight = None
