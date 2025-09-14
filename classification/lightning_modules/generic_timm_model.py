import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim

import pytorch_lightning as pl
import torchmetrics
from torchmetrics.classification import (
    MultilabelF1Score, MultilabelPrecision, MultilabelRecall, MultilabelAveragePrecision, MultilabelHammingDistance
)
from pytorch_lightning.loggers import WandbLogger
try:
    from timm.loss import AsymmetricLossMultiLabel
except Exception:
    from timm.loss.asymmetric_loss import AsymmetricLossMultiLabel

class GenericTimmLitModel(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-3, weight_decay=1e-4, freeze_backbone: bool = False, loss_fn: str = "bce", eval_tta: bool = False):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.loss_fn = loss_fn.lower()
        self.asl = None
        self.eval_tta = eval_tta
        if self.loss_fn == "asl":
            self.asl = AsymmetricLossMultiLabel(gamma_neg=1.0, gamma_pos=0.0, clip=0.05, eps=1e-8)

        self.model = model
        head_names = ("head", "classifier", "fc", "head.fc", "last_linear")
        backbone_modules = [m for n, m in self.model.named_children() if n not in head_names]
        self.backbone = nn.Sequential(*backbone_modules) if backbone_modules else self.model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.num_labels = getattr(self.model, "num_classes", None)
        if self.num_labels is None:
            self.num_labels = self.model.get_classifier().out_features

        # Inicjalizacja metryk mAP
        self.val_mean_ap = MultilabelAveragePrecision(num_labels=self.num_labels, average="macro")
        self.val_f1 = MultilabelF1Score(num_labels=self.num_labels, average="macro", threshold=0.5)
        self.val_precision = MultilabelPrecision(num_labels=self.num_labels, average="macro", threshold=0.5)
        self.val_recall = MultilabelRecall(num_labels=self.num_labels, average="macro", threshold=0.5)
        self.val_hamming = MultilabelHammingDistance(num_labels=self.num_labels, threshold=0.5)

        # zamrożenie modeli
        if freeze_backbone:
            self.freeze_backbone()

        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[DEBUG] trainable at start: {n_train}")

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
        elif self.loss_fn == 'asl':
            return self.asl(x, y)
        elif self.loss_fn == "hinge":
            y_hinge = y * 2 - 1  # z {0,1} do {-1,1}
            return F.multilabel_margin_loss(torch.sigmoid(x), y_hinge.long())

        else:
            raise ValueError(f"Nieznana funkcja straty: {self.loss_fn}")

    def common_step(self, batch, batch_idx, eval_mode: bool = False):
        x, y = batch
        outputs = self._inference_logits(x) if eval_mode and self.eval_tta else self(x)
        loss = self.compute_loss(outputs, y)
        return loss, outputs, y

    def common_test_valid_step(self, batch, batch_idx, eval_mode: bool = False):
        loss, outputs, y = self.common_step(batch, batch_idx, eval_mode)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).int()
        acc = torchmetrics.functional.accuracy(
            preds, y.int(), task="multilabel", num_labels=outputs.size(1)
        )
        return loss, acc, probs, y

    def training_step(self, batch, batch_idx):
        loss, acc, _, _ = self.common_test_valid_step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc, probs, y = self.common_test_valid_step(batch, batch_idx, eval_mode=True)
        self.val_mean_ap.update(probs, y.int())
        self.val_f1.update(probs, y.int())
        self.val_precision.update(probs, y.int())
        self.val_recall.update(probs, y.int())
        self.val_hamming.update(probs, y.int())

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    # obliczanie i logowanie metryk na końcu końcu epoki walidacyjnej
    def on_validation_epoch_end(self):
        mean_ap = self.val_mean_ap.compute()
        f1 = self.val_f1.compute()
        prec = self.val_precision.compute()
        rec = self.val_recall.compute()
        hl = self.val_hamming.compute()

        self.log("val_mean_ap", mean_ap, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_f1", f1, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_precision", prec, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_recall", rec, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_hamming", hl, on_step=False, on_epoch=True, sync_dist=True)

        self.val_mean_ap.reset()
        self.val_f1.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_hamming.reset()

    def test_step(self, batch, batch_idx):
        loss, acc, _, _ = self.common_test_valid_step(batch, batch_idx, eval_mode=True)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.3, patience=1#, verbose=True
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

    def _inference_logits(self, x: torch.Tensor) -> torch.Tensor:
        return (self.model(x) + self.model(torch.flip(x, dims=[-1]))) / 2

    def on_fit_start(self):
        pos_w = getattr(self.trainer.datamodule, "pos_weight", None)
        if pos_w is not None:
            self.register_buffer("pos_weight", pos_w.to(self.device))
        else:
            self.pos_weight = None

        # definicja agregacji max w WandB
        if isinstance(self.logger, WandbLogger):
            exp = self.logger.experiment
            exp.define_metric("val_mean_ap", summary="max")
            exp.define_metric("val_f1", summary="max")
            exp.define_metric("val_precision", summary="max")
            exp.define_metric("val_recall", summary="max")
            exp.define_metric("val_hamming", summary="min")
