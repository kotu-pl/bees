import torch
import pytorch_lightning as pl

class ThresholdFinder(pl.Callback):
    def __init__(self, metric="f1", step=0.01):
        super().__init__()
        self.metric = metric
        self.step = step
        self._probs = []
        self._targets = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if outputs is None:
            return
        self._probs.append(outputs["probs"].cpu())
        self._targets.append(outputs["targets"].cpu())

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self._probs:
            return

        probs = torch.cat(self._probs)
        targets = torch.cat(self._targets)

        best_thr = []
        for c in range(probs.size(1)):
            p, t = probs[:, c], targets[:, c]
            f1_scores = []
            thresholds = torch.arange(0.0, 1.0 + self.step, self.step)
            for thr in thresholds:
                preds = (p > thr).int()
                tp = (preds * t).sum().float()
                fp = (preds * (1 - t)).sum().float()
                fn = ((1 - preds) * t).sum().float()
                f1 = (2 * tp) / (2 * tp + fp + fn + 1e-8)
                f1_scores.append(f1.item())
            best_thr.append(thresholds[torch.argmax(torch.tensor(f1_scores))].item())

        pl_module.decision_thresholds = torch.tensor(best_thr, device=pl_module.device)

        trainer.logger.log_metrics({
            "optimal_thresholds_mean": pl_module.decision_thresholds.mean().item()
        }, step=trainer.global_step)

        # wyczyść na następną epokę
        self._probs.clear()
        self._targets.clear()
