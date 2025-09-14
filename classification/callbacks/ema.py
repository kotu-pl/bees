# classification/callbacks/ema.py
from copy import deepcopy
import torch
from pytorch_lightning import Callback

class EMACallback(Callback):
    def __init__(self, decay: float = 0.9999, device: str | None = None):
        self.decay = decay
        self.device = device
        self.ema_state = None
        self._swapped = False

    @torch.no_grad()
    def on_fit_start(self, trainer, pl_module):
        self.ema_state = {k: v.detach().clone() for k, v in pl_module.state_dict().items()}
        if self.device:
            for k in self.ema_state:
                self.ema_state[k] = self.ema_state[k].to(self.device)

    @torch.no_grad()
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        model_sd = pl_module.state_dict()
        for k, v in model_sd.items():
            v_ema = self.ema_state[k]
            v_ema.mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    @torch.no_grad()
    def _swap_in(self, pl_module):
        if self._swapped: return
        self._backup = {k: v.detach().clone() for k, v in pl_module.state_dict().items()}
        pl_module.load_state_dict(self.ema_state, strict=False)
        self._swapped = True

    @torch.no_grad()
    def _swap_out(self, pl_module):
        if not self._swapped: return
        pl_module.load_state_dict(self._backup, strict=False)
        self._swapped = False

    def on_validation_start(self, trainer, pl_module): self._swap_in(pl_module)
    def on_validation_end(self, trainer, pl_module):   self._swap_out(pl_module)
    def on_test_start(self, trainer, pl_module):       self._swap_in(pl_module)
    def on_test_end(self, trainer, pl_module):         self._swap_out(pl_module)
