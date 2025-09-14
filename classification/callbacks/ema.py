import pytorch_lightning as pl
from timm.utils import ModelEmaV2

class EmaCallback(pl.Callback):
    def __init__(self, decay=0.9998):
        self.decay = decay
        self.ema = None
        self._swap = None

    def on_fit_start(self, trainer, pl_module):
        # zakładam, że właściwy model siedzi pod pl_module.model
        self.ema = ModelEmaV2(pl_module.model, decay=self.decay, device=pl_module.device)

    def on_after_backward(self, trainer, pl_module):
        self.ema.update(pl_module.model)

    def on_validation_start(self, trainer, pl_module):
        self._swap = pl_module.model
        pl_module.model = self.ema.module  # walidujemy/tesujemy na EMA

    def on_validation_end(self, trainer, pl_module):
        pl_module.model = self._swap
