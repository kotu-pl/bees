import os
import sys
import logging
import hydra
import uuid
from hydra.utils import instantiate
from hydra.utils import get_original_cwd, to_absolute_path

from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl

import torch
# lightning related imports
import pytorch_lightning as pl
#from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
    
OmegaConf.register_new_resolver("uuid", lambda *args: uuid.uuid4().hex)

@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # debugowanie
    # print(OmegaConf.to_yaml(cfg))

    data_module = instantiate(cfg.data)
    data_module.prepare_data()
    data_module.setup()

    # Init our model
    model = instantiate(cfg.network)

    # Initialize trainer
    trainer = instantiate(cfg.trainer)

    overrides = HydraConfig.get().overrides.task
    trainer.logger.experiment.config.update(
        {"overrides": overrides,
         "seed": cfg.seed,
         "trainer": {"max_epochs": cfg.trainer.max_epochs}},
        allow_val_change=True
    )

    # Train the model
    trainer.fit(model, data_module)

    # Evaluate the model on the held out test set ⚡⚡
    trainer.test(model=model, datamodule=data_module)

if __name__ == "__main__":
    main()
