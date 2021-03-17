import hydra
from omegaconf import DictConfig, OmegaConf
import os
import sys
import pdb
import pathlib
import torchlayers
import torch
import logging


import pytorch_lightning as pl


from models import Resnet50
from MyLightningModule import LightningModule
from data import ImagenetDataModule

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.gpu_stats_monitor import GPUStatsMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="config")
def main(cfg):

    project_path = cfg.project_path
    log_path = os.path.join(project_path, "logs")
    ckp_path = os.path.join(project_path, "model-checkpoints")
    data_path = os.path.join(project_path, "data")
    callbacks = [
        GPUStatsMonitor(),
        EarlyStopping(
            monitor="validation_accuracy",
            mode="max",
            patience=cfg.hparams.early_stopping_patience,
        ),
        ModelCheckpoint(
            dirpath=ckp_path,
            filename="epoch{epoch}model",
            monitor="validation_accuracy",
            mode="max",
        ),
    ]

    logger = TensorBoardLogger(
        save_dir=log_path,
        name=cfg.name,
    )

    trainer = pl.Trainer(
        # limit_train_batches=0.015,
        # limit_val_batches=0.02,
        progress_bar_refresh_rate=50,
        gpus=1,
        accelerator="ddp",
        callbacks=callbacks,
        logger=logger,
    )

    model = LightningModule(model=Resnet50())

    datamodule = ImagenetDataModule(
        path=cfg.dataset_path,
        batch_size=cfg.hparams.batch_size,
        num_workers_factor=cfg.num_workers_factor,
    )

    trainer.fit(model, datamodule)


if __name__ == "__main__":
    main()
