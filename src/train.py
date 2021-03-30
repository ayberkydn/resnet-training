import hydra
from omegaconf import DictConfig, OmegaConf
import os
import sys
import pdb
import pathlib
import torchlayers
import torch
import logging
import torchvision


import pytorch_lightning as pl

from src.models.bottleneck_resnet50 import BottleneckResnet50

from src.MyLightningModule import LightningModule
from src.datamodules.imagenet import ImagenetDataModule
from src.datamodules.tinyimagenet import TinyImagenetDataModule

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.gpu_stats_monitor import GPUStatsMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


def train(cfg):

    callbacks = [
        GPUStatsMonitor(),
        EarlyStopping(
            monitor="validation_accuracy",
            mode="max",
            patience=cfg.hparams.early_stopping_patience,
        ),
        ModelCheckpoint(
            dirpath=os.path.join(cfg.project_root, "model-checkpoints"),
            filename="epoch{epoch}model",
            monitor="validation_accuracy",
            mode="max",
        ),
    ]

    logger = TensorBoardLogger(
        save_dir=os.path.join(cfg.project_root, "logs"),
        name=cfg.name,
    )

    trainer = pl.Trainer(
        # limit_train_batches=0.015,
        # limit_val_batches=0.02,
        progress_bar_refresh_rate=50,
        gpus=cfg.trainer.gpus,
        accelerator="ddp",
        callbacks=callbacks,
        logger=logger,
    )

    datamodule = hydra.utils.instantiate(cfg.datamodule)
    model      = hydra.utils.instantiate(cfg.model)

    pl_module = LightningModule(model=model, cfg=cfg)

    trainer.fit(
        pl_module,
        datamodule,
    )
