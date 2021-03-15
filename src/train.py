import hydra
from omegaconf import DictConfig, OmegaConf
import os
import sys
import pdb
import pathlib
import torchlayers
import torch



import pytorch_lightning as pl


from models import Resnet50
from MyLightningModule import LightningModule
from data import ImagenetDataModule

project_path = pathlib.Path(os.path.realpath(__file__)).parent.parent

log_path = os.path.join(project_path, "logs")
ckp_path = os.path.join(project_path, "model-checkpoints")
data_path = os.path.join(project_path, "data")


@hydra.main(config_name='../configs/config.yaml')
def main(cfg):
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping
    from pytorch_lightning.callbacks.gpu_stats_monitor import GPUStatsMonitor
    from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger

    gpu_stats = GPUStatsMonitor()

    early_stopping = EarlyStopping(
        monitor="validation_accuracy",
        patience=cfg.hparams.early_stopping_patience,
        verbose=True,
        mode="max"
    )

    tb_logger = TensorBoardLogger(save_dir=log_path,
                                  name="my_MODEL")
    checkpoint = ModelCheckpoint(
        dirpath=ckp_path,
        filename="epoch{epoch}model",
        monitor="validation_accuracy",
        mode="max",
    )

    trainer = pl.Trainer(
        # limit_train_batches=0.015,
        # limit_val_batches=0.02,
        progress_bar_refresh_rate=50,
        gpus=1,
        accelerator="ddp",
        callbacks=[gpu_stats, early_stopping, checkpoint],
        logger=tb_logger,
    )

    model = LightningModule(model=Resnet50())
    datamodule = ImagenetDataModule(
        path=os.path.join(data_path, "ILSVRC/Data/CLS-LOC"), batch_size=16
    )

    trainer.fit(model, datamodule)


if __name__ == '__main__':
    main()
