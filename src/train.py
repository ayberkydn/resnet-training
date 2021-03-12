import os, sys, pdb

import argparse 

import torch, torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import kornia
import pytorch_lightning as pl
import torchfunc, torchlayers, torchsummary

from einops import rearrange, reduce, repeat
import numpy as np
import matplotlib.pyplot as plt

from torchvision import transforms


from models import Resnet50
from MyLightningModule import LightningModule


project_path = os.environ["PROJECT_ROOT"]






from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.gpu_stats_monitor import GPUStatsMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


gpu_stats = GPUStatsMonitor()
early_stopping = EarlyStopping(monitor='validation_accuracy', 
                               patience=3, 
                               verbose=True, 
                               mode='max')

log_path = os.path.join(project_path, 'logs')
ckp_path = os.path.join(project_path, 'model-checkpoints')

print(log_path)

tb_logger = TensorBoardLogger(
    save_dir=log_path,
    name='my_MODEL'
)
checkpoint = ModelCheckpoint(dirpath=ckp_path,
    filename='epoch{epoch}model',
    monitor='validation_accuracy',
    mode='max'
)



model = LightningModule(model=Resnet50())


#trainer = pl.Trainer(
#    #limit_train_batches=0.015,
#    #limit_val_batches=0.02,
#    gpus=2,
#    accelerator='ddp',
#    callbacks=[gpu_stats, early_stopping, checkpoint],
#    logger = tb_logger)
#
#trainer.fit(model, training_loader, validation_loader)

