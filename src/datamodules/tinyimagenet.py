import pytorch_lightning as pl
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import os

class TinyImagenetDataModule(pl.LightningDataModule):
    def __init__(self, path, batch_size, num_workers_factor):
        super().__init__()
        self.path = path
        self.batch_size = batch_size
        self.num_workers_factor = num_workers_factor

        self.transform_train = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )

        self.transform_validation = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        self.training_set = datasets.ImageFolder(
            root=os.path.join(self.path, "train"), transform=self.transform_train
        )

        self.validation_set = datasets.ImageFolder(
            root=os.path.join(self.path, "val"), transform=self.transform_validation
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.training_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers_factor,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.validation_set,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=self.num_workers_factor * 2,
            pin_memory=True,
        )

