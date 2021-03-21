import pytorch_lightning as pl
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import os


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, path, batch_size, num_workers_factor, pin_memory, download):
        super().__init__()
        self.path = path
        self.batch_size = batch_size
        self.num_workers_factor = num_workers_factor
        self.pin_memory = pin_memory


        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        # Normalize the test set same as training set without augmentation
        self.transform_validation = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.training_set = datasets.CIFAR10(
            root=self.path, train=True, download=download, transform=self.transform_train)

        self.validation_set = datasets.CIFAR10(
            root=self.path, train=False, download=download, transform=self.transform_validation)


    def train_dataloader(self):
        return DataLoader(
            dataset=self.training_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers_factor,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.validation_set,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=self.num_workers_factor * 2,
            pin_memory=self.pin_memory,
        )


