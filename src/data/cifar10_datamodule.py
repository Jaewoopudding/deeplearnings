from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms


class CIFAR10DataModule(LightningDataModule):
    def __init__(self,
                 data_dir: str = "data/",
                 train_val_test_split: Tuple[int, int, int] = (55_000, 5_000, 10_000),
                 batch_size: int = 64,
                 num_workers: int = 0,
                 pin_memory: bool = True,
                 persistent_workers: bool = True):
        self.save_hyperparameters(logger=False) # self.hparams activated
        self.transforms = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
        
    def prepare_data(self):
        CIFAR10(self.hparams.data_dir, train=True, download=True)
        CIFAR10(self.hparams.data_dir, train=False, download=True)
        
    def setup(self, stage):
        if not self.data_train and not self.data_val and not self.data_test:
            trainset = CIFAR10(self.hparams.data_dir, train=True, transforms=self.transforms)
            validset = CIFAR10(self.hparams.data_dir, train=False, transforms=self.transforms)
            dataset = ConcatDataset([trainset, validset])
            self.data_train, self.data_valid, self.data_test = random_split(
                dataset=dataset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42)
            )
            
    def train_dataloader(self):
        return DataLoader(dataset=self.data_train,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory,
                          persistent_workers=self.hparams.persistent_workers,
                          shuffle=True)
        
    def val_dataloader(self):
        return DataLoader(dataset=self.data_valid,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory,
                          persistent_workers=self.hparams.persistent_workers,
                          shuffle=False)
        
    def test_dataloader(self):
        return DataLoader(dataset=self.data_test,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory,
                          persistent_workers=self.hparams.persistent_workers,
                          shuffle=False)
        
        
if __name__ == '__main__':
    _ = CIFAR10DataModule()