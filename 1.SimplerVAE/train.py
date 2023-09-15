# -*- coding: utf-8 -*-
"""
Created on 2022/10/28 9:36

@author: Yang Fan

模型训练代码，使用pytorch-lightning包装，wandb做实验记录，实验配置文件位于"config-defaults.yaml"中
"""
import os
import pickle
from typing import List

import pandas as pd
import pytorch_lightning as pl
import torch
import wandb
from dataset_rand import Dataset
from model import VAE
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from utils import VAELoss, rank_ics

# wandb offline mode
os.environ["WANDB_API_KEY"] = "f7bb19bc81f5d12f426a67949f434e397dd1dbc0"
os.environ["WANDB_MODE"] = "offline"


class Model(pl.LightningModule):
    """
    Wrapping the training process with pytorch-lightning
    """

    def __init__(
        self,
        C: int,
        H: int,
        K: int,
        M: int,
        gamma: float,
        alpha: float,
    ):
        super(Model, self).__init__()
        self.save_hyperparameters()
        self.model = VAE(
            feature_num=C,
            embedding_dim=H,
            latent_dim=K,
            decoder_layers=M,
        )
        self.loss_fn = VAELoss(gamma=gamma, alpha=alpha)

    def configure_optimizers(self):
        """
        Chose "SGD" or "Adam" to optimize the model
        Returns
        -------

        """
        if wandb.config.optim == "sgd":
            opt = torch.optim.SGD(
                self.model.parameters(),
                lr=wandb.config.lr,
                weight_decay=wandb.config.weight_decay,
                nesterov=wandb.config.use_nesterov,
                momentum=wandb.config.momentum,
            )
        else:
            opt = torch.optim.Adam(
                self.model.parameters(),
                lr=wandb.config.lr,
                weight_decay=wandb.config.weight_decay,
            )

        lr_scheduler = {
            # "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
            #     opt, T_max=wandb.config.max_epochs, eta_min=wandb.config.lr_min
            # ),
            "scheduler": torch.optim.lr_scheduler.MultiStepLR(
                opt,
                milestones=wandb.config.milestones,
                gamma=0.1,
            ),
            "interval": "epoch",
            "frequency": 1,
        }

        return [opt], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        feature = torch.nan_to_num(batch[:, :, 0:-1].float(), nan=0)
        label = torch.nan_to_num(batch[:, -1, -1].float(), nan=0)
        y_hat, z = self.model(feature)
        loss, reconstruction_loss, embedding_loss, rank_loss = self.loss_fn(
            y_hat, label, z
        )
        score = rank_ics(y_hat, label)
        self.log("train_loss", loss, on_step=True)
        self.log("reconstruction_loss", reconstruction_loss, on_step=True)
        self.log("embedding_loss", embedding_loss, on_step=True)
        self.log("rank_loss", rank_loss, on_step=True)
        self.log("train_ic", score, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        feature = torch.nan_to_num(batch[:, :, 0:-1].float(), nan=0)
        label = torch.nan_to_num(batch[:, -1, -1].float(), nan=0)
        y_hat, z = self.model(feature)

        loss, reconstruction_loss, embedding_loss, rank_loss = self.loss_fn(
            y_hat, label, z
        )
        score = rank_ics(y_hat, label)
        self.log("val_loss", reconstruction_loss, on_step=True, on_epoch=True)
        self.log("val_ic", score, on_epoch=True, prog_bar=True)
        return (
            label.detach().cpu(),
            y_hat.detach().cpu(),
        )

    def validation_epoch_end(self, outputs):
        label, pred = zip(*outputs)
        label = pd.Series(torch.cat(label))
        pred = pd.Series(torch.cat(pred))
        torch.save(label, "label.pt")
        torch.save(pred, "pred.pt")
        skew_label, kurt_label = label.skew(), label.kurt()
        skew_pred, kurt_pred = pred.skew(), pred.kurt()
        self.log("skew_label", skew_label, on_epoch=True)
        self.log("kurt_label", kurt_label, on_epoch=True)
        self.log("skew_pred", skew_pred, on_epoch=True)
        self.log("kurt_pred", kurt_pred, on_epoch=True)

    def test_step(self, batch, batch_idx):
        feature = torch.nan_to_num(batch[:, :, 0:-1].float(), nan=0)
        label = torch.nan_to_num(batch[:, -1, -1].float(), nan=0)
        y_hat, z = self.model(feature)

        loss, reconstruction_loss, embedding_loss, rank_loss = self.loss_fn(
            y_hat, label, z
        )
        score = rank_ics(y_hat, label)
        self.log("test_loss", loss, on_step=True, on_epoch=True)
        self.log("test_ic", score, on_epoch=True)

        return (
            label.detach().cpu(),
            y_hat.detach().cpu(),
        )

    def test_epoch_end(self, outputs):
        label, pred = zip(*outputs)
        label = pd.Series(torch.cat(label))
        pred = pd.Series(torch.cat(pred))
        torch.save(label, "test_label.pt")
        torch.save(pred, "test_pred.pt")
        skew_label, kurt_label = label.skew(), label.kurt()
        skew_pred, kurt_pred = pred.skew(), pred.kurt()
        self.log("test_skew_label", skew_label, on_epoch=True)
        self.log("test_kurt_label", kurt_label, on_epoch=True)
        self.log("test_skew_pred", skew_pred, on_epoch=True)
        self.log("test_kurt_pred", kurt_pred, on_epoch=True)


class StockData(pl.LightningDataModule):
    """
    Warpping the dataloader by LightningDataModule.
    """

    def __init__(
        self,
        data_dir: List[str],
        num_workers: int,
        batch_size: int,
        train_date: List[int],
        val_date: List[int],
        test_date: List[int],
        step_len: int,
        col_filter_path: str = None,
    ):
        super(StockData, self).__init__()
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.train_date = train_date
        self.val_date = val_date
        self.test_date = test_date
        self.step_len = step_len
        self.col_filter_path = col_filter_path
        if self.col_filter_path is not None:
            with open(self.col_filter_path, "r") as f:
                self.col = pickle.load(f)

    def prepare_data(self):
        if self.col_filter_path is not None:
            self.raw_data = pd.read_hdf(
                self.data_dir[0], key="data", columns=self.col
            )
        else:
            self.raw_data = pd.read_hdf(self.data_dir[0], key="data")

    def setup(self, stage=None):
        if stage == "fit":
            self.train_dataset = Dataset(
                raw_data=self.raw_data,
                data_path=self.data_dir,
                step_len=self.step_len,
                start=self.train_date[0],
                end=self.train_date[1],
            )
            self.val_dataset = Dataset(
                raw_data=self.raw_data,
                data_path=self.data_dir,
                step_len=self.step_len,
                start=self.val_date[0],
                end=self.val_date[1],
            )
        if stage == "test":
            self.test_dataset = Dataset(
                raw_data=self.raw_data,
                data_path=self.data_dir,
                step_len=self.step_len,
                start=self.test_date[0],
                end=self.test_date[1],
            )

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return test_loader


if __name__ == "__main__":
    wandb_logger = WandbLogger(name="hypergraph", project="SimplerVAE")
    seed_everything(seed=wandb.config.seed)
    dataloader = StockData(
        data_dir=wandb.config.data_dir,
        batch_size=wandb.config.batch_size,
        num_workers=wandb.config.num_workers,
        train_date=wandb.config.train_date,
        val_date=wandb.config.val_date,
        test_date=wandb.config.test_date,
        step_len=wandb.config.step_len,
        col_filter_path=wandb.config.col_filter_path,
    )
    if dataloader.col_filter_path is not None:
        wandb.config.update(
            {"C": int(len(dataloader.col) - 2)}, allow_val_change=True
        )
    model = Model(
        C=wandb.config.C,
        H=wandb.config.H,
        M=wandb.config.M,
        K=wandb.config.K,
        gamma=wandb.config.gamma,
        alpha=wandb.config.alpha,
    )
    wandb.watch(model.model, log="all")
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint = ModelCheckpoint(monitor="val_ic", mode="max")
    early_stop = EarlyStopping(monitor="val_ic", mode="max", patience=3)
    trainer = pl.Trainer(
        callbacks=[lr_monitor, checkpoint],
        max_epochs=wandb.config.max_epochs,
        logger=wandb_logger,
        gpus=-1,
    )
    trainer.fit(
        model=model,
        datamodule=dataloader,
    )
