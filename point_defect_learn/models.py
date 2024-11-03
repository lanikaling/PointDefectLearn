import pytorch_lightning as pl
import os
import torch
import numpy as np
import json
import shutil
from torch import nn
from torch.utils.data import DataLoader
from distortion_learning.model_utils import Conv2DRobustClassifier
from distortion_learning.dataset import DistortionLearningDataset
from distortion_learning.data_utils import count_data_types


def mkdir(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)


class Conv2DModel(pl.LightningModule):
    def __init__(
        self,
        lr: float = 1e-4,
        seed: int = 1,
        if_cuda: bool = False,
        if_test: bool = False,
        log_dir: str = "logs",
        num_workers: int = 8,
        train_batch: int = 32,
        val_batch: int = 32,
        test_batch: int = 32,
        model_name: str = "conv2d",
        loss: str = "CrossEntropy",
        data_filepath: str = "../data/pdf",
        input_noise: bool = False,
        noise_scale: float = 0.0,
        dropout_rate: float = 0.3,
        normalization="sample",
    ):
        super().__init__()
        self.save_hyperparameters()

        self.input_noise = input_noise
        self.noise_scale = noise_scale
        self.normalization = normalization
        self.dropout_rate = dropout_rate
        self.lr = lr
        self.seed = seed
        self.if_cuda = if_cuda
        self.if_test = if_test
        self.num_workers = num_workers
        self.loss = loss
        self.log_dir = os.path.join(log_dir, "preds")

        if not self.if_test:
            mkdir(self.log_dir)

        self.train_batch = train_batch
        self.val_batch = val_batch
        self.test_batch = test_batch
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        self.data_filepath = data_filepath
        self.model_name = model_name
        self.__build_model()

    def __build_model(self):
        # Model
        if self.model_name == "conv2d_robust":  # original default model
            self.model = Conv2DRobustClassifier(self.dropout_rate)
        else:
            raise NameError(f"{self.model_name} is not supported")

        # Loss
        if self.loss == "MSE":
            self.loss_func = nn.MSELoss()
        elif self.loss == "CrossEntropy":
            # vac, sub, selfint, int
            file_path = "../distortion_learning/data_split.json"
            class_counts = count_data_types(file_path, "train")
            class_counts = torch.tensor(class_counts, dtype=torch.float32)
            # Calculate weights: Inverse of the frequency
            weights = 1.0 / class_counts
            # Normalize weights such that the smallest weight is 1.0
            weights = weights / weights.min()
            self.loss_func = nn.CrossEntropyLoss(weight=weights)
        else:
            raise ValueError(f"{self.loss} is not supported")

    def forward(self, x):
        return self.model(x)

    def on_after_backward(self):
        for name, param in self.named_parameters():
            if param.grad is not None:
                if torch.any(torch.isnan(param.grad)):
                    print(f"NaN gradient in {name}")

    def compute_loss(self, output, target):
        return self.loss_func(output, target)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        train_loss = self.compute_loss(outputs, targets)
        if train_loss is None or torch.isnan(train_loss):
            print(f"NaN in training loss at batch index {batch_idx}.")
        self.log(
            "train_loss_step",
            train_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )
        self.training_step_outputs.append(train_loss)
        return train_loss

    def on_train_epoch_end(self, unused=None):
        # Filter out None values which represent batches where NaN was detected
        valid_losses = [
            loss for loss in self.training_step_outputs if loss is not None
        ]
        if valid_losses:
            train_loss_epoch = torch.stack(valid_losses).mean()
        else:
            train_loss_epoch = torch.tensor(float("nan"))

        self.log(
            "train_loss_epoch",
            train_loss_epoch,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.training_step_outputs.clear()  # Clearing the list for the next epoch

        lr = self.optimizers().param_groups[0]["lr"]
        self.log(
            "learning_rate",
            lr,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        val_loss = self.compute_loss(outputs, targets)
        if val_loss is None or torch.isnan(val_loss):
            print(f"NaN in validation loss at batch index {batch_idx}.")
        self.log(
            "val_loss_step",
            val_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )  # Log validation loss
        self.validation_step_outputs.append(val_loss)
        return val_loss

    def on_validation_epoch_end(self):
        # Filter out None values which represent batches where NaN was detected
        valid_losses = [
            loss for loss in self.validation_step_outputs if loss is not None
        ]
        if valid_losses:
            val_loss_epoch = torch.stack(valid_losses).mean()
        else:
            val_loss_epoch = torch.tensor(float("nan"))

        self.log(
            "val_loss_epoch",
            val_loss_epoch,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.validation_step_outputs.clear()  # Clearing the list for the next epoch

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        test_loss = self.compute_loss(outputs, targets)
        if test_loss is None or torch.isnan(test_loss):
            print(f"NaN in test loss at batch index {batch_idx}.")
        self.log(
            "test_loss",
            test_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )  # Log test loss
        self.test_step_outputs.append(test_loss)
        return test_loss

    def on_test_epoch_end(self):
        test_losses = [
            loss for loss in self.test_step_outputs if loss is not None
        ]
        if test_losses:
            test_loss_epoch = torch.stack(test_losses).mean()
        else:
            test_loss_epoch = torch.tensor(float("nan"))
        print(test_loss_epoch.item())
        self.log(
            "test_loss_epoch",
            test_loss_epoch,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.test_step_outputs.clear()  # Clearing the list for the next epoch

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_schedule, gamma=self.gamma)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=0.85,
            patience=5,
            threshold=0.005,
            threshold_mode="rel",
            min_lr=1e-7,
        )

        return [optimizer], [
            {"scheduler": scheduler, "monitor": "val_loss_epoch"}
        ]

    # setup dataloaders
    def setup(self, stage=None):
        # split data into train, val, test
        with open("../distortion_learning/data_split.json") as f:
            data = json.load(f)
        train_files = data["train"]
        test_files = data["test"]
        val_files = data["val"]

        if stage == "fit":
            self.train_dataset = DistortionLearningDataset(
                train_files,
                self.data_filepath,
                seed=self.seed,
                noisy=self.input_noise,
                noise_scale=self.noise_scale,
                normalization=self.normalization,
                loss=self.loss,
            )

            self.val_dataset = DistortionLearningDataset(
                val_files,
                self.data_filepath,
                seed=self.seed,
                normalization=self.normalization,
                loss=self.loss,
            )

        if stage == "test":
            self.test_dataset = DistortionLearningDataset(
                test_files,
                self.data_filepath,
                seed=self.seed,
                normalization=self.normalization,
                loss=self.loss,
            )

    def train_dataloader(self):
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.train_batch,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.val_batch,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=self.test_batch,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )
        return test_loader
