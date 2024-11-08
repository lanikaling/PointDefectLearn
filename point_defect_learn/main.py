import sys
import yaml
import os
import torch
import pprint
from munch import munchify

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from point_defect_learn.models import Conv2DModel


def load_config(filepath):
    with open(filepath, "r") as stream:
        try:
            trainer_params = yaml.safe_load(stream)
            return trainer_params
        except yaml.YAMLError as exc:
            print(exc)


def seed(cfg):
    torch.manual_seed(cfg.seed)
    if cfg.if_cuda:
        torch.cuda.manual_seed(cfg.seed)


def main():
    # load model config for training
    # config_filepath = str(sys.argv[1])
    # cfg = load_config(filepath=config_filepath)
    cfg = load_config(filepath="../distortion_learning/config.yaml")
    pprint.pprint(cfg)
    cfg = munchify(cfg)
    seed(cfg)
    seed_everything(cfg.seed)

    log_foldername = "_".join(
        ["logs"]
        + (["noise"] if cfg.input_noise else [])
        + [cfg.loss, cfg.model_name, str(cfg.seed)],
    )
    log_dir = os.path.join(cfg.log_dir, log_foldername)

    model = Conv2DModel(
        lr=cfg.lr,
        seed=cfg.seed,
        num_workers=cfg.num_workers,
        if_cuda=cfg.if_cuda,
        if_test=False,
        log_dir=log_dir,
        train_batch=cfg.train_batch,
        val_batch=cfg.val_batch,
        test_batch=cfg.test_batch,
        model_name=cfg.model_name,
        loss=cfg.loss,
        data_filepath=cfg.data_filepath,
        input_noise=cfg.input_noise,
        noise_scale=cfg.noise_scale,
        dropout_rate=cfg.dropout_rate,
    )  # input

    # define callback for selecting checkpoints during training via lightining module
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(log_dir, "checkpoints"),
        monitor="val_loss_epoch",
        verbose=True,
        mode="min",
        save_top_k=3,  # Save the top 3 models
        filename="{epoch:02d}-{val_loss:.2f}",
    )

    # define trainer
    trainer = Trainer(
        accelerator="mps",
        # devices=cfg.num_gpus,
        max_epochs=cfg.epochs,
        # deterministic=True,
        default_root_dir=log_dir,
        val_check_interval=1.0,
        precision=cfg.precision,
        callbacks=checkpoint_callback,
    )
    trainer.fit(model)


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    main()
