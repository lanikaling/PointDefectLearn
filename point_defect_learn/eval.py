import sys
import yaml, json
import torch
import pprint
from munch import munchify
import numpy as np
import os
import glob
from pytorch_lightning import Trainer, seed_everything
from tensorboard.backend.event_processing import event_accumulator
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from distortion_learning.models import Conv2DModel
from distortion_learning.dataset import DistortionLearningDataset
from distortion_learning.plot_utils import plot_cm


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


def load_model(epoch=None, print_cfg=False):
    cfg = load_config(filepath="../distortion_learning/config.yaml")
    if print_cfg:
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
    checkpoint_file_pattern = f"epoch={epoch}*.ckpt" if epoch else "*.ckpt"
    checkpoint_filepath = glob.glob(
        os.path.join(log_dir + "/checkpoints", checkpoint_file_pattern)
    )[0]
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

    # load trained model
    ckpt = torch.load(checkpoint_filepath, map_location=torch.device("cpu"))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    model.freeze()
    return model, log_dir, checkpoint_filepath, cfg


def main(epoch=None, print_cfg=False):
    model, log_dir, _, _ = load_model(epoch=epoch, print_cfg=print_cfg)
    # define trainer
    trainer = Trainer(
        accelerator="mps",
        # devices=cfg.num_gpus,
        deterministic=True,
        default_root_dir=log_dir + "/preds",
    )
    trainer.test(model)


def retrieve_data(
    print_cfg=False,
    version=0,
):
    cfg = load_config(filepath="../distortion_learning/config.yaml")
    if print_cfg:
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
    log_dir = os.path.join(
        log_dir, "lightning_logs", "version_{}".format(version)
    )
    log_filepath = glob.glob(os.path.join(log_dir, "events.out.*"))[0]
    print("log_filepath: " + log_filepath)
    ea = event_accumulator.EventAccumulator(log_filepath)
    ea.Reload()
    print("Scalar Keys:", ea.Tags()["scalars"])

    data = {}
    scalar_keys = ea.Tags()["scalars"]
    for key in scalar_keys:
        if key.endswith("epoch"):
            values_only = [s.value for s in ea.Scalars(key)]
            data[key] = np.array(values_only)
        else:
            steps_and_values = [(s.step, s.value) for s in ea.Scalars(key)]
            data[key] = np.array(steps_and_values)
        # Convert list of tuples to NumPy array

    return data


def pred_single(filename, epoch=None, print_cfg=False):
    model, log_dir, checkpoint_filepath, cfg = load_model(
        epoch=epoch, print_cfg=print_cfg
    )
    print(checkpoint_filepath)
    dataset = DistortionLearningDataset(
        filenames=[filename], path_to_data=cfg.data_filepath
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    data_point, target = next(iter(dataloader))
    with torch.no_grad():
        logits = model(data_point)
        probabilities = softmax(
            logits, dim=1
        )  # Apply softmax to convert logits to probabilities
    print("Ground Truth:", target.numpy())
    print("Probabilities:", probabilities.numpy())
    predicted_class = torch.argmax(
        probabilities, dim=1
    )  # Get the predicted class
    print("Predicted Class:", predicted_class.item() + 1)
    return probabilities


def compute_accuracy(
    dataset_label,
    epoch=None,
    print_cfg=False,
):
    """
    Computes the accuracy of the model on the specified dataset.

    Parameters:
    - model: The trained model.
    - dataloaders: A dictionary containing DataLoader objects for 'train', 'val', and 'test'.
    - dataset_label: A string specifying which dataset to use ('train', 'val', or 'test').

    Returns:
    - accuracy: The computed accuracy on the specified dataset.
    """
    model, log_dir, checkpoint_filepath, cfg = load_model(
        epoch=epoch, print_cfg=print_cfg
    )
    print(checkpoint_filepath)
    model.eval()

    with open("../distortion_learning/data_split.json") as f:
        data = json.load(f)

    dataset = DistortionLearningDataset(
        data[dataset_label],
        cfg.data_filepath,
        seed=cfg.seed,
        noisy=cfg.input_noise,
        noise_scale=cfg.noise_scale,
        normalization=cfg.normalization,
        loss=cfg.loss,
    )

    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

    correct = 0
    top2_correct = 0
    top3_correct = 0
    total = 0

    with torch.no_grad():
        for data, labels in dataloader:
            outputs = model(data)
            labels_idx = labels.argmax(
                dim=1
            )  # Convert one-hot labels to indices

            # Calculate top-k accuracies
            _, predicted_topk = outputs.topk(3, dim=1)  # Get top-3 predictions
            total += labels.size(0)

            # Check for correctness
            correct += (
                (predicted_topk[:, :1] == labels_idx.unsqueeze(1)).sum().item()
            )
            top2_correct += (
                (predicted_topk[:, :2] == labels_idx.unsqueeze(1)).sum().item()
            )
            top3_correct += (
                (predicted_topk[:, :3] == labels_idx.unsqueeze(1)).sum().item()
            )

    accuracy = 100 * correct / total
    top2_accuracy = 100 * top2_correct / total
    top3_accuracy = 100 * top3_correct / total

    print(f"Accuracy on the {dataset_label} set: {accuracy:.2f}%")
    print(f"Top-2 accuracy on the {dataset_label} set: {top2_accuracy:.2f}%")
    print(f"Top-3 accuracy on the {dataset_label} set: {top3_accuracy:.2f}%")

    return {
        "accuracy": accuracy,
        "top2_accuracy": top2_accuracy,
        "top3_accuracy": top3_accuracy,
    }


def plot_confusion_matrix(
    dataset_label,
    ticklabels,
    figsize=(10, 8),
    epoch=None,
    print_cfg=False,
    if_plot=True,
    if_save=False,
    save_dir=None,
):
    """
    Plots or saves the confusion matrix of the model's predictions on the specified dataset.

    Parameters:
    - model: The trained model, already loaded.
    - dataset_label: A string specifying which dataset to use ('train', 'val', or 'test').
    - if_plot: Boolean, whether to plot the confusion matrix.
    - if_save: Boolean, whether to save the confusion matrix plot.
    - save_dir: Directory where to save the confusion matrix plot.
    """
    model, log_dir, checkpoint_filepath, cfg = load_model(
        epoch=epoch, print_cfg=print_cfg
    )
    print(checkpoint_filepath)
    device = next(model.parameters()).device  # Get the device the model is on

    # Load dataset configuration and initialize DataLoader
    with open("../distortion_learning/data_split.json") as f:
        data = json.load(f)
    dataset = DistortionLearningDataset(
        data[dataset_label],
        cfg.data_filepath,
        seed=cfg.seed,
        noisy=cfg.input_noise,
        noise_scale=cfg.noise_scale,
        normalization=cfg.normalization,
        loss=cfg.loss,
    )
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

    all_predictions = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for data, labels in dataloader:
            outputs = model(data)
            _, predicted = torch.max(outputs, dim=1)  # For top-1 prediction
            all_predictions.extend(predicted.cpu().numpy())

            # Convert one-hot encoded labels to indices if necessary
            if labels.ndim > 1:
                # Assuming one-hot encoding if labels have more than 1 dimension
                labels = labels.argmax(dim=1)
            all_labels.extend(labels.cpu().numpy())

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)

    if if_plot or if_save:
        plot_cm(
            cm, ticklabels, dataset_label, figsize, if_plot, if_save, save_dir
        )


if __name__ == "__main__":
    main()
