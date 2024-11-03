import numpy as np
import matplotlib.pyplot as plt
from bg_mpl_stylesheet.bg_mpl_stylesheet import bg_mpl_style
import seaborn as sns
import os


def plot_processed_and_original(
    filename,
    normalization=None,
    offset=2,
    directory="../data/pdf",
    if_show=True,
    if_save=False,
    save_dir=None,
):
    """
    Plots data from a processed file and its corresponding original file with enhanced title and save filename features.

    Parameters:
    - filename: The name of the processed file.
    - directory: Base directory where the files are located. Default is '../data/pdf'.
    - if_show: Boolean indicating whether to show the plot.
    - if_save: Boolean indicating whether to save the plot.
    - save_dir: Directory where to save the plot if if_save is True.
    """
    # Define the mappings for folder names to descriptions
    description_map = {
        "pure_metal_int": "Interstitial Impurities",
        "pure_metal_selfint": "Self-interstitial",
        "pure_metal_sub": "Substitutional Impurities",
        "pure_metal_vac": "Vacancy",
    }

    # Extract details from the filename
    folder, file_details = filename.split("/")
    defect_type = description_map.get(folder, "")
    original_filename_parts = file_details.split("_")
    atom_species = original_filename_parts[-2]
    percentage = original_filename_parts[-1][:-3].replace("p", ".") + "%"

    # Update the title and save_filename based on the extracted details
    title = f"{defect_type} ({atom_species}, {percentage})"
    save_filename = (
        f"{defect_type}_{atom_species}_{percentage}".replace(".", "p").replace(
            "%", ""
        )
        + ".png"
    )

    # Paths
    processed_dir = os.path.join(directory, filename)
    original_filename = "_".join(original_filename_parts[0:3]) + ".gr"
    original_dir = os.path.join(
        directory, "pure_metal_supercell", original_filename
    )

    # Load data
    x_processed, y_processed = np.loadtxt(processed_dir, unpack=True)
    x_original, y_original = np.loadtxt(original_dir, unpack=True)
    if normalization == "sample":
        y_processed = (y_processed - np.min(y_processed)) / (
            np.max(y_processed) - np.min(y_processed)
        )
        y_original = (y_original - np.min(y_original)) / (
            np.max(y_original) - np.min(y_original)
        )

    # Plot
    plt.figure(figsize=(10, 3))
    plt.style.use(bg_mpl_style)
    plt.plot(
        x_processed, y_processed, label="Distorted", linestyle="-", marker=""
    )
    plt.plot(
        x_original, y_original, label="Original", linestyle="--", marker=""
    )
    plt.plot(
        x_original,
        y_original - y_processed - offset,
        label="difference",
        linestyle="-.",
        marker="",
    )
    plt.legend()
    plt.xlabel("X Axis Label")
    plt.ylabel("Y Axis Label")
    plt.title(title)

    # Show or save
    if if_save and save_dir is not None:
        save_path = os.path.join(save_dir, save_filename)
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    if if_show:
        plt.show()
    plt.close()


def plot_eval_data(
    data,
    keys,
    figsize=(10, 5),
    title=None,
    ylabel=None,
    save=False,
    save_dir=None,
    save_filename=None,
):
    plt.style.use(bg_mpl_style)

    # Start a new figure
    fig, ax = plt.subplots(figsize=figsize)

    # Define common xlabel based on keys
    xlabel = "Steps" if any("step" in key for key in keys) else "Epoch"

    # Prepare Y-labels if not provided
    if ylabel is None:
        ylabel = ", ".join(
            {k.split("_")[0] for k in keys}
        )  # Create a set of unique ylabels

    for key in keys:
        if "step" in key:
            x = data[key][:, 0]
            y = data[key][:, 1]
        else:
            x = np.arange(data[key].size)
            y = data[key]

        ax.plot(x, y, label=key.replace("_", " "))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title:
        ax.set_title(title)

    ax.legend()

    if save and save_dir and save_filename:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        full_path = os.path.join(save_dir, save_filename)
        plt.savefig(full_path)

    plt.show()


def plot_cm(
    cm,
    ticklabels,
    dataset_label,
    figsize=(10, 8),
    if_plot=None,
    if_save=None,
    save_dir=None,
):
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=ticklabels,
        yticklabels=ticklabels,
    )
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title(f"Confusion Matrix for {dataset_label} Set")

    if if_save:
        if save_dir:
            save_path = os.path.join(
                save_dir, f"confusion_matrix_{dataset_label}.png"
            )
            plt.savefig(save_path)
            print(f"Confusion matrix saved to {save_path}")
        else:
            print("Save directory not provided. Skipping save.")

    if if_plot:
        plt.show()
    else:
        plt.close()
