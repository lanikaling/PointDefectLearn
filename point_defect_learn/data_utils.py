import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
from sympy import *
from typing import *
from scipy.interpolate import interp1d


def process_pdf_input_single(r, g, dim, rmin, rmax):
    """Reprocess the PDF into PDF input for the CNN model.
    The feature input to the CNN is a 209 x 1 array, with r-range from 1.5 A to 30. A

    Parameters
    ----------
    r: 1-D numpy array, shape (num_peaks,)
        Distances in the raw PDF data. Assuming r is sorted from smallest to largest.
    g : 1-D numpy array, shape (num_peaks,)
        Peak intensity in the raw PDF data.
    Return:
    ----------
    input_PDF: 2-D numpy array, shape (209, 1)
        reprocessed PDF data to input to the CNN model.
    """
    # cut off the PDF at 30 A
    cut_off_indices = np.where((r <= rmin) & (r >= rmax))
    r_cut = r[cut_off_indices]
    g_cut = g[cut_off_indices]

    # interpolating peak intensity in the input cnn r range
    # any peak in the cnn r-range that falls outside of the input range
    # i.e. if r_cut[0] > 1.5 or r_cut[-1] < 30. will be set to 0
    pdf_interp = interp1d(
        r_cut, g_cut, kind="quadratic", bounds_error=False, fill_value=0.0
    )
    r_range = np.linspace(rmin, rmax, dim)
    input_pdf = pdf_interp(r_range)

    return input_pdf


def get_pdf_filenames(directory):
    """
    Recursively collect .gr file paths in a given directory and all its subdirectories,
    excluding directories named 'pure_metal_supercell'.

    Parameters:
    directory (str): The path to the directory to explore.

    Returns:
    list: A list of .gr file paths found in the directory and its subdirectories,
          excluding those in 'pure_metal_supercell'.
          Paths are relative to the initial directory provided.
    """
    result = []
    # Walk through the directory and its subdirectories
    for root, dirs, files in os.walk(directory):
        # Skip the directory 'pure_metal_supercell'
        if "pure_metal_supercell" in os.path.split(root):
            continue
        # Filter and append .gr files to the result list
        for file in files:
            if file.endswith(".gr"):
                # Construct the full path
                full_path = os.path.join(root, file)
                # Make the path relative to the initial directory
                relative_path = os.path.relpath(full_path, directory)
                result.append(relative_path)
    return result


def split_data(
    pdf_filenames_list,
    train_ratio,
    val_ratio,
    if_save=False,
    save_directory=None,
):
    """
    Splits a list of PDF filenames into training, validation, and testing subsets.

    Parameters:
    - data_directory (str): The base directory where the data is stored (not used in this simplified version).
    - pdf_filenames_list (list): A list of .gr file paths.
    - train_ratio (float): The proportion of the dataset to include in the train split.
    - val_ratio (float): The proportion of the dataset to include in the validation split.
    - if_save (bool, optional): Whether to save the split data to a JSON file. Defaults to False.
    - save_directory (str, optional): The directory where to save the split data if if_save is True.

    Returns:
    - dict: A dictionary with keys "train", "val", and "test", each containing a list of file paths.
    """

    # Initialize a dictionary to hold the split filenames
    data_dict = {"train": [], "val": [], "test": []}

    # Perform the split
    filenames_train, filenames_temp = train_test_split(
        pdf_filenames_list, test_size=1 - train_ratio, random_state=7
    )
    filenames_val, filenames_test = train_test_split(
        filenames_temp, test_size=val_ratio / (1 - train_ratio), random_state=7
    )

    # Fill the data_dict with the splits
    data_dict["train"] = filenames_train
    data_dict["val"] = filenames_val
    data_dict["test"] = filenames_test

    # Save to JSON file if required
    if if_save:
        if save_directory is None:
            print("Save directory not specified.")
        else:
            # Ensure the save directory exists
            os.makedirs(save_directory, exist_ok=True)
            with open(
                os.path.join(save_directory, "data_split.json"), "w"
            ) as f:
                json.dump(data_dict, f)
    else:
        return data_dict


def count_data_types(file_path, key="train"):
    # Initialize counts for each type
    counts = {"vac": 0, "sub": 0, "selfint": 0, "int": 0}

    # Try to read the JSON file and handle errors
    try:
        with open(file_path, "r") as f:
            data = json.load(f)

        # Get the specified key data list
        data_list = data[key]

        # Count each type in the data list
        for filename in data_list:
            if "vac" in filename:
                counts["vac"] += 1
            elif "sub" in filename:
                counts["sub"] += 1
            elif "selfint" in filename:
                counts["selfint"] += 1
            elif "int" in filename:
                counts["int"] += 1
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None
    except KeyError:
        print(f"Error: The key '{key}' is not in the data.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

    # Prepare the result list in the specified order
    result_counts = [
        counts["vac"],
        counts["sub"],
        counts["selfint"],
        counts["int"],
    ]
    return result_counts
