from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
import re
import os
from scipy.interpolate import interp1d


class DistortionLearningDataset(Dataset):
    def __init__(
        self,
        filenames,
        path_to_data,
        seed=1,
        noisy=False,
        noise_scale=0.01,
        normalization="sample",
        loss="CrossEntropy",
    ):
        self.filenames = filenames
        self.path_to_data = path_to_data
        self.seed = seed
        self.noisy = noisy  # if training, apply input noise?
        self.normalization = normalization
        self.loss = loss

        ### future optimization: make this a parameter
        if self.noisy:
            self.noise_scale = noise_scale

    # return dataset size
    def __len__(self):
        return len(self.filenames)

    # retrieve item from dataset
    def __getitem__(self, index):
        x_distorted = self.load_x(self.filenames[index])

        # Generate original filename and load original PDF
        original_filename = self.get_original_filename(self.filenames[index])
        x_original = self.load_x(original_filename)

        # Convert numpy arrays to PyTorch tensors
        x_distorted = torch.from_numpy(x_distorted).float()
        x_original = torch.from_numpy(x_original).float()

        # Ensure both tensors have the same second dimension
        length = min(x_distorted.size(0), x_original.size(0))
        x_distorted = x_distorted[:length]
        x_original = x_original[:length]

        # Concatenate x_original and x_distorted to make shape (2, len(x))
        x_combined = torch.stack([x_original, x_distorted], dim=0).reshape(
            (2, length)
        )

        y = self.classify_defect_type(self.filenames[index])

        # if self.loss == "BCE":
        # mask labels
        # y[y > 1e-4] = 1
        # y[y <= 1e-4] = 0

        # Data loading logic
        if torch.any(torch.isnan(x_combined)) or torch.any(torch.isnan(y)):
            print(f"NaN detected in loaded data for index {index}")

        return x_combined, y

    def load_x(self, filename):
        r, x = np.loadtxt(
            os.path.join(self.path_to_data, filename),
            unpack=True,
        )
        x = self.process_pdf_input_single(r, x)
        if self.normalization == "sample":
            x = (x - np.min(x)) / (np.max(x) - np.min(x))
        # elif self.normalization == "global":
        # x = (x + 0.002529386025064303) / (3.8752065410913166)
        if self.noisy:
            x += (
                torch.randn(x.shape) * self.noise_scale
            )  # self.noisy_scale = std
        return x

    def process_pdf_input_single(self, r, g, dim=300, rmin=1.5, rmax=30):
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
        cut_off_indices = np.where((r <= rmax) & (r >= rmin))
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

    def get_original_filename(self, filename):
        # Step 1: Split the string to isolate the base part and the extension
        base, extension = filename.rsplit("/", 1)

        # Step 2: Replace the defect type with "supercell" in the base part
        base_transformed = re.sub(
            r"pure_metal_(selfint|int|vac|sub)", "pure_metal_supercell", base
        )

        # Step 3: Reassemble the string, keeping only up to the "dim5" part
        # and assuming all filenames follow the same pattern
        # This uses a regular expression to find the appropriate part of the string
        match = re.search(r"(icsd_\d+_dim5)", extension)
        if match:
            transformed_filename = f"{base_transformed}/{match.group(1)}.gr"
            return transformed_filename
        else:
            print(f"Unexpected filename format: {filename}")
            return None

    def classify_defect_type(self, s):
        """
        Classifies a string based on whether it contains 'sub', 'vac', 'int', or 'selfint',
        and returns a corresponding PyTorch tensor.

        Parameters:
        s (str): The input string to classify.

        Returns:
        torch.Tensor: A tensor representing the class of the defect.
        """
        # Define the pattern for matching
        pattern = r"(sub|vac|int|selfint)"
        match = re.search(pattern, s)

        if match:
            if match.group(0) == "vac":
                return torch.from_numpy(np.array([1, 0, 0, 0])).float()
            elif match.group(0) == "sub":
                return torch.from_numpy(np.array([0, 1, 0, 0])).float()
            elif match.group(0) == "selfint":
                return torch.from_numpy(np.array([0, 0, 1, 0])).float()
            elif match.group(0) == "int":
                return torch.from_numpy(np.array([0, 0, 0, 1])).float()
        else:
            print("No matching defect type found.")
            return None
