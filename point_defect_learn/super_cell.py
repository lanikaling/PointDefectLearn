from typing import List, Tuple, Union
import numpy as np
import pandas as pd
import json
import os
import shutil
import tempfile
from tqdm import tqdm
from scipy.stats import norm
from diffpy.structure import Lattice, loadStructure, Structure
from diffpy.srreal.pdfcalculator import PDFCalculator
from pymatgen.io.cif import CifParser
from distortion_learning.constants import (
    METAL_ELEMENTS,
    ATOMIC_NUMBER,
    SG_LABELS,
    UISO_VALUES,
)
from distortion_learning.interstitial_sites import interstitial_sites


class SuperCell:
    def __init__(self, atoms_info=None, supercell_lattice=None, space_group=0):
        if atoms_info is None:
            atoms_info = pd.DataFrame(
                columns=["x", "y", "z", "element", "uiso", "occupancy"]
            )
        self._atoms_info = atoms_info
        self._supercell_lattice = supercell_lattice
        self._space_group = space_group

    @property
    def atoms_info(self) -> pd.DataFrame:
        return self._atoms_info

    @property
    def supercell_lattice(self) -> Lattice:
        return self._supercell_lattice

    @property
    def space_group(self) -> int:
        """
        The space group number of the unit cell. The number should be between
        1 and 230.
        """
        return self._space_group

    @property
    def xyz(self) -> np.ndarray:
        return self._atoms_info[["x", "y", "z"]].to_numpy()

    def set_supercell_lattice(self, new_lattice: Lattice):
        """
        Set the lattice parameters for the unit cell.
        """
        self._supercell_lattice = new_lattice

    def set_space_group(self, new_space_group: int):
        """
        Set the space group number for the unit cell. Ensure the number is between 1 and 230.
        """
        if 1 <= new_space_group <= 230:
            self._space_group = new_space_group
        else:
            raise ValueError("Space group number should be between 1 and 230.")

    def set_xyz(self, new_xyz: np.ndarray):
        """
        atoms_xyz : np.ndarray
            The information of the atoms in the finite cluster. The array
            has at least three columns, which specifies the xyz Cartesian
            coordinates of the atoms.
        atoms_num_of_electron : np.ndarray
            The number of electrons of the atoms. Default to be 1.
        atoms_uiso : np.ndarray
            The number of electrons of the atoms. Default to be 1/(8pi^2).
        atoms_occupancy : np.ndarray
            The number of electrons of the atoms. Default to be 1.
        """
        # Check if the number of rows in new_xyz matches the current number of rows in self._atoms_info
        self._atoms_info[["x", "y", "z"]] = new_xyz
        num_of_atoms = new_xyz.shape[0]

        if self._atoms_info["element"].isnull().all():
            self._atoms_info["element"] = ["Li"] * num_of_atoms
        if self._atoms_info["uiso"].isnull().all():
            self._atoms_info["uiso"] = (
                np.ones((num_of_atoms, 1)) * 1 / (8 * np.pi**2)
            )
        if self._atoms_info["occupancy"].isnull().all():
            self._atoms_info["occupancy"] = np.ones((num_of_atoms, 1))

    def set_element(self, new_element: List[str]):
        """
        Set the number of electrons for atoms.
        """
        if len(self._atoms_info.index) != len(new_element):
            raise ValueError(
                "The number of rows in new_element must match the number of atoms."
            )
        self._atoms_info["element"] = new_element

    def set_uiso(self, new_uiso: Union[int, float, np.ndarray]):
        """
        Set the isotropic atomic displacement parameter for atoms, replacing zeros
        with default values from UISO_VALUES based on the species of each atom.
        """
        # Ensure new_uiso is a numpy array for uniform handling
        new_uiso = np.array(new_uiso, dtype=float)

        # Check if the length matches
        if len(self._atoms_info.index) != len(new_uiso):
            raise ValueError(
                "The number of rows in new_uiso must match the number of atoms."
            )

        # Iterate through new_uiso to replace zeros
        for idx, uiso in enumerate(new_uiso):
            if uiso == 0:
                # Extract species (element symbol) from the corresponding atom info row
                species = self._atoms_info.iloc[idx]["element"]
                species = "".join([s for s in species if s.isalpha()])
                # Replace this zero uiso with a value from UISO_VALUES
                new_uiso[idx] = UISO_VALUES.get(
                    species, uiso
                )  # keep original uiso if species not in UISO_VALUES

        # Update the DataFrame
        self._atoms_info["uiso"] = new_uiso

    def set_occupancy(self, new_occupancy: Union[int, float, np.ndarray]):
        """
        Set the occupancy for atoms.
        """
        if len(self._atoms_info.index) != new_occupancy.shape[0]:
            raise ValueError(
                "The number of rows in new_occupancy must match the number of atoms."
            )
        self._atoms_info["occupancy"] = new_occupancy

    @staticmethod
    def cart_to_frac(cart_xyz: np.ndarray, lattice) -> np.ndarray:
        """
        Change the atom sites from Cartesian coordinates to fractional
        coordinates.

        Parameters:
        -----------
        atoms_info: np.ndarray
            The information of the atoms in the finite cluster. The array
            has at least three columns, which specifies the xyz Cartesian
            coordinates of the atoms.

        Returns:
        --------
        frac_atoms_info: np.ndarray
            Same as atoms_info, except the first three columns has been
            changed from Cartesian coordinates to fractional coordinates.
        """
        frac_xyz = np.matmul(cart_xyz, np.linalg.inv(lattice.stdbase))
        return frac_xyz

    @staticmethod
    def frac_to_cart(frac_xyz: np.ndarray, lattice) -> np.ndarray:
        """
        Change the atom sites from fractional coordinates to Cartesian
        coordinates.

        Parameters:
        -----------
        atoms_info: np.ndarray
            The information of the atoms in the finite cluster. The array
            has at least three columns, which specifies the xyz fractional
            coordinates of the atoms.

        Returns:
        --------
        cart_atoms_info: np.ndarray
            Same as atoms_info, except the first three columns has been
            changed from fractional coordinates to Cartesian coordinates.
        """
        cart_xyz = np.matmul(frac_xyz, lattice.stdbase)
        return cart_xyz

    @staticmethod
    def get_elements_from_cif(cif_directory: str, original=False):
        structure = SuperCell.my_loadStructure(cif_directory)
        if structure is None:
            return None
        if original:
            return list(structure.element)
        elements_set = set()  # Use a set to automatically remove duplicates
        for symbol in structure.element:
            element = "".join([s for s in symbol if s.isalpha()])
            elements_set.add(element)

        elements_list = list(elements_set)  # Convert back to list if necessary
        return elements_list

    @staticmethod
    def is_pure_metal(elements_list: list) -> bool:
        if len(elements_list) > 1:
            return False
        return all(element in METAL_ELEMENTS for element in elements_list)

    @staticmethod
    def is_alloy(elements_list: list) -> bool:
        # An alloy must have at least two different elements, and at least one must be a metal.
        if len(elements_list) < 2:
            return False  # Not an alloy if it has less than two elements
        return any(
            element in METAL_ELEMENTS for element in elements_list
        )  # At least one element must be a metal

    @staticmethod
    def process_cif_files(source_dir, pure_metal_dir, alloy_dir):
        for filename in tqdm(os.listdir(source_dir)):
            if filename.endswith(".cif"):
                full_path = os.path.join(source_dir, filename)
                elements_list = SuperCell.get_elements_from_cif(full_path)
                if elements_list is None:
                    continue
                if SuperCell.is_pure_metal(elements_list):
                    destination = os.path.join(pure_metal_dir, filename)
                    shutil.move(full_path, destination)
                elif SuperCell.is_alloy(elements_list):
                    destination = os.path.join(alloy_dir, filename)
                    shutil.move(full_path, destination)

    @staticmethod
    def my_loadStructure(cif_directory: str):
        # Read and clean the file content
        with open(cif_directory, "r", errors="ignore") as file:
            content = file.read()

        # Create a temporary file
        temp_fd, temp_path = tempfile.mkstemp()
        with os.fdopen(temp_fd, "w") as temp_file:
            temp_file.write(content)

        # Load the structure from the temporary file
        try:
            structure = loadStructure(temp_path)
        except:
            print("Not able to load {}".format(cif_directory))
            return None

        # Clean up by closing the temporary file and deleting it
        os.remove(temp_path)
        return structure

    @staticmethod
    def remove_dup_xyz(atoms_info, decimal_precision=2):
        # Round 'x', 'y', 'z' to the specified number of decimal places and store in new columns
        atoms_info["x_rounded"] = atoms_info["x"].round(decimal_precision)
        atoms_info["y_rounded"] = atoms_info["y"].round(decimal_precision)
        atoms_info["z_rounded"] = atoms_info["z"].round(decimal_precision)

        # Group by the rounded values and the 'element' column, then calculate the mean for each group
        grouped = atoms_info.groupby(
            ["element", "x_rounded", "y_rounded", "z_rounded"]
        )
        averaged_df = grouped.mean().reset_index()

        # Drop the rounded columns as they are no longer needed
        final_df = averaged_df.drop(
            columns=["x_rounded", "y_rounded", "z_rounded"]
        )

        return final_df

    def load_cif(self, cif_directory: str):
        # Parsing the cif file
        parser = CifParser(cif_directory)
        structure_df = parser.as_dict()
        structure_df = structure_df[list(structure_df.keys())[0]]
        self._space_group = structure_df["_symmetry_Int_Tables_number"]

        structure = SuperCell.my_loadStructure(cif_directory)
        self._supercell_lattice = structure.lattice
        self.set_xyz(structure.xyz_cartn)
        self.set_element(list(structure.element))
        self.set_uiso(structure.Uisoequiv)
        self.set_occupancy(structure.occupancy)

        self._atoms_info = SuperCell.remove_dup_xyz(
            self._atoms_info, decimal_precision=2
        )
        return self.atoms_info

    def save_cif(self, target_file_path):
        # Convert Cartesian coordinates to fractional
        atoms_info_frac = self._atoms_info.copy()
        atoms_info_frac[["x", "y", "z"]] = self.cart_to_frac(
            atoms_info_frac[["x", "y", "z"]].to_numpy(),
            self._supercell_lattice,
        )

        # Start constructing the CIF content
        cif_content = f"""
data_my_structure
_symmetry_space_group_name_H-M    '{SG_LABELS[str(self._space_group)]}'
_cell_length_a    {self._supercell_lattice.a}
_cell_length_b    {self._supercell_lattice.b}
_cell_length_c    {self._supercell_lattice.c}
_cell_angle_alpha    {self._supercell_lattice.alpha}
_cell_angle_beta     {self._supercell_lattice.beta}
_cell_angle_gamma    {self._supercell_lattice.gamma}
_symmetry_Int_Tables_number {self._space_group}

loop_
 _atom_site_label
 _atom_site_type_symbol
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 _atom_site_occupancy
 _atom_site_U_iso_or_equiv
"""

        # Adding atom information from fractional DataFrame
        for index, atom in atoms_info_frac.iterrows():
            cif_content += f"  {atom['element']}{index}  {atom['element']}  {atom['x']}  {atom['y']}  {atom['z']}  {atom['occupancy']}  {atom['uiso']}\n"

        # Write the CIF content to a file
        with open(target_file_path, "w") as file:
            file.write(cif_content)

    def load_json(self, source_file_path):
        with open(source_file_path, "r") as f:
            data = json.load(f)

        # Convert atoms_info back into a DataFrame
        self._atoms_info = pd.DataFrame(
            data["atoms_info"], columns=data["columns"]
        )

        # Create a Lattice object from the supercell_lattice data
        lattice_params = data["supercell_lattice"]
        self._supercell_lattice = Lattice(**lattice_params)

        # Extract the space_group
        self._space_group = data["space_group"]
        return self._atoms_info

    def save_json(self, target_file_path):
        # Convert atoms_info DataFrame to a compact list of lists
        compact_atoms_info = self._atoms_info.values.tolist()

        # Extracting _supercell_lattice attributes
        lattice_attributes = {
            "a": self._supercell_lattice.a,
            "b": self._supercell_lattice.b,
            "c": self._supercell_lattice.c,
            "alpha": self._supercell_lattice.alpha,
            "beta": self._supercell_lattice.beta,
            "gamma": self._supercell_lattice.gamma,
        }

        # Prepare data for saving in a more compact format
        data_to_save = {
            "atoms_info": compact_atoms_info,
            "columns": self._atoms_info.columns.tolist(),  # Include column names for reconstruction
            "supercell_lattice": lattice_attributes,
            "space_group": self._space_group,
        }

        # Save the data in the target directory
        with open(target_file_path, "w") as f:
            json.dump(data_to_save, f, indent=4)

    @staticmethod
    def calc_num_electrons(symbol_list: List[str]) -> np.ndarray:
        num_electrons = np.zeros(len(symbol_list))
        for i, symbol in enumerate(symbol_list):
            if symbol[-1] not in ["+", "-"]:
                num_electrons[i] = ATOMIC_NUMBER[symbol]
            else:
                species = "".join([s for s in symbol if s.isalpha()])
                species_val = ATOMIC_NUMBER[species]
                charges = "".join([s for s in symbol if s.isdigit()])
                charges = float(charges) if charges else 1
                sign = 1 if symbol[-1] == "+" else -1
                num_electrons[i] = species_val - sign * charges
        return num_electrons

    def generate_supercell(self, dim: int, replace_original=False):
        # Convert to fractional coordinates
        xyz = self.atoms_info[["x", "y", "z"]].to_numpy()
        xyz_frac = self.cart_to_frac(xyz, self._supercell_lattice)
        atoms_info_frac = self.atoms_info.copy()
        atoms_info_frac[["x", "y", "z"]] = xyz_frac

        # Pre-compute the translation vectors for the supercell generation
        translation_vectors = [
            np.array([i, j, k])
            for i in range(dim)
            for j in range(dim)
            for k in range(dim)
        ]

        # Generate the supercell using broadcasting and vectorization
        supercell_atoms_list = []
        for vector in translation_vectors:
            translated_atoms = atoms_info_frac.copy()
            translated_atoms[["x", "y", "z"]] += vector
            supercell_atoms_list.append(translated_atoms)
        supercell_info = pd.concat(supercell_atoms_list, ignore_index=True)

        # Remove duplicates based on fractional coordinates
        supercell_info = SuperCell.remove_dup_xyz(
            supercell_info, decimal_precision=2
        )
        supercell_info[["x", "y", "z"]] = self.frac_to_cart(
            supercell_info[["x", "y", "z"]], self._supercell_lattice
        )

        if replace_original:
            self._atoms_info = supercell_info
            self._supercell_lattice.a *= dim
            self._supercell_lattice.b *= dim
            self._supercell_lattice.c *= dim

        return supercell_info

    @staticmethod
    def generate_supercell_multi(
        dim: int,
        source_directory: str,
        target_directory: str,
        file_list: str = None,
    ):
        if file_list:
            with open(file_list, "r") as f:
                file_dict = json.load(f)
                source_files = []
                for _, files in file_dict.items():
                    for file in files:
                        source_files.append(file)
        else:
            source_files = []
            for root, dirs, files in os.walk(source_directory):
                for file in files:
                    if file.endswith(".cif"):
                        source_files.append(os.path.join(root, file))

        for source_file_path in tqdm(source_files):
            # Construct the target directory structure
            relative_path = os.path.relpath(
                os.path.dirname(source_file_path), source_directory
            )
            target_root = os.path.join(target_directory, relative_path)
            os.makedirs(target_root, exist_ok=True)

            file = os.path.basename(source_file_path)
            if file.endswith(".cif"):
                # Process the .cif file
                sc = SuperCell()
                sc.load_cif(source_file_path)
                sc.generate_supercell(dim, replace_original=True)
                sc.save_json(
                    os.path.join(
                        target_root, file.replace(".cif", f"_dim{dim}.json")
                    )
                )

    @staticmethod
    def get_interstitial_sites_supercell(dim=5, decimal_precision=2):
        interstitial_sites_supercell = {}

        # Pre-compute the translation vectors for the supercell generation
        translation_vectors = [
            np.array([i, j, k])
            for i in range(dim)
            for j in range(dim)
            for k in range(dim)
        ]

        # Iterate through each key-value pair in the original dictionary
        for key, atoms in interstitial_sites.items():
            supercell_atoms_list = []

            # Apply each translation vector to the atoms coordinates
            for vector in translation_vectors:
                translated_atoms = atoms + vector
                supercell_atoms_list.append(translated_atoms)

            # Concatenate all translated atom positions
            concatenated_atoms = np.vstack(supercell_atoms_list)

            # Round the coordinates to two decimal places
            rounded_atoms = np.round(
                concatenated_atoms, decimals=decimal_precision
            )

            # Remove duplicates
            unique_atoms = np.unique(rounded_atoms, axis=0)

            # Store the unique atoms in the result dictionary
            interstitial_sites_supercell[key] = unique_atoms

        return interstitial_sites_supercell

    def apply_vacancy(self, ratio):
        # Calculate the number of rows to delete
        total_rows = len(self._atoms_info)
        rows_to_delete = int(np.floor(total_rows * ratio))

        # Randomly select row indices to delete
        indices_to_delete = np.random.choice(
            total_rows, rows_to_delete, replace=False
        )

        # Delete the selected rows
        self._atoms_info = self._atoms_info.drop(
            self._atoms_info.index[indices_to_delete]
        ).reset_index(drop=True)

    def apply_self_interstitial(self, interstitial_candidates, ratio):
        # Calculate the number of new interstitial rows to add
        total_rows = len(self._atoms_info)
        rows_to_add = int(np.floor(total_rows * ratio))

        # Get the element and Uiso from the existing atoms info
        element = self._atoms_info["element"][0]
        uiso = self._atoms_info["uiso"][0]

        # Convert existing fractional coordinates to Cartesian
        frac_xyz_total = self.cart_to_frac(
            self._atoms_info[["x", "y", "z"]], self._supercell_lattice
        )

        # Select a larger batch of rows initially
        selection_size = int(rows_to_add * 1.5)
        if selection_size > interstitial_candidates.shape[0]:
            print(
                "The number of interstitial atoms is larger than available interstitial sites."
            )
        selected_indices = np.random.choice(
            interstitial_candidates.shape[0], selection_size, replace=False
        )
        selected_rows = interstitial_candidates[selected_indices]

        # Filter out any rows that duplicate existing Cartesian coordinates
        unique_rows = []
        for row in selected_rows:
            if not np.any(
                np.all(np.isclose(frac_xyz_total, row, atol=1e-2), axis=1)
            ):
                unique_rows.append(row)
                if len(unique_rows) >= rows_to_add:
                    break

        # Convert the unique fractional coordinates to Cartesian
        frac_xyz = np.array(unique_rows)
        cart_xyz = self.frac_to_cart(frac_xyz, self._supercell_lattice)

        # Prepare DataFrame for new rows with the element, its Uiso, and full occupancy
        new_rows = pd.DataFrame(
            {
                "x": cart_xyz[:, 0],
                "y": cart_xyz[:, 1],
                "z": cart_xyz[:, 2],
                "element": [element] * rows_to_add,
                "uiso": [uiso] * rows_to_add,
                "occupancy": [1] * rows_to_add,
            }
        )

        # Append new interstitial rows to the existing atoms_info DataFrame
        self._atoms_info = pd.concat(
            [self._atoms_info, new_rows], ignore_index=True
        )

    def apply_substitutional_impurities(self, ratio, chosen_element=None):
        # Calculate the number of rows to modify
        total_rows = len(self._atoms_info)
        rows_to_modify = int(np.floor(total_rows * ratio))

        # Randomly select row indices to modify
        indices_to_modify = np.random.choice(
            total_rows, rows_to_modify, replace=False
        )

        # Randomly select a metal element
        if chosen_element is None:
            chosen_element = np.random.choice(list(METAL_ELEMENTS))

        # Modify the selected rows by changing their 'element' to the chosen element
        for idx in indices_to_modify:
            self._atoms_info.at[idx, "element"] = chosen_element
            self._atoms_info.at[idx, "uiso"] = UISO_VALUES[chosen_element]

        return chosen_element

    def apply_interstitial_impurities(
        self, interstitial_candidates, ratio, chosen_element=None
    ):
        # Calculate the number of new interstitial rows to add
        total_rows = len(self._atoms_info)
        rows_to_add = int(np.floor(total_rows * ratio))

        # Randomly select a metal element
        if chosen_element is None:
            chosen_element = np.random.choice(list(METAL_ELEMENTS))

        # Convert existing fractional coordinates to Cartesian
        frac_xyz_total = self.cart_to_frac(
            self._atoms_info[["x", "y", "z"]], self._supercell_lattice
        )

        # Select a larger batch of rows initially
        selection_size = int(rows_to_add * 1.5)
        if selection_size > interstitial_candidates.shape[0]:
            print(
                "The number of interstitial atoms is larger than available interstitial sites."
            )
        selected_indices = np.random.choice(
            interstitial_candidates.shape[0], selection_size, replace=False
        )
        selected_rows = interstitial_candidates[selected_indices]

        # Filter out any rows that duplicate existing Cartesian coordinates
        unique_rows = []
        for row in selected_rows:
            if not np.any(
                np.all(np.isclose(frac_xyz_total, row, atol=1e-2), axis=1)
            ):
                unique_rows.append(row)
                if len(unique_rows) >= rows_to_add:
                    break

        # Convert the unique fractional coordinates to Cartesian
        frac_xyz = np.array(unique_rows)
        cart_xyz = self.frac_to_cart(frac_xyz, self._supercell_lattice)

        # Prepare DataFrame for new rows with the element, its Uiso, and full occupancy
        new_rows = pd.DataFrame(
            {
                "x": cart_xyz[:, 0],
                "y": cart_xyz[:, 1],
                "z": cart_xyz[:, 2],
                "element": [chosen_element] * rows_to_add,
                "uiso": [UISO_VALUES[chosen_element]] * rows_to_add,
                "occupancy": [1] * rows_to_add,
            }
        )

        # Append new interstitial rows to the existing atoms_info DataFrame
        self._atoms_info = pd.concat(
            [self._atoms_info, new_rows], ignore_index=True
        )

        return chosen_element

    @staticmethod
    def apply_point_defect_multi(
        source_directory, target_directory, defect_type, ratio_range
    ):
        interstitial_sites_supercell = (
            SuperCell.get_interstitial_sites_supercell(dim=5)
        )
        for root, dirs, files in os.walk(source_directory):
            # Construct the target directory structure
            relative_path = os.path.relpath(root, source_directory)
            target_root = os.path.join(target_directory, relative_path)
            os.makedirs(target_root, exist_ok=True)
            sorted_files = sorted(files)

            for file in tqdm(sorted_files):
                if file.endswith(".json"):
                    source_file_path = os.path.join(root, file)
                    sc = SuperCell()
                    sc.load_json(source_file_path)
                    ratio = np.random.uniform(ratio_range[0], ratio_range[1])
                    chosen_element = None

                    if defect_type == "vacancy":
                        sc.apply_vacancy(ratio)

                    elif defect_type == "self_interstitial":
                        sc.apply_self_interstitial(
                            interstitial_sites_supercell[sc._space_group],
                            ratio,
                        )

                    elif defect_type == "substitutional_impurities":
                        chosen_element = sc.apply_substitutional_impurities(
                            ratio
                        )

                    elif defect_type == "interstitial_impurities":
                        chosen_element = sc.apply_interstitial_impurities(
                            interstitial_sites_supercell[sc._space_group],
                            ratio,
                        )

                    SuperCell._generate_and_save(
                        sc,
                        ratio,
                        defect_type,
                        chosen_element,
                        file,
                        target_root,
                    )

    @staticmethod
    def _generate_and_save(
        sc, ratio, defect_type, chosen_element, file, target_root
    ):
        """Generates target file name and saves JSON."""
        ratio_formatted = "{:.6f}".format(ratio * 100).replace(".", "p")

        if defect_type == "self_interstitial":
            defect_suffix = f"_selfint_{ratio_formatted}"
        else:
            defect_suffix = (
                f"_{defect_type[:3]}_{ratio_formatted}"
                if not chosen_element
                else f"_{defect_type[:3]}_{chosen_element}_{ratio_formatted}"
            )

        target_file_name = file.replace(".json", f"{defect_suffix}.json")
        target_file_path = os.path.join(target_root, target_file_name)
        sc.save_json(target_file_path)

    def _build_structure(self, modify_element=False):
        s = Structure()
        s.lattice = self._supercell_lattice
        atoms_info_frac = self.atoms_info.copy()
        atoms_info_frac[["x", "y", "z"]] = self.cart_to_frac(
            self._atoms_info[["x", "y", "z"]].to_numpy(),
            self._supercell_lattice,
        )
        if modify_element:
            atoms_info_frac["element"] = atoms_info_frac[
                "element"
            ].str.replace("[^a-zA-Z]", "", regex=True)

        for _, row in atoms_info_frac.iterrows():
            s.addNewAtom(
                atype=row["element"],
                xyz=row[["x", "y", "z"]],
                occupancy=row["occupancy"],
                Uisoequiv=row["uiso"],
            )
        return s

    @staticmethod
    def generate_pdf_from_json(
        json_directory, pdf_directory, pc, modify_element=False
    ):
        sc = SuperCell()
        sc.load_json(json_directory)
        s = sc._build_structure(modify_element)
        try:
            r, g = pc(s)
            file = open(pdf_directory, "a")
            rg = np.array([r, g])
            rg = rg.T
            np.savetxt(file, rg, fmt=["%f", "%f"])
            file.close()
        except Exception as ex:
            print(str(ex))
            print(
                "Exception: PDFCalculator not applied on {}.".format(
                    json_directory
                )
            )
            if modify_element is False:
                SuperCell.generate_pdf_from_json(
                    json_directory, pdf_directory, pc, modify_element=True
                )

    @staticmethod
    def generate_pdf_from_json_multi(
        source_directory,
        target_directory,
        construct_tasks=False,
        **pc_cfg,
    ):
        pc = PDFCalculator(**pc_cfg)
        skipped_files_count = 0  # Counter for skipped files

        if construct_tasks:
            tasks = []
        for root, dirs, files in os.walk(source_directory):
            # Construct the target directory structure
            relative_path = os.path.relpath(root, source_directory)
            target_root = os.path.join(target_directory, relative_path)
            os.makedirs(target_root, exist_ok=True)
            sorted_files = sorted(files)

            for file in tqdm(sorted_files):
                if file.endswith(".json"):
                    source_file_path = os.path.join(root, file)
                    target_file_name = file.replace(".json", ".gr")
                    target_file_path = os.path.join(
                        target_root, target_file_name
                    )

                    # Check if target file already exists
                    if os.path.exists(target_file_path):
                        skipped_files_count += (
                            1  # Increment the counter for skipped files
                        )
                        continue

                    if construct_tasks:
                        tasks.append((source_file_path, target_file_path, pc))
                    else:
                        sc = (
                            SuperCell()
                        )  # Assume SuperCell is defined elsewhere
                        sc.generate_pdf_from_json(
                            source_file_path, target_file_path, pc
                        )

        print(f"Total skipped files: {skipped_files_count}")
        # Print total number of skipped files
        if construct_tasks:
            return tasks
