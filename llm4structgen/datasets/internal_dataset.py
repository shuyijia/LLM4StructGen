import os
import glob
import torch
import random
import numpy as np
import pandas as pd
from pymatgen.core.structure import Structure

from dataclasses import dataclass
import transformers

from torch.utils.data import Dataset

from llm4structgen.constants import *
from llm4structgen.zmatrix import *

class InternalCoordinatesDataset(Dataset):
    def __init__(
        self,
        csv_fn,
        format_options={},
        llama_tokenizer=None,
        w_attributes=False,
        task_probabilities=None,
        add_perturbed_example=False,
        permutation_invariance=False,
        **kwargs
    ):
        super().__init__()

        if not os.path.exists(csv_fn) and not glob.glob(csv_fn):
            raise ValueError(f"CSV file {csv_fn} does not exist")

        df = pd.concat([pd.read_csv(fn) for fn in glob.glob(csv_fn)])
        self.inputs = df.to_dict(orient="records")
        self.llama_tokenizer = llama_tokenizer
        self.format_options = format_options
        self.w_attributes = w_attributes
        self.add_perturbed_example = add_perturbed_example
        self.permutation_invariance = permutation_invariance

        if task_probabilities is None:
            task_probabilities = {"generation": 2/3., "infill": 1/3.}
        self.task_probabilities = task_probabilities

    def get_zmatrix_string(self, input_dict, perturb=False):
        """
        under dev
        """
        atoms = zmatrix2struct(input_dict["zmatrix"], return_ase=True)

        cell = atoms.get_cell()
        chemical_symbols = atoms.get_chemical_symbols()
        positions = atoms.get_positions()

        if perturb:
            _min, _max = np.min(positions), np.max(positions)
            positions += np.random.normal(size=positions.shape) * 0.1
            positions = np.clip(positions, _min, _max)

        try:
            # copy positions
            _positions = positions.copy()
            zmatrix = struct2zmatrix(_positions, cell, chemical_symbols)
        except:
            _positions = positions.copy()
            # add very small random noise to positions
            _positions += np.random.normal(size=_positions.shape) * 1e-5
            zmatrix = struct2zmatrix(_positions, cell, chemical_symbols)

        return format_zmatrix_str(zmatrix)
    
    def get_permuted_zmatrix_string(self, input_dict):
        """
        permute the order of atoms
        """
        atoms = zmatrix2struct(input_dict["zmatrix"], return_ase=True)
        cell = atoms.get_cell()
        chemical_symbols = atoms.get_chemical_symbols()
        positions = atoms.get_positions()

        # permute the order of atoms
        # np.random.seed(2024)
        idx = np.random.permutation(len(chemical_symbols))
        chemical_symbols = [chemical_symbols[i] for i in idx]
        positions = positions[idx]

        # try:
        #     # copy positions
        #     _positions = positions.copy()
        #     zmatrix = struct2zmatrix(_positions, cell, chemical_symbols)
        # except:
        #     _positions = positions.copy()
        #     # add very small random noise to positions
        #     _positions += np.random.normal(size=_positions.shape) * 1e-6
        #     zmatrix = struct2zmatrix(_positions, cell, chemical_symbols)

        # always add very small random noise to positions
        _positions = positions.copy()
        _positions += np.random.normal(size=_positions.shape) * 1e-6
        zmatrix = struct2zmatrix(_positions, cell, chemical_symbols)
        
        return format_zmatrix_str(zmatrix)

    def generation_with_perturbation_task(self, input_dict):
        prompt = (
            "Below is a description of a bulk material containing "
            "the lengths and angles of the lattice vectors and the element "
            "type and its distance, bond angle and dihedral angle for each atom within the lattice. "
            "However, gaussian noises have been added to each of the distance, bond angle and dihedral angle:"
        )

        zmatrix_str = input_dict["zmatrix"]
        perturbed_zmatrix_str = perturb_zmatrix_str(zmatrix_str)
        
        end_prompt = (
            "Use the above noisy version of the bulk material to complete the following task:\n"
        )

        prompt = prompt + "\n" + perturbed_zmatrix_str + "\n" + end_prompt
        prompt = prompt + self.generation_prompt(input_dict) + zmatrix_str + self.llama_tokenizer.eos_token

        tokens = self.llama_tokenizer(
            prompt,
            return_tensors="pt",
            max_length=MAX_LENGTH,
            truncation=True,
        )

        return tokens
    
    def generation_prompt(self, input_dict):
        prompt = (
            "Below is a description of a bulk material where each atom is "
            "described by its element type and three attributes: "
            "1. distance to the previous atom, "
            "2. angle to the previous two atoms, " 
            "3. dihedral angle to the previous three atoms. "
            "The first three Fm atoms are dummies that help define the rest of the material. "
        )
        
        all_attributes = [
            "formation_energy_per_atom",
            "band_gap",
            "e_above_hull",
            "spacegroup.number",
        ]

        # sample a random collection of attributes
        num_attributes = random.randint(0, len(all_attributes))
        if num_attributes > 0 and self.w_attributes:
            attributes = random.sample(all_attributes, num_attributes)
            attributes = ["pretty_formula"] + attributes

            prompt_lookup = {
                "formation_energy_per_atom": "The formation energy per atom is",
                "band_gap": "The band gap is",
                "pretty_formula": "The chemical formula is",
                "e_above_hull": "The energy above the convex hull is",
                "elements": "The elements are",
                "spacegroup.number": "The spacegroup number is",
            }

            for attr in attributes:
                if attr == "elements":
                    prompt += f"{prompt_lookup[attr]} {', '.join(input_dict[attr])}. "
                elif attr in ["formation_energy_per_atom", "band_gap", "e_above_hull"]:
                    prompt += f"{prompt_lookup[attr]} {round(float(input_dict[attr]), 4)}. "
                else:
                    prompt += f"{prompt_lookup[attr]} {input_dict[attr]}. "

        prompt += (
            "Generate a description of the lengths and angles of the lattice vectors "
            "and the three dummy Fm atoms, followed by "
            "the element type and the three attributes for each atom within the lattice:\n"
        )

        return prompt

    def generation_task(self, input_dict):
        prompt = (
            "Below is a description of a bulk material where each atom is "
            "described by its element type and three attributes: "
            "1. distance to the previous atom, "
            "2. angle to the previous two atoms, " 
            "3. dihedral angle to the previous three atoms. "
            "The first three Fm atoms are dummies that help define the rest of the material. "
        )
        
        all_attributes = [
            "formation_energy_per_atom",
            "band_gap",
            "e_above_hull",
            "spacegroup.number",
        ]

        # sample a random collection of attributes
        num_attributes = random.randint(0, len(all_attributes))
        if num_attributes > 0 and self.w_attributes:
            attributes = random.sample(all_attributes, num_attributes)
            attributes = ["pretty_formula"] + attributes

            prompt_lookup = {
                "formation_energy_per_atom": "The formation energy per atom is",
                "band_gap": "The band gap is",
                "pretty_formula": "The chemical formula is",
                "e_above_hull": "The energy above the convex hull is",
                "elements": "The elements are",
                "spacegroup.number": "The spacegroup number is",
            }

            for attr in attributes:
                if attr == "elements":
                    prompt += f"{prompt_lookup[attr]} {', '.join(input_dict[attr])}. "
                elif attr in ["formation_energy_per_atom", "band_gap", "e_above_hull"]:
                    prompt += f"{prompt_lookup[attr]} {round(float(input_dict[attr]), 4)}. "
                else:
                    prompt += f"{prompt_lookup[attr]} {input_dict[attr]}. "

        prompt += (
            "Generate a description of the lengths and angles of the lattice vectors "
            "and the three dummy Fm atoms, followed by "
            "the element type and the three attributes for each atom within the lattice:\n"
        )

        if self.permutation_invariance:
            crystal_str = self.get_permuted_zmatrix_string(input_dict)
        else:
            crystal_str = input_dict["zmatrix"]

        tokens = self.llama_tokenizer(
            prompt + crystal_str  + self.llama_tokenizer.eos_token,
            return_tensors="pt",
            max_length=MAX_LENGTH,
            truncation=True,
        )

        return tokens

    def infill_task(self, input_dict):
        prompt = (
            'Below is a partial description of a bulk material where '
            'Fm atoms are dummies that help define the rest of the material and '
            'one element has been replaced with the string "[MASK]":\n'
        )

        k = 'cif' if 'cif' in input_dict else 'cif_str'
        structure = Structure.from_str(input_dict[k], fmt="cif")
        species = [str(s) for s in structure.species]
        species_to_remove = random.choice(species)

        if self.permutation_invariance:
            crystal_string = self.get_permuted_zmatrix_string(input_dict)
        else:
            crystal_string = input_dict["zmatrix"]

        partial_crystal_str = crystal_string.replace(
            species_to_remove, "[MASK]"
        )

        infill_str = prompt + partial_crystal_str + "\n"

        infill_str += (
            "Generate an element that could replace [MASK] in the bulk material:\n"
        )

        infill_str += str(species_to_remove) + self.llama_tokenizer.eos_token

        tokens = self.llama_tokenizer(
            infill_str,
            return_tensors="pt",
            max_length=MAX_LENGTH,
            truncation=True,
        )

        return tokens

    def tokenize(self, input_dict):
        random_val = random.random()
        if "perturbation" in self.task_probabilities:
            if random_val < self.task_probabilities["perturbation"]:
                tokens = self.generation_with_perturbation_task(input_dict)
            elif random_val < self.task_probabilities["perturbation"] + self.task_probabilities["generation"]:
                tokens = self.generation_task(input_dict)
            else:
                tokens = self.infill_task(input_dict)
        else:
            if random_val < self.task_probabilities["generation"]:
                tokens = self.generation_task(input_dict)
            else:
                tokens = self.infill_task(input_dict)

        input_ids = labels = tokens.input_ids[0]
        input_ids_lens = labels_lens = tokens.input_ids.ne(
            self.llama_tokenizer.pad_token_id).sum().item()
        return dict(
            input_ids=input_ids,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens,
        )

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError("Index out of range")

        vals = self.inputs[index]
        vals = self.tokenize(vals)
        return vals