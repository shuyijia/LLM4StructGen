import os
import glob
import torch
import random
import numpy as np
import pandas as pd
from pymatgen.core.structure import Structure

from torch.utils.data import Dataset

from llm4structgen.constants import *


class DistanceMatrixDataset(Dataset):
    def __init__(
        self,
        csv_fn,
        format_options={},
        llama_tokenizer=None,
        w_attributes=False,
        task_probabilities=None,
        add_perturbed_example=False,
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

        if task_probabilities is None:
            task_probabilities = {"generation": 2 / 3.0, "infill": 1 / 3.0}
        self.task_probabilities = task_probabilities

    def get_distance_matrix_string(self, input_dict, permute=False):
        distance_matrix = input_dict["distance_matrix"]
        return distance_matrix

    def generation_task(self, input_dict):
        prompt = "Below is a description of a bulk material."

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
                    prompt += (
                        f"{prompt_lookup[attr]} {round(float(input_dict[attr]), 4)}. "
                    )
                else:
                    prompt += f"{prompt_lookup[attr]} {input_dict[attr]}. "

        prompt += (
            "Generate a description of the lengths and angles of the lattice vectors "
            "followed by the element type and distances for each atom within the lattice "
            "ensuring that each atom solely references distances to preceding atoms, "
            "resembling the lower triangular portion of a distance matrix:\n"
        )

        crystal_str = input_dict["distance_matrix"]

        tokens = self.llama_tokenizer(
            prompt + crystal_str + self.llama_tokenizer.eos_token,
            return_tensors="pt",
            max_length=MAX_LENGTH,
            truncation=True,
        )

        return tokens

    def infill_task(self, input_dict):
        prompt = (
            "Below is a partial description of a bulk material where one "
            'element has been replaced with the string "[MASK]":\n'
        )

        k = "cif" if "cif" in input_dict else "cif_str"
        structure = Structure.from_str(input_dict[k], fmt="cif")
        species = [str(s) for s in structure.species]
        species_to_remove = random.choice(species)

        crystal_string = input_dict["distance_matrix"]

        partial_crystal_string = crystal_string.replace(species_to_remove, "[MASK]")

        infill_str = prompt + partial_crystal_string + "\n"

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
        if random_val < self.task_probabilities["generation"]:
            tokens = self.generation_task(input_dict)
        else:
            tokens = self.infill_task(input_dict)

        input_ids = labels = tokens.input_ids[0]
        input_ids_lens = labels_lens = (
            tokens.input_ids.ne(self.llama_tokenizer.pad_token_id).sum().item()
        )
        return dict(
            input_ids=input_ids,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens,
        )

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        if not 0 <= index < len(self.inputs):
            raise IndexError(f"Index {index} out of range")

        vals = self.inputs[index]
        vals = self.tokenize(vals)
        return vals
