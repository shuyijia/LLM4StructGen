import os
import glob
import torch
import random
import numpy as np
import pandas as pd
from pymatgen.core.structure import Structure



from torch.utils.data import Dataset

from llm4structgen.constants import *

def get_crystal_string(cif_str, perturb=False):
    structure = Structure.from_str(cif_str, fmt="cif")

    # Randomly translate within the unit cell
    structure.translate_sites(
        indices=range(len(structure.sites)), vector=np.random.uniform(size=(3,))
    )

    lengths = structure.lattice.parameters[:3]
    angles = structure.lattice.parameters[3:]
    atom_ids = structure.species
    frac_coords = structure.frac_coords

    crystal_str = \
        " ".join(["{0:.1f}".format(x) for x in lengths]) + "\n" + \
        " ".join([str(int(x)) for x in angles]) + "\n" + \
        "\n".join([
            str(t) + "\n" + " ".join([
                "{0:.2f}".format(x) for x in c
            ]) for t,c in zip(atom_ids, frac_coords)
        ])

    if perturb:
        frac_coords = np.array(frac_coords)
        frac_coords += np.random.normal(size=frac_coords.shape) * 0.1
        frac_coords = np.clip(frac_coords, 0, 1)

        perturbed_crystal_str = \
            " ".join(["{0:.1f}".format(x) for x in lengths]) + "\n" + \
            " ".join([str(int(x)) for x in angles]) + "\n" + \
            "\n".join([
                str(t) + "\n" + " ".join([
                    "{0:.2f}".format(x) for x in c
                ]) for t,c in zip(atom_ids, frac_coords)
            ])
        
        return crystal_str, perturbed_crystal_str

    return crystal_str

class CartesianDataset(Dataset):
    def __init__(
        self,
        csv_fn,
        format_options={},
        llama_tokenizer=None,
        w_attributes=False,
        task_probabilities=None,
        add_perturbed_example=False,
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
            task_probabilities = {"generation": 2/3., "infill": 1/3.}
        self.task_probabilities = task_probabilities
   
    def crystal_string(self, input_dict, perturb=False):
        k = 'cif' if 'cif' in input_dict else 'cif_str'
        return get_crystal_string(input_dict[k], perturb=perturb)
    
    def generation_with_perturbation_task(self, input_dict):
        prompt = (
            "Below is a description of a bulk material containing "
            "the lengths and angles of the lattice vectors and the element "
            "type and coordinates for each atom within the lattice. "
            "However, gaussian noises have been added to each coordinate:"
        )

        crystal_str, perturbed_crystal_str = self.crystal_string(input_dict, perturb=True)
        end_prompt = (
            "Use the above noisy version of the bulk material to complete the following task:\n"
        )

        prompt = prompt + "\n" + perturbed_crystal_str + "\n" + end_prompt
        prompt = prompt + self.generation_prompt(input_dict) + crystal_str + self.llama_tokenizer.eos_token

        tokens = self.llama_tokenizer(
            prompt,
            return_tensors="pt",
            max_length=MAX_LENGTH,
            truncation=True,
        )

        return tokens

    def generation_prompt(self, input_dict):
        prompt = "Below is a description of a bulk material. "
        
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
            "and then the element type and coordinates for each atom within the lattice:\n"
        )

        return prompt

    def generation_task(self, input_dict):
        prompt = "Below is a description of a bulk material. "
        
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
            "and then the element type and coordinates for each atom within the lattice:\n"
        )

        crystal_str = self.crystal_string(input_dict)

        tokens = self.llama_tokenizer(
            prompt + crystal_str  + self.llama_tokenizer.eos_token,
            return_tensors="pt",
            max_length=MAX_LENGTH,
            truncation=True,
        )

        return tokens

    def infill_task(self, input_dict):
        prompt = (
            'Below is a partial description of a bulk material where one '
            'element has been replaced with the string "[MASK]":\n'
        )

        k = 'cif' if 'cif' in input_dict else 'cif_str'
        structure = Structure.from_str(input_dict[k], fmt="cif")
        species = [str(s) for s in structure.species]
        species_to_remove = random.choice(species)

        crystal_string = self.crystal_string(input_dict)

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
