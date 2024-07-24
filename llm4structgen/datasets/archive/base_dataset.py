import os
import json
import copy
import glob
import torch
import random
import numpy as np

from typing import Any

import torch
from torch.utils.data import Dataset

from llm4structgen.constants import *

class BaseDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        tokenizer: Any,
        encoder: Any,
        prompt_header: str,
        duplicate_count: int = 1,
        attributes: Any = False
    ) -> None:
        """
        Args:
            data_dir (str): path to the data directory
            tokenizer (Tokenizer): tokenizer that breaks representations into tokens
            encoder (Encoder): representation encoder and decoder
            duplicate_count (int): number of times to duplicate each data point
            prompt_header (str): header for the prompt
        """
        super().__init__()
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.prompt_header = prompt_header
        self.duplicate_count = duplicate_count
        self.attributes = attributes

        self.data_list = self.load_data()

    def load_data(self):
        f = open(self.data_dir, "r")
        data_list = json.load(f)
        f.close()

        if self.duplicate_count == 1:
            return data_list
        
        return [copy.deepcopy(d) for d in data_list for _ in range(self.duplicate_count)]
    
    def get_attributes(self, data):
        if isinstance(self.attributes, list):
            # get user-specified attributes
            return self._add_specific_attributes(data)
        elif self.attributes == "random" or self.attributes is True:
            # get random attributes
            return self._add_random_attributes(data)
        elif self.attributes is None or self.attributes is False:
            # no attributes
            return ""
        else:
            raise ValueError("Invalid attributes configuration")
    
    def _add_specific_attributes(self, data: dict):
        # always include the chemical formula
        out_str = f"The chemical formula is {data['pretty_formula']}. "

        for attr in self.attributes:
            if attr in ["formation_energy_per_atom", "e_above_hull", "band_gap"]:
                out_str += f"{prompt_lookup[attr]} {data[attr]:.2f}. "
            else:
                out_str += f"{prompt_lookup[attr]} {data[attr]}. "
        
        return out_str
        
    def _add_random_attributes(self, data: dict):
        num_attributes = random.randint(0, len(prompt_lookup))
        attributes = random.sample(prompt_lookup.keys(), num_attributes)

        # always include the chemical formula
        out_str = f"The chemical formula is {data['pretty_formula']}. "

        for attr in attributes:
            if attr in ["formation_energy_per_atom", "e_above_hull", "band_gap"]:
                out_str += f"{prompt_lookup[attr]} {data[attr]:.2f}. "
            else:
                out_str += f"{prompt_lookup[attr]} {data[attr]}. "
        
        return out_str.strip()
    
    def get_prompt_header(self, data):
        attributes = self.get_attributes(data)

        # add attributes to the prompt header
        prompt_header = self.prompt_header.replace("<attributes>", attributes).strip()

        return prompt_header
    
    def tokenize(self, text):
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=MAX_LENGTH,
            truncation=True
        )

        input_ids = labels = tokens.input_ids[0]
        input_ids_lens = labels_lens = tokens.input_ids.ne(
            self.tokenizer.pad_token_id
        ).sum().item()
        
        return dict(
            input_ids=input_ids,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens
        )
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data = self.data_list[idx]
        prompt_header = self.get_prompt_header(data)
        representation = self.encoder.encode(
            data['atomic_symbols'],
            data['positions'],
            data['cell']
        )

        full_str = prompt_header + "\n" + representation + self.tokenizer.eos_token
        return self.tokenize(full_str)


prompt_lookup = {
    "formation_energy_per_atom": "The formation energy per atom is",
    "band_gap": "The band gap is",
    "e_above_hull": "The energy above the convex hull is",
    "spacegroup_number": "The spacegroup number is",
}
