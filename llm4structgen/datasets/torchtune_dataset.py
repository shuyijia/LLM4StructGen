import os
import json
import copy
import glob
import torch
import random
import numpy as np

from typing import Any, Dict, List, Mapping, Optional

import torch
from datasets import load_dataset
from torch.utils.data import Dataset

from torchtune.data import truncate
from torchtune.datasets._packed import PackedDataset
from torchtune.modules.tokenizers import ModelTokenizer

prompt_lookup = {
    "formation_energy_per_atom": "The formation energy per atom is",
    "band_gap": "The band gap is",
    "e_above_hull": "The energy above the convex hull is",
    "spacegroup_number": "The spacegroup number is",
}

class TextCompletionDataset(Dataset):
    """
    modified from github.com/pytorch/torchtune/blob/main/torchtune/datasets/_text_completion.py

    Freeform dataset for any unstructured text corpus. Quickly load any dataset
    from Hugging Face or local disk and tokenize it for your model.

    Args:
        tokenizer (ModelTokenizer): Tokenizer used by the model that implements the ``tokenize_messages`` method.
        source (str): path string of dataset, anything supported by Hugging Face's ``load_dataset``
            (https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path)
        column (str): name of column in the sample that contains the text data. This is typically required
            for Hugging Face datasets or tabular data. For local datasets with a single column, use the default "text",
            which is what is assigned by Hugging Face datasets when loaded into memory. Default is "text".
        max_seq_len (Optional[int]): Maximum number of tokens in the returned input and label token id lists.
            Default is None, disabling truncation. We recommend setting this to the highest you can fit in memory
            and is supported by the model. For example, llama2-7B supports up to 4096 for sequence length.
        add_eos (bool): Whether to add an EOS token to the end of the sequence. Default is True.
        **load_dataset_kwargs (Dict[str, Any]): additional keyword arguments to pass to ``load_dataset``.
    """

    def __init__(
        self,
        tokenizer: ModelTokenizer,
        representation_type: str,
        source: str,
        data_files: str,
        max_seq_len: Optional[int] = None,
        add_eos: bool = True,
        attributes: Any = False,
        translate: bool = False,
        rotate: bool = False,
        permute: bool = False,
        decimals: int = 2,
        duplicate_count: int = 1,
    ) -> None:
        self._tokenizer = tokenizer
        self.representation_type = representation_type
        self.max_seq_len = max_seq_len
        self.add_eos = add_eos
        self.attributes = attributes
        self.duplicate_count = duplicate_count
        
        # self._data = load_dataset(source, data_files=data_files)
        self._data = self.load_data(source, data_files)

        # initialize encoder
        if self.representation_type == "cartesian":
            from llm4structgen.representations.cartesian import Cartesian
            self.encoder = Cartesian(
                translate=translate,
                rotate=rotate,
                permute=permute,
                decimals=decimals
            )
        elif self.representation_type == "zmatrix":
            from llm4structgen.representations.z_matrix import ZMatrix
            self.encoder = ZMatrix(
                translate=translate,
                rotate=rotate,
                permute=permute,
                decimals=decimals
            )
        elif self.representation_type == "distance":
            from llm4structgen.representations.distance_matrix import DistanceMatrix
            self.encoder = DistanceMatrix(
                translate=translate,
                rotate=rotate,
                permute=permute,
                decimals=decimals
            )
        elif self.representation_type == "slices":
            from llm4structgen.representations.slices import SLICES
            self.encoder = SLICES(
                translate=translate,
                rotate=rotate,
                permute=permute
            )
        else:
            raise ValueError("Invalid representation type; must be one of 'cartesian', 'zmatrix', 'distance', or 'slices'")
        
    def load_data(self, source, data_files):
        if source != "json":
            raise NotImplementedError

        f = open(data_files, "r")
        data_list = json.load(f)
        f.close()

        if self.duplicate_count == 1:
            return data_list
        
        return [copy.deepcopy(d) for d in data_list for _ in range(self.duplicate_count)]

    def __len__(self):
        return len(self._data)

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
        prompt_header = self.encoder.prompt_header.replace("<attributes>", attributes).strip()

        return prompt_header
    
    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        sample = self._data[index]
        return self._prepare_sample(sample)
    
    def _prepare_sample(self, sample: Mapping[str, Any]) -> Dict[str, List[int]]:
        prompt_header = self.get_prompt_header(sample)
        representation = self.encoder.encode(
            sample['atomic_symbols'],
            sample['positions'],
            sample['cell']
        )

        full_str = prompt_header + "\n" + representation
        tokens = self._tokenizer.encode(
            text=full_str, 
            add_bos=True, 
            add_eos=self.add_eos
        )

        # Truncate if needed, but don't coerce EOS id
        if self.max_seq_len is not None:
            tokens = truncate(tokens, self.max_seq_len - 1)

        # No need to offset labels by 1 - happens in the recipe
        labels = tokens.copy()

        return {"tokens": tokens, "labels": labels}

def text_completion_dataset(
    tokenizer: ModelTokenizer,
    representation_type: str,
    source: str,
    data_files: str,
    max_seq_len: Optional[int] = None,
    add_eos: bool = True,
    packed: bool = False,
    attributes: Any = False,
    translate: bool = False,
    rotate: bool = False,
    permute: bool = False,
    decimals: int = 2,
    duplicate_count: int = 1,
) -> TextCompletionDataset:
    """
    Build a configurable dataset from a freeform, unstructured text corpus similar
    to datasets used in pre-training. This method should be
    used to configure a custom text dataset from the yaml config instead of
    using :class:`~torchtune.datasets.TextCompletionDataset` directly, as it is made to be config friendly.
    """
    ds = TextCompletionDataset(
        tokenizer=tokenizer,
        representation_type=representation_type,
        source=source,
        data_files=data_files,
        max_seq_len=max_seq_len,
        add_eos=add_eos,
        attributes=attributes,
        translate=translate,
        rotate=rotate,
        permute=permute,
        decimals=decimals,
        duplicate_count=duplicate_count
    )

    return (
        PackedDataset(ds, max_seq_len=max_seq_len, padding_idx=tokenizer.pad_id)
        if packed
        else ds
    )

