import itertools
import sys
import time
import json
import copy
import numpy as np
from tqdm import tqdm
from ase.data import chemical_symbols
from datetime import datetime
from omegaconf import OmegaConf
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn

from torchtune import config, training, utils
from torchtune.config._utils import _get_component_from_path
from torchtune.data import ChatFormat, InstructTemplate, Message

from llm4structgen.generation.unconditional_generation_prompts import *

def setup(cfg):
    _device = utils.get_device(device=cfg.device)
    _dtype = training.get_dtype(dtype=cfg.dtype, device=_device)
    checkpointer = config.instantiate(cfg.checkpointer)
    ckpt_dict = checkpointer.load_checkpoint()

    # set up model
    with training.set_default_dtype(_dtype), _device:
        model = config.instantiate(cfg.model)
    model.load_state_dict(ckpt_dict[training.MODEL_KEY])
    training.validate_expected_param_dtype(
        model.named_parameters(), dtype=_dtype
    )

    # set up tokenizer
    tokenizer = config.instantiate(cfg.tokenizer)

    return model, tokenizer

def calculate_perplexity(model, input_seq):
    assert input_seq.ndim == 2, f"Input sequence should be of shape [1, seq_length], but got {input_seq.shape}"

    with torch.no_grad():
        # Get model logits [batch, seq_length, vocab_size]
        logits = model(input_seq, input_pos=torch.arange(input_seq.size(1), device=input_seq.device))
        
        # Shift inputs and outputs for teacher forcing
        logits = logits[:, :-1, :]  # Remove last logit (since there's no next token)
        targets = input_seq[:, 1:].long()  # Remove first token (since it's conditioned on previous tokens)

        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=-1)  # Convert logits to log probabilities
        token_log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)  # Select correct token probabilities

        # Compute negative log likelihood loss
        nll = -token_log_probs.mean(dim=-1)  # Average over sequence length

    # Compute perplexity
    ppl = torch.exp(nll).item()
    return ppl

def get_all_prompts(tokenizer, encoder, prompt_header, _device):
    # get test set data
    f = open("data/mp-20/test.json", "r")
    test_data = json.load(f)
    test_data = {x['material_id'] : x for x in test_data}
    f.close()

    # get permutations
    f = open("data/permute/selected_500.json", "r")
    perms = json.load(f)
    f.close()

    all_prompts = {}

    for k, v in tqdm(perms.items(), desc="Generating prompts & tokens."):
        all_prompts[k] = []

        s = test_data[k]
        symbols = np.array([chemical_symbols[x] for x in s['atomic_numbers']])
        positions = np.array(s['positions'])
        cell = s['cell']

        for p in v:
            _p = np.array(p)
            _syms = symbols[p]
            _pos = positions[p]
            rep = encoder.encode(_syms, _pos, cell)

            _prompt = prompt_header + rep
            tokens = tokenizer.encode(_prompt, add_bos=True, add_eos=False)
            prompt = torch.tensor(tokens, dtype=torch.int, device=_device).view(1,-1)
            all_prompts[k].append(prompt)

    return all_prompts

def run(cfg):
    _device = utils.get_device(device=cfg.device)
    _dtype = training.get_dtype(dtype=cfg.dtype, device=_device)
    model, tokenizer = setup(cfg)

    # set up prompt header and encoder
    if cfg.representation_type == "cartesian":
        from llm4structgen.representations.cartesian import Cartesian
        _prompt = UNCONDITIONAL_CARTESIAN_GENERATION_PROMPT_HEADER
        encoder = Cartesian()
    elif cfg.representation_type == "distance":
        from llm4structgen.representations.distance_matrix import DistanceMatrix
        _prompt = UNCONDITIONAL_DISTANCE_MATRIX_GENERATION_PROMPT_HEADER
        encoder = DistanceMatrix()
    elif cfg.representation_type == "slices":
        from llm4structgen.representations.slices import SLICES
        _prompt = UNCONDITIONAL_SLICES_GENERATION_PROMPT_HEADER
        encoder = SLICES()
    elif cfg.representation_type == "zmatrix":
        from llm4structgen.representations.z_matrix import ZMatrix
        _prompt = UNCONDITIONAL_Z_MATRIX_GENERATION_PROMPT_HEADER
        encoder = ZMatrix()
    else:
        raise ValueError(f"Invalid representation type: {cfg.representation_type}")

    tokens = tokenizer.encode(_prompt, add_bos=True, add_eos=False)
    prompt = torch.tensor(tokens, dtype=torch.int, device=_device)
    prompt = prompt.view(1, -1) if prompt.ndim == 1 else prompt

    # generated_tokens = utils.generate(
    #     model=model,
    #     prompt=prompt,
    #     max_generated_tokens=cfg.max_new_tokens,
    #     temperature=cfg.temperature,
    #     top_k=cfg.top_k,
    #     stop_tokens=tokenizer.stop_tokens,
    #     custom_generate_next_token=None
    # )
    # output_str = tokenizer.decode(generated_tokens[0])
    # print(output_str)

    all_prompts = get_all_prompts(tokenizer, encoder, _prompt, _device)
    all_ppls = []
    for k, prompts in tqdm(all_prompts.items(), desc="Calculating PPLs"):
        curr_ppl = []
        for each in prompts:
            curr_ppl.append(calculate_perplexity(model, each))
        all_ppls.append(curr_ppl)
    
    arr = np.array(all_ppls)

    # calculate IPT
    row_min = arr.min(axis=1, keepdims=True)
    row_normalized = arr - row_min
    row_means = row_normalized.mean(axis=1)
    ipt = row_means.mean()
    print(ipt)

@config.parse
def main(cfg: DictConfig) -> None:
    config.log_config(recipe_name="InferenceRecipe", cfg=cfg)
    run(cfg)

if __name__ == "__main__":
    sys.exit(main())