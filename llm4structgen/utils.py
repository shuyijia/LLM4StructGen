from pathlib import Path
from dataclasses import dataclass
from llm4structgen.constants import *

import torch

from transformers import ( 
    LlamaForCausalLM,
    LlamaTokenizer, 
    BitsAndBytesConfig
)

from peft import LoraConfig, get_peft_model

def smart_tokenizer_and_embedding_resize(
    model,
    llama_tokenizer
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    special_tokens_dict = dict()
    if llama_tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if llama_tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if llama_tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if llama_tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    num_new_tokens = llama_tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(llama_tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def get_model_name(model_name):
    llama_options = model_name.split("-")
    is_chat = len(llama_options) == 2
    model_size = llama_options[0]

    def llama2_model_string(model_size, chat):
        chat = "chat-" if chat else ""
        return f"meta-llama/Llama-2-{model_size.lower()}-{chat}hf"

    return llama2_model_string(model_size, is_chat)

def get_model(args, rank):
    # for SFT Trainer
    model_string = get_model_name(args.model_name)

    compute_dtype = getattr(torch, "float16")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )

    model = LlamaForCausalLM.from_pretrained(
        model_string,
        quantization_config=quant_config,
        device_map={"": rank}
    )

    return model

def get_crystal_llm_model(args, rank):
    model_string = get_model_name(args.model_name)

    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = LlamaForCausalLM.from_pretrained(
        model_string,
        quantization_config=quantization_config,
        device_map={"": rank},
        torch_dtype=torch.bfloat16,

    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model

def get_tokenizer(args):
    model_string = get_model_name(args.model_name)

    llama_tokenizer = LlamaTokenizer.from_pretrained(
        model_string,
        model_max_length=MAX_LENGTH,
        padding_side="right",
        use_fast=False,
    )

    return llama_tokenizer

@dataclass
class ModelConfig:
    run_name: str
    expdir: Path = Path("exp")
    model_name: str = "13b"
    fp8: bool = True
    lora_rank: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    dataset_type: str = "cif"
    data_path: Path = Path("data/mp20-cif/")
    num_epochs: int = 5
    batch_size: int = 2
    gradient_accumulation_steps: int = 1
    lr: float = 5e-4
    lr_scheduler: str = "cosine"
    warmup_ratio: int = 0.03
    num_warmup_steps: int = 100
    weight_decay: float = 0.0
    eval_freq: int = 1000
    save_freq: int = 500
    log_freq: int = 1
    format_permute_composition: bool = False
    format_permute_structure: bool = False
    w_attributes: bool = True
    resume_dir: Path = None
    task_probabilities: dict = None
