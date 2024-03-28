from pathlib import Path
from dataclasses import dataclass
from cif_dataset import CifDataset
from constants import *

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

def setup_datasets(args, llama_tokenizer, transform_args={}):    
    format_options = {
        "permute_composition": args.format_permute_composition,
        "permute_structure": args.format_permute_structure,
    }

    datasets = {
        "train": CifDataset(
            str(args.data_path / "train.csv"), 
            format_options,
            llama_tokenizer=llama_tokenizer,
            w_attributes=args.w_attributes,
        ),
        "val": CifDataset(
            str(args.data_path / "val.csv"),
            format_options,
            llama_tokenizer=llama_tokenizer,
            w_attributes=args.w_attributes,
        ),
    }

    return datasets

@dataclass
class ModelConfig:
    run_name: str
    expdir: Path = Path("exp")
    model_name: str = "13b"
    bf16: bool = True
    fp16: bool = False
    fp8: bool = False
    lora_rank: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    data_path: Path = Path("data/full_no_val_94209")
    num_epochs: int = 44
    batch_size: int = 2
    gradient_accumulation_steps: int = 1
    lr: float = 5e-4
    lr_scheduler: str = "cosine"
    warmup_ratio: int = 0.03
    weight_decay: float = 0.0
    eval_freq: int = 1000
    save_freq: int = 500
    log_freq: int = 1
    format_permute_composition: bool = False
    format_permute_structure: bool = False
    w_attributes: int = 1
    resume_dir: Path = None