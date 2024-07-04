import os
import torch
import argparse
from transformers import TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig
from pathlib import Path

from llm4structgen.datasets.base_dataset import BaseDataset
from llm4structgen.utils import parse_attributes
from llm4structgen.datasets.prompts import *
from llm4structgen.llms.llama2_utils import *
from llm4structgen.representations.z_matrix import ZMatrix
from llm4structgen.datasets.collators import DataCollatorForSupervisedDataset

os.environ["WANDB_PROJECT"] = "llm4structgen"

def main(args):
    # create output directory
    output_dir = Path(args.expdir) / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ["ACCELERATE_MIXED_PRECISION"] = "no"

    training_args = TrainingArguments(
        run_name=args.run_name,
        fsdp=False,
        fp16=not args.fp8,
        bf16=False,
        gradient_checkpointing=False,
        ddp_find_unused_parameters=False,
        num_train_epochs=args.num_epochs,
        eval_steps=args.eval_freq,
        save_steps=args.save_freq,
        logging_steps=10,
        evaluation_strategy="steps",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler,
        warmup_steps=args.num_warmup_steps,
        # warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        output_dir=output_dir,
        report_to="wandb",
        dataloader_num_workers=8,
        remove_unused_columns=False,
        label_names=["crystal_ids"],
    )

    # get model and tokenizer
    model = get_model(args, training_args.local_rank)
    tokenizer = get_tokenizer(args)
    smart_tokenizer_and_embedding_resize(model, tokenizer)
    
    # get dataset
    encoder = ZMatrix(args.translate, args.rotate, args.permute)

    train_dataset = BaseDataset(
        data_dir=args.train_data,
        tokenizer=tokenizer,
        encoder=encoder,
        prompt_header=Z_MATRIX_GENERATION_PROMPT_HEADER,
        duplicate_count=args.duplicate_count,
        attributes=args.attributes,
    )

    val_dataset = BaseDataset(
        data_dir=args.val_data,
        tokenizer=tokenizer,
        encoder=encoder,
        prompt_header=Z_MATRIX_GENERATION_PROMPT_HEADER,
        duplicate_count=args.duplicate_count,
        attributes=args.attributes,
    )

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    trainer = SFTTrainer(
        model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        max_seq_length=MAX_LENGTH,
        peft_config=lora_config,
        data_collator=data_collator,
        packing=True
    )

    train_result = trainer.train()
    trainer.save_state()
    trainer.save_model()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="run")
    parser.add_argument("--expdir", type=str, default="exp")
    parser.add_argument("--train_data", type=str, default="data/mp-20/train.json")
    parser.add_argument("--val_data", type=str, default="data/mp-20/val.json")
    parser.add_argument("--model_name", type=str, default="7b")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--eval_freq", type=int, default=500)
    parser.add_argument("--save_freq", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--num_warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--fp8", action="store_true")
    parser.add_argument("--translate", action="store_false")
    parser.add_argument("--rotate", action="store_false")
    parser.add_argument("--permute", action="store_false")
    parser.add_argument("--duplicate_count", type=int, default=1)
    parser.add_argument("--attributes", type=parse_attributes, default=None)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    args = parser.parse_args()

    main(args)