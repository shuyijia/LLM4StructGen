import os
import torch
from transformers import Trainer, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig

from llm4structgen.utils import *
from llm4structgen.datasets import get_datasets, DataCollatorForSupervisedDataset

args = ModelConfig(
    run_name="sft-crystal",
    model_name="7b",
    batch_size=8,
)

output_dir= args.expdir / args.run_name
output_dir.mkdir(parents=True, exist_ok=True)

os.environ["ACCELERATE_MIXED_PRECISION"] = "no"
training_args = TrainingArguments(
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
    run_name=args.run_name,
    report_to="wandb",
    dataloader_num_workers=8,
    remove_unused_columns=False,
    label_names=["crystal_ids"], #this is just to get trainer to behave how I want
)

model = get_crystal_llm_model(args, training_args.local_rank)
tokenizer = get_tokenizer(args)
smart_tokenizer_and_embedding_resize(model, tokenizer)
datasets = get_datasets(args, tokenizer)
data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)


trainer = Trainer(
    model,
    args=training_args,
    train_dataset=datasets["train"],
    eval_dataset=datasets["val"],
    data_collator=data_collator,
)

train_result = trainer.train()
trainer.save_state()
trainer.save_model()
