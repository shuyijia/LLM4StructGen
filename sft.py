import os
import torch
from transformers import TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig

from llm4structgen.utils import *
from llm4structgen.datasets import get_datasets, DataCollatorForSupervisedDataset

os.environ["WANDB_PROJECT"] = "internal-coordinates"

args = ModelConfig(
    run_name="sft-zmatrix-7b-10epochs-unconditional",
    model_name="7b",
    batch_size=4,
    num_epochs=10,
    dataset_type="zmatrix",
    data_path=Path("data/mp20-zmatrix/mp-20/"),
    w_attributes=False, # unconditional generation
    task_probabilities={"generation": 1., "infill": 0.} # only generation task
)

output_dir= args.expdir / args.run_name
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
    label_names=["crystal_ids"], #this is just to get trainer to behave how I want
)

model = get_model(args, training_args.local_rank)
tokenizer = get_tokenizer(args)
smart_tokenizer_and_embedding_resize(model, tokenizer)
datasets = get_datasets(args, tokenizer)
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
    train_dataset=datasets["train"],
    eval_dataset=datasets["val"],
    max_seq_length=MAX_LENGTH,
    peft_config=lora_config,
    data_collator=data_collator,
    packing=True
)

print(args)

train_result = trainer.train()
trainer.save_state()
trainer.save_model()

