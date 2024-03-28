import torch 
from transformers import ( 
    LlamaForCausalLM,
    LlamaTokenizer, 
    TrainingArguments,
    BitsAndBytesConfig
)
from trl import SFTTrainer

from peft import (
    LoraConfig
)

from utils import *
from constants import *
from cif_dataset import DataCollatorForSupervisedDataset

# model and tokenizer
def get_model_and_tokenizer(args, rank):
    llama_options = args.model_name.split("-")
    is_chat = len(llama_options) == 2
    model_size = llama_options[0]

    def llama2_model_string(model_size, chat):
        chat = "chat-" if chat else ""
        return f"meta-llama/Llama-2-{model_size.lower()}-{chat}hf"

    model_string = llama2_model_string(model_size, is_chat)

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

    llama_tokenizer = LlamaTokenizer.from_pretrained(
        model_string,
        model_max_length=MAX_LENGTH,
        padding_side="right",
        use_fast=False,
    )

    smart_tokenizer_and_embedding_resize(model, llama_tokenizer)

    return model, llama_tokenizer

def get_trainer(args):
    output_dir = args.expdir / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # create TrainingArguments
    training_args = TrainingArguments(
        # output_dir
        output_dir=output_dir,
        # precision
        bf16=args.bf16,
        fp16=args.fp16,
        # epoch, batch size, steps
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size // 2,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        # save and logging
        logging_strategy="steps",
        logging_steps=args.log_freq,
        save_strategy="steps",
        save_steps=args.save_freq,
        evaluation_strategy="epoch",
        # learning rate
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        # wandb
        report_to="wandb",
    )

    model, llama_tokenizer = get_model_and_tokenizer(args, training_args.local_rank)
    datasets = setup_datasets(args, llama_tokenizer)
    data_collator = DataCollatorForSupervisedDataset(llama_tokenizer)

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    return SFTTrainer(
        model,
        train_dataset=datasets["train"],
        eval_dataset=datasets["val"],
        args=training_args,
        max_seq_length=MAX_LENGTH,
        peft_config=lora_config,
        data_collator=data_collator,
        packing=True,
    )

if __name__ == "__main__":
    args = ModelConfig(
        run_name="13b-full-no-val",
        batch_size=2
    )

    trainer = get_trainer(args)

    if args.resume_dir is not None:
        train_result = trainer.train(resume_from_checkpoint=args.resume_dir)
    else:
        train_result = trainer.train()

    print(train_result)
    trainer.save_state()
    trainer.save_model()