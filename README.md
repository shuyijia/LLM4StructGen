# LLM4StructGen
Fine-tuning LLMs for Benchmarking Textual Representations in Crystal Generation

### Usage
```
pip install -e .
```

#### Sample run (LLaMA-2 with LoRA on Cartesian Representations)
First, we need to download checkpoint weights from HuggingFace:
```
tune download meta-llama/Llama-2-7b-hf \
  --output-dir /tmp/Llama-2-7b-hf \
  --hf-token <ACCESS TOKEN>
```
where `<ACESS TOKEN>` is your HuggingFace authorization token.

To start training, do

```
tune run lora_finetune_single_device --config configs/7B_lora_single_device.yaml
```

The first argument, lora_finetune_single_device, specifies the recipe to be used for this run. Recipes can be thought of as pipelines for training or inference. (see more [here](https://pytorch.org/torchtune/main/deep_dives/recipe_deepdive.html)).

By default, `wandb` is used for logging.

#### Modify Configs
You can copy the template `.yaml` config file and modify the fields in the copy for your training purposes. 

Additionally, you can also modify it in the commandline:

```
tune run lora_finetune_single_device --config configs/7B_lora_single_device.yaml batch_size=8
```

### Environment Setup
```
conda create -n llm4structgen python=3.10
conda activate llm4structgen

# mamba for faster solve
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# finetuning
pip install transformers wandb trl peft pymatgen bitsandbytes sentencepiece ase

# SLICES
pip install tensorflow==2.15.0
pip install slices
```

### Supported Representations
- [x] Cartesian (CIF)
- [x] Z-matrix
- [x] Distance matrix
- [ ] SLICES

### Key Updates
#### 07/24/24 (Shuyi)
After some experimentation, I decided to use [torchtune](https://github.com/pytorch/torchtune) as a training framework. 

This allows us to quickly fine-tune LLMs with the following benefits without writing tons of custom codes:
- several supported fine-tuning techniques, e.g. full fine-tuning, LoRA, QLoRA
- several supported LLMs including LLaMA-2/3/3.1, Gemma and others 
- tested and memory efficient quantization settings
- distributed training out of the box 

#### 07/04/24 (Shuyi)
- Restructured the entire codebase to enhance extensibility
- Added support for `translate`, `rotate` and `permute` of data
- Added support for `duplicity`, which allows duplicating a single sample in the dataset `x` times