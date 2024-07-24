# LLM4StructGen
Fine-tuning LLMs for Benchmarking Textual Representations in Crystal Generation

### Supported Representations
- [x] Cartesian (CIF)
- [x] Z-matrix
- [x] Distance matrix
- [ ] SLICES

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
where `<ACCESS TOKEN>` is your HuggingFace authorization token. Note that the `/tmp/` folder is recycled after a session ends.

To start training, do

```
tune run lora_finetune_single_device --config configs/7B_lora_single_device.yaml
```

The first argument, lora_finetune_single_device, specifies the recipe to be used for this run. Recipes can be thought of as pipelines for training or inference. (see more [here](https://pytorch.org/torchtune/main/deep_dives/recipe_deepdive.html)).

By default, `wandb` is used for logging.

#### Modify Configs
You can copy the template `.yaml` config file and modify the fields in the copy for your training purposes. 

Additionally, you can also modify it in the command line:

```
tune run lora_finetune_single_device \
    --config configs/7B_lora_single_device.yaml \
    batch_size=8 \
    ...
```

### Environment Setup
```
conda create -n llm4structgen python=3.10
pip install torch torchvision

# install torchtune in target folder
git clone https://github.com/pytorch/torchtune.git
cd torchtune
pip install -e .

# install llm4structgen in target folder
git clone https://github.com/shuyijia/LLM4StructGen.git
cd LLM4StructGen
pip install -e .

# additional packages
pip install ase pymatgen wandb

# SLICES (not tested)
pip install tensorflow==2.15.0
pip install slices
```

> DO NOT install `torchtune` via `pip` directly (`pip install torchtune` won't work with our code)

### Key Updates
#### 07/24/24 (Shuyi)
After some experimentation, I decided to use [torchtune](https://github.com/pytorch/torchtune) as the training framework. 

This allows us to quickly fine-tune LLMs with the following benefits without writing tons of custom codes:
- several supported fine-tuning techniques, e.g. full fine-tuning, LoRA, QLoRA
- several supported LLMs including LLaMA-2/3/3.1, Gemma and others 
- tested and memory efficient quantization settings
- distributed training out of the box 

#### 07/04/24 (Shuyi)
- Restructured the entire codebase to enhance extensibility
- Added support for `translate`, `rotate` and `permute` of data
- Added support for `duplicity`, which allows duplicating a single sample in the dataset `x` times