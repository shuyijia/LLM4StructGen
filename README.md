# LLM4StructGen
Fine-tuning LLMs for Benchmarking Textual Representations in Crystal Generation

### TO-DOs
- [] Batch inference
- [] CDVAE evaluation workflow
- [] Decoding for distance matrix

### Supported Representations
- [x] Cartesian (CIF)
- [x] Z-matrix
- [x] Distance matrix
- [x] SLICES

### Tested Training
| Recipe                      | Model       | Cluster                 | GPUs          | Batch Size | VRAM | Time (hrs) |
|-----------------------------|-------------|-------------------------|---------------|------------|---------------------|----------------------|
| `lora_finetune_single_device` | LLaMA-2-7B  | Perlmutter login   | 1 x A100 40GB | 4          | 15GB/GPU                | 1-2/epoch                |
| `lora_finetune_distributed`   | LLaMA-2-13B | Perlmutter compute | 4 x A100 80GB | 2          | 20GB/GPU                | 1 /epoch                   |

## Usage
```
pip install -e .
```

In general, to train a model in `torchtune` we can do the following:

```
# single device
tune run [RECIPE] --config [PATH/TO/YAML]

# distributed 
# 1 node, 4 GPUs
tune run --nnodes 1 --nproc_per_node 4 [RECIPE] --config [PATH/TO/YAML]
```

The first argument, [RECIPE], specifies the template to be used for this run. Recipes can be thought of as pipelines for training or inference. (see more [here](https://pytorch.org/torchtune/main/deep_dives/recipe_deepdive.html)).

> The default logging directory in the config files is set to `exp/logs`. To use the same, do `mkdir -p exp/logs`.

### Sample Run (LLaMA-2 with LoRA on Cartesian Representations)
First, we need to download checkpoint weights from HuggingFace:
```
tune download meta-llama/Llama-2-7b-hf \
  --output-dir /tmp/Llama-2-7b-hf \
  --hf-token <ACCESS TOKEN>
```
where `<ACCESS TOKEN>` is your HuggingFace authorization token. Note that the `/tmp/` folder is recycled after a session ends. You might want to move the weights to the scratch directory or the project space.

To start training, do

```
tune run lora_finetune_single_device \
--config configs/train/llama2/7B_lora_single_device.yaml
```

By default, `wandb` is used for logging.

### Generation
```
tune run llm4structgen/generation/inference.py \
--config configs/inference/generation.yaml \
generation.n_structures=10 \
...
```

You need to modify the checkpoint paths in the yaml file to load the finetuned checkpoints.

Currently, only single-device non-batch inference is supported. 

> Note: The generation script produces only text output. Post-decoding is required to convert these strings into Atoms objects. This decoupling ensures that any parsing errors are handled separately from the generation process.

### Modify Configs
You can copy the template `.yaml` config file and modify the fields in the copy for your training purposes. 

Additionally, you can also modify it in the command line:

```
tune run lora_finetune_single_device \
    --config configs/train/llama2/7B_lora_single_device.yaml \
    batch_size=8 \
    dataset.representation_type=distance \
    ...
```

## Environment Setup
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

# SLICES
pip install tensorflow==2.15.0
pip install slices
```

> DO NOT install `torchtune` via `pip` directly (`pip install torchtune` won't work with our code)

The current installation of `tensorflow==2.15.0` and `slices` will throw the following warnings:

```
[...] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
[...] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
[...] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
WARNING:tensorflow:
[...]
experimental_relax_shapes is deprecated, use reduce_retracing instead
```

These should have no effect on our use cases.

## Key Updates
### 07/24/24 (Shuyi)
After some experimentation, I decided to use [torchtune](https://github.com/pytorch/torchtune) as the training framework. 

This allows us to quickly fine-tune LLMs with the following benefits without writing tons of custom codes:
- several supported fine-tuning techniques, e.g. full fine-tuning, LoRA, QLoRA
- several supported LLMs including LLaMA-2/3/3.1, Gemma and others 
- tested and memory efficient quantization settings
- distributed training out of the box 

### 07/04/24 (Shuyi)
- Restructured the entire codebase to enhance extensibility
- Added support for `translate`, `rotate` and `permute` of data
- Added support for `duplicity`, which allows duplicating a single sample in the dataset `x` times