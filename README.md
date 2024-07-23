# LLM4StructGen
Fine-tuning LLMs for Benchmarking Textual Representations in Crystal Generation

### Usage
```
pip install -e .
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
- [ ] Distance matrix
- [ ] SLICES

### Supported Models
- [x] LLaMA-2 (7B/13B/70B)
- [ ] LLaMA-3
- [ ] Gemma
- [ ] Mixtral

### Key Updates
#### 07/04/24
- Restructured the entire codebase to enhance extensibility
- Added support for `translate`, `rotate` and `permute` of data
- Added support for `duplicity`, which allows duplicating a single sample in the dataset `x` times