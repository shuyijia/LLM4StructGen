# LLM4StructGen
Fine-tuning LLMs for Benchmarking Textual Representations in Crystal Generation

### Usage
```
pip install -e .
```

Full environment setup instruction coming soon.

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