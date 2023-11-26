# Long-Short-Working Transformer
A private repo for the LSWTransformer, codename `memory-transformer-pt4`.

## Features
- **Transformer-XL style cache** for long sequence modelling.
- **Reversed RoPE** with adjusted base frequency to help extrapolate to longer sequences.
- **Attention Registers** to provide sinks for unwanted attention.
- **Flash Attention** for IO aware accelerated attention.
- **SwiGLU Activation** in the FFN layers for higher model capacity.
- **HuggingFace Model Format** for integration with the 🤗 ecosystem.
- **Warmstart Word Embeddings** taken from `facebook/opt-125m` to accelerate convergence.
- **Input/Output Projections** to decouple embedding matrix dimension from model dimension.
- **SimCTG Loss** to encourage embedding diversity.
- **Sophia Optimizer** for faster convergence.

## To Do List
- TODO: docstrings for packages, classes and functions
- TODO: improved commenting within functions
- TODO: add citations to readme
- TODO: improve SimCTG code for scattered padding + on device compute
- TODO: add embedding + projection helper method to LSWTModel and LSWTForCausalLM
- TODO: refactor cache transfer list comprehension

## Model Sizes
| Name | $d_{model}$ | $n_{layers}$ | $n_{heads}$ | $d_{key}$ | Parameters |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| **Tiny** 	| 768	| 12 | 12 | 64 | 153M |
| **Small** | 1024	| 18 | 16 | 64 | 343M |
| **Medium**| 1536	| 24 | 24 | 64 | 949M |
| **Large**	| 2048	| 36 | 32 | 64 | 2.5B |
| **XL**	| 4096	| 48 | 64 | 64 | 13B  |
