# Long-Short-Working Transformer

[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/pylint-dev/pylint)

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

## Usage
### Environment
We recommend using the official PyTorch docker container and installing all of `requirments.txt`. In the future we'll provide a dockerfile which does all the setup for you.

### Envars
The following environment variables **must** be set:
- `WANDB_API_KEY` - Your WandB API key to track training/evaluation.
- `HF_API_KEY` - Your Hugging Face API key to access protected datasets.
- `HF_CACHE_DIR` - The directory to store loaded models and datasets.

The following environment variables are **optional** but recommended:
- `KERAS_BACKEND=torch` - Forces Keras to use the torch backend (for Keras NLP)
- `TOKENIZERS_PARALLELISM=true` - Forces HF tokenizers to support parallelism

## To Do List
- TODO: docstrings for packages, classes and functions
- TODO: improved commenting within functions
- TODO: add citations to readme
- TODO: improve SimCTG code for scattered padding + on device compute
- TODO: deploy models to 🤗 and use shield.io for pretty links
- TODO: use DDP compatible metrics

## Model Sizes
| Name | $d_{model}$ | $n_{layers}$ | $n_{heads}$ | $d_{key}$ | Parameters |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| **Tiny** 	| 768	| 12 | 12 | 64 | 153M |
| **Small** | 1024	| 18 | 16 | 64 | 343M |
| **Medium**| 1536	| 24 | 24 | 64 | 949M |
| **Large**	| 2048	| 36 | 32 | 64 | 2.5B |
| **XL**	| 3072	| 48 | 48 | 64 | 7.3B |
| **XXL**	| 4096	| 48 | 64 | 64 | 13B  |
