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
- **SimCTG Loss (optional)** to encourage embedding diversity.
- **LaProp Optimizer** for faster convergence.

## Training and Evaluation Data
LSWT is currently being trained on the original distribution of The Pile dataset. Runs without DDP use all 30 shards (0-29 inclusive), while runs using DDP are trained using the first 24 shards (0-23 inclusive) for easier distribution of data for typical world sizes.

We evaluate our models between 'epochs' using a non-chunked version of Wikitext. At the end of training we make use of the full validation split of The Pile Uncopyrighted to gauge model performance, we specifically use this new publicly available version of The Pile as the original version is now defunct and only available through non official sources. 

## Usage
### Environment
We recommend using the official PyTorch docker container and installing all of `requirments.txt`. In the future we'll provide a dockerfile which does all the setup for you.

### Envars
The following environment variables **must** be set:
- `WANDB_API_KEY` - Your WandB API key to track training/evaluation.
- `WANDB_PROJECT_NAME` - The WandB project destination for all runs.
- `HF_API_KEY` - Your Hugging Face API key to access protected datasets.
- `HF_CACHE_DIR` - The directory to store loaded models and datasets.

The following environment variables are **optional** but recommended:
- `TOKENIZERS_PARALLELISM=true` - Forces HF tokenizers to support parallelism.
- `KERAS_BACKEND=torch` - Forces Keras to use the torch backend (for Keras NLP).
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` - Helps reduce fragmentation of the PyTorch allocator.

## To Do List
- TODO: add missing docstrings for packages, classes and functions
- TODO: improved commenting within functions
- TODO: add citations to readme
- TODO: deploy models to 🤗 and use shield.io for pretty links
- TODO: move private functions out of classes when relevant or make public

## Model Sizes
| Name | $d_{model}$ | $n_{layers}$ | $n_{heads}$ | $d_{key}$ | Parameters |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| **Tiny** 	| 768	| 12 | 12 | 64 | 153M |
| **Small** | 1024	| 18 | 16 | 64 | 343M |
| **Medium**| 1536	| 24 | 24 | 64 | 949M |
| **Large**	| 2048	| 36 | 32 | 64 | 2.5B |
| **XL**	| 3072	| 48 | 48 | 64 | 7.3B |
| **XXL**	| 4096	| 48 | 64 | 64 | 13B  |
