# Long-Short-Working Transformer

[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/pylint-dev/pylint)

A public repo for the LSWTransformer, codename `memory-transformer-pt4`.

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
- ~~**QK RMSNorm** grouped per head, suggested in multiple works to improve performance.~~ Added, but hasn't improved performance.
- **KV Recompute** during training for additional memory gradients. Can be disabled in config and is disabled when in eval mode.

### Planned Features
- **Partial RoPE** applying positional information to only a fraction of each head, suggested in GPT-Neo and GPT-J to improve performance.
- **Long Term Memory** as an additional `past_key_values` argument.
- **Segment Embeddings** utilising whitening to facilitate segment retrieval into long term memory.
- **Spectral Normalisation** of QK projections using the 'Spectra' post-optimizer.

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
- `PILE_PATH_PATTERN` - A python string format pattern pointing to the pile shards. Example: `.../the_pile/{:02d}.jsonl`
- `PILE_SHARDS` - The number of shards available for use.

The following environment variables are **optional** but recommended:
- `TOKENIZERS_PARALLELISM=true` - Forces HF tokenizers to support parallelism.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` - Helps reduce fragmentation of the PyTorch allocator.

The following envars should be used for debugging `torch.compile` related issues:
- `TORCH_LOGS="+dynamo"` to enable dynamo logging
- `TORCHDYNAMO_VERBOSE=1` to force verbose dynamo logging
- `TORCH_LOGS=recompiles` to enable dynamo logging on recompiles

## To Do List
- TODO: add missing docstrings for packages, classes and functions
- TODO: improved commenting within functions
- TODO: add citations to readme
- TODO: deploy models to 🤗 and use shield.io for pretty links
- TODO: move private functions out of classes when relevant or make public
- TODO: add flag to disable wandb stats tracking

## Model Sizes
| Name | $d_{model}$ | $d_{ffn}$ | $n_{layers}$ | $n_{heads}$ | $d_{key}$ | Parameters |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| **Tiny** 	| 768	| 2048 | 12 | 12 | 64 | 125M |
| **Medium**| 1536 | 4096 | 28 | 24 | 64 | 551M |

## Citation
If you wish to cite this work before then, you may refer to this repository and link to [my Semantic Scholar profile](https://www.semanticscholar.org/author/Avelina-Asada-Hadji-Kyriacou/2139984073).
