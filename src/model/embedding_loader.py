"""
Embedding loader module to load embedding matrices from HF models.

Contains:
    - embedding_loader: the loading method to retrieve embedding tensors.
"""

from transformers import AutoConfig, GPT2Model, OPTModel
import torch

from .configuration import LSWTConfig

def _get_model_type( name: str, cache_dir: str | None=None ):
    return AutoConfig.from_pretrained( name, cache_dir=cache_dir ).model_type

def _load_embeddings( name: str, cache_dir: str | None=None ):
    model_type = _get_model_type( name, cache_dir )

    if model_type == 'gpt2':
        print( 'Loading GPT2 style embeddings' )

        model_hf = GPT2Model.from_pretrained( name, n_layer=0, cache_dir=cache_dir )
        embeddings = model_hf.get_input_embeddings() # type: ignore

    elif model_type == 'opt':
        print( 'Loading OPT style embeddings' )

        model_hf = OPTModel.from_pretrained( name, num_hidden_layers=0, cache_dir=cache_dir )
        embeddings = model_hf.get_input_embeddings() # type: ignore

    else:
        raise ValueError( f'Embedding loading not implemented for {model_type} style models!' )

    return embeddings


def embedding_loader( config: LSWTConfig, cache_dir: str | None=None ) -> torch.Tensor:
    """
    Retrieves the weight tensor from a `torch.nn.Embedding` layer of a transformer from the HF hub

    Args:
        config (LSWTConfig): The LSWT model config containing the desired parent embeddings
        cache_dir (Optional[str], optional): Cache directory to download models to. Defaults to None.

    Returns:
        torch.Tensor: the `embedding.weight` tensor
    """
    embeddings = _load_embeddings( config.parent_embeddings, cache_dir )

    if embeddings.num_embeddings != config.vocab_size:
        raise ValueError( 'Loaded embeddings vocab size =/= config.vocab_size' )
    if embeddings.embedding_dim != config.d_vocab:
        raise ValueError( 'Loaded embeddings dim =/= config.d_model' )

    return embeddings.weight # type: ignore
