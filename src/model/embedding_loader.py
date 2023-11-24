from .configuration import LSWTConfig
from transformers import AutoConfig, GPT2Model, OPTModel

from typing import Optional

import torch

def _get_model_type( name: str, cache_dir: Optional[str]=None ):
    return AutoConfig.from_pretrained( name, cache_dir=cache_dir ).model_type

def _load_embeddings( name: str, cache_dir: Optional[str]=None ):
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
    

def embedding_loader( config: LSWTConfig, cache_dir: Optional[str]=None ) -> torch.Tensor:
    embeddings = _load_embeddings( config.parent_embeddings, cache_dir )
    
    assert embeddings.num_embeddings == config.vocab_size, 'Loaded embeddings vocab size =/= config.vocab_size'
    assert embeddings.embedding_dim == config.d_vocab, 'Loaded embeddings dim =/= config.d_model'
    
    return embeddings.weight # type: ignore

    