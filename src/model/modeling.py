from typing import Tuple, List, Optional

from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
import torch

from .configuration import LSWTConfig
from .layers import SharedEmbeddings, RotaryEmbedding, LSWTBlock

class LSWTPreTrainedModel( PreTrainedModel ):
    config_class = LSWTConfig
    base_model_prefix = 'model'
    
    def _init_weights( self, module ):
        std = self.config.init_std
        
        if isinstance( module, torch.nn.Linear ):
            module.weight.data.normal_( mean=0.0, std=std )
            if module.bias is not None:
                module.bias.data.zero_()
                
        elif isinstance( module, torch.nn.Embedding ):
            module.weight.data.normal_( mean=0.0, std=std )
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
    
        elif isinstance( module, torch.nn.LayerNorm ):
            module.bias.data.zero_()
            module.weight.data.fill_( 1.0 )
    
    def get_param_groups( self ):
        decay_params = []
        non_decay_params = []
        
        non_decay_names = [ 'norm', 'bias', 'embedding.weight' ]
        
        for name, p in self.named_parameters():
            if p.requires_grad:
                if any( i in name for i in non_decay_names ):
                    non_decay_params.append( p )
                
                else:
                    decay_params.append( p )
        
        return [
            { 'params': decay_params },
            { 'params': non_decay_params, 'weight_decay': 0.0 }
        ]
    
    def cache_to( self, cache, device, trim=0 ):
        if cache is not None:
            # cache = tuple( [ tuple( [ kv.detach()[ :, :, -trim : , : ].to( device=device, non_blocking=True ) for kv in layer ] ) for layer in cache ] )
            cache = [ [ kv.detach()[ :, :, -trim : , : ].to( device=device, non_blocking=False ) for kv in layer ] for layer in cache ]
        return cache
                

class LSWTModel( LSWTPreTrainedModel ):
    def __init__( self, config: LSWTConfig, parent_embeddings: Optional[torch.Tensor]=None ):
        super().__init__( config )
        
        self.input_embedding = SharedEmbeddings( config.vocab_size, config.d_vocab )
        self.input_proj = torch.nn.Linear( config.d_vocab, config.d_model, bias=False )
        self.input_norm = torch.nn.LayerNorm( config.d_model )
        
        self.rope_embedding = RotaryEmbedding(
            config.d_model // config.n_heads,
            config.rope_xpos_scale,
            config.rope_xpos_enabled,
            config.rope_base_freq,
            config.rope_reversed,
        )
        
        self.blocks = torch.nn.ModuleList( [ LSWTBlock( config ) for _ in range( config.n_layers ) ] )
        
        self.output_norm = torch.nn.LayerNorm( config.d_model )
        
        self.post_init()
        
        if parent_embeddings is not None:
            self.input_embedding.embedding.weight = torch.nn.Parameter( torch.clone( parent_embeddings ) )
        
        if not config.trainable_embeddings:
            self.input_embedding.requires_grad_( False )
            self.input_embedding.half()
        else:
            self.input_embedding.requires_grad_( True )
        
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[List[torch.Tensor]]] = None, # (n_layers)+(k_v)+(B, heads, seq_length, d_key)
    ):
        hidden_state_list = []
        past_key_value_list = []
        
        if ( input_ids is not None ) and ( inputs_embeds is not None ):
            raise ValueError( 'You cannot specify both input_ids and inputs_embeds at the same time' )
        if ( input_ids is None ) and ( inputs_embeds is None ):
            raise ValueError( 'You have to specify either input_ids or inputs_embeds' )
        
        # Embed inputs if present
        if input_ids is not None:
            embeddings = self.input_embedding( input_ids, mode='embed' )
            embeddings = self.input_proj( embeddings )
        else:
            embeddings = inputs_embeds
        embeddings = self.input_norm( embeddings )
        
        hidden_state_list.append( embeddings )
        
        # Get total sequence length including past
        sequence_length = embeddings.shape[-2] + ( past_key_values[0][0].shape[-2] if past_key_values is not None else 0 )
        
        # RoPE embeddings
        rope_pos, rope_scale = self.rope_embedding( sequence_length, embeddings.device )
        
        for i in range( self.config.n_layers ):
            curr_key_values = past_key_values[i] if past_key_values is not None else None
            embeddings, new_key_values = self.blocks[i]( embeddings, curr_key_values, rope_pos, rope_scale )
            
            hidden_state_list.append( embeddings )
            past_key_value_list.append( new_key_values )
        
        # Final normalisation
        embeddings = self.output_norm( embeddings )
        
        return BaseModelOutputWithPast(
            last_hidden_state=embeddings,
            past_key_values=past_key_value_list, # type: ignore
            hidden_states=hidden_state_list, # type: ignore
            attentions=None,
        )
            
        

class LSWTForCausalLM( LSWTPreTrainedModel ):
    def __init__( self, config: LSWTConfig, parent_embeddings: Optional[torch.Tensor]=None ):
        super().__init__( config )
        
        self.model = LSWTModel( config, parent_embeddings )
        self.head_proj = torch.nn.Linear( config.d_model, config.d_vocab, bias=False )
        self.post_init()
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None, # (n_layers)+(k_v)+(B, heads, seq_length, d_key)
        
        use_cache=True,
        return_dict=True,
        output_attentions=False,
        output_hidden_states=True,
    ):
        assert return_dict, "Must always return_dict"
        assert not output_attentions, "Must never output_attentions"
        
        base_outputs = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
        )
        
        embeddings = self.head_proj( base_outputs.last_hidden_state )
        logits = self.model.input_embedding( embeddings, mode='linear' )
        
        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=base_outputs.past_key_values if use_cache else None,
            hidden_states=( base_outputs.hidden_states + [ base_outputs.last_hidden_state ] ) if output_hidden_states else None,
            attentions=base_outputs.attentions,
        )
    
    # TODO: do this legit
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # # Some generation methods already pass only the last input ID
            # if input_ids.shape[1] > past_length:
            #     remove_prefix_length = past_length
            # else:
            #     # Default to old behavior: keep only final ID
            #     remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, -1 : ]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
            }
        )
        return model_inputs

