"""
Module containing the LSWT model implementations.

Contains:
    - LSWTPreTrainedModel: abstract base class for the LSWT.
    - LSWTModel: backbone base class with no head.
    - LSWTForCausalLM: causal head model containing an LSWTModel instance.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from operator import itemgetter
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, ModelOutput
from transformers.activations import ACT2FN
import torch
import torch.nn.functional as F

from .configuration import LSWTConfig, LSWTPoolerConfig
from .layers import RotateLayer, SharedEmbeddings, RotaryEmbedding, LSWTBlock, ActGLU, complex_scan, complex_selective_scan, prolu_ste, prolu_relu

class LSWTPreTrainedModel( PreTrainedModel ):
    """
    Base class for LSW Transformer models.

    Class attributes:
        - config_class: The config class to use for this model architecture.
        - base_model_prefix: A string indicating the attribute associated to the base model in derived
        classes of same architecture adding modules on top of the base model.
    """

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
            
        elif isinstance( module, LSWTSparseAutoEncoder ):
            torch.nn.init.kaiming_uniform_( module.decoder_weight.data )
            torch.nn.init.kaiming_uniform_( module.encoder_weight.data )
            module.encoder_bias.data.zero_()
            module.decoder_bias.data.zero_()

    def get_param_groups( self, decay_mask: Sequence[str] ) -> list[dict]:
        """
        Returns optimizer parameter groups with weight decay disabled for certain params.
        Weight decay is disabled for:
            - layer norm
            - bias terms
            - embedding weights

        Args:
            decay_mask (Sequence[str]): list of string patterns which disable weight decay

        Returns:
            List[Dict]: list of param groups to be used by the optimizer
        """

        decay_params = []
        non_decay_params = []

        for name, p in self.named_parameters():
            if p.requires_grad:
                if any( i in name for i in decay_mask ):
                    non_decay_params.append( p )

                else:
                    decay_params.append( p )

        return [
            { 'params': decay_params },
            { 'params': non_decay_params, 'weight_decay': 0.0 }
        ]

    def trim_cache(
        self,
        cache: list[torch.Tensor],
        trim: int | None = 0,
    ) -> list[torch.Tensor]:
        """ Trims the key and value tuple to a max length.

        Should be applied per layer, rather than to the list of all past key values.

        Args:
            cache (list[torch.Tensor]): The key value cache to trim.
            trim (int, optional): Desired trim size. Zero means no trim. Defaults to 0.

        Returns:
            list[torch.Tensor]: Trimmed cache
        """

        if trim is not None:
            return [ kv[ :, :, -trim :, : ] for kv in cache ]
        return cache

    def cache_to(
        self,
        cache: list[list[torch.Tensor]] | None,
        device: str | torch.device,
        trim: int = 0,
        non_blocking: bool = False,
    ) -> list[list[torch.Tensor]] | None:
        """
        Moves KV cache between devices.

        TODO: deprecate trim != 0

        Args:
            cache (Optional[list[list[torch.Tensor]]]): Key value cache to move
            device (str | torch.device): the device to move to
            trim (int, optional): Desired trim size. Zero means no trim. Defaults to 0.
            non_blocking (bool): Determines if the transfer should be `non_blocking`. Defaults to False.

        Returns:
            list[list[torch.Tensor]]: Moved key value cache
        """

        if cache is not None:
            cache = [
                [
                    kv.detach()[ :, :, -trim : , : ].to(
                        device=device,
                        non_blocking=non_blocking
                    ) for kv in layer
                ] for layer in cache
            ]
        return cache


class LSWTModel( LSWTPreTrainedModel ):
    """
    Base model class for the LSW Transformer decoder.

    Contains the input embeddings and model backbone, but does not contain the model head.
    """

    def __init__( self, config: LSWTConfig, parent_embeddings: torch.Tensor | None=None ):
        """
        Constructs a new LSWTModel.

        Args:
            config (LSWTConfig): Config for the LSWT architecture
            parent_embeddings (Optional[torch.Tensor], optional): Optinal warm start embeddings.
        """

        super().__init__( config )

        self.input_embedding = SharedEmbeddings( config.vocab_size, config.d_vocab )
        self.input_proj = torch.nn.Linear( config.d_vocab, config.d_model, bias=False )
        self.input_norm = torch.nn.LayerNorm( config.d_model )

        self.rope_embedding = RotaryEmbedding( config )

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

    def get_input_embeddings( self ):
        return self.input_embedding.embedding

    def embed_input( self, input_ids: torch.LongTensor ) -> torch.Tensor:
        """
        Embedds and projects inputs.

        Args:
            input_ids (torch.LongTensor): input ids of size [Batch x Seq_Length]

        Returns:
            torch.Tensor: input embeddings of size [Batch x Seq_Length x D_Model]
        """
        embeddings = self.input_embedding( input_ids, mode='embed' )
        embeddings = self.input_proj( embeddings )
        return embeddings

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        past_key_values: list[list[torch.Tensor]] | None = None,
        max_key_values: int | None = None,
    ) -> BaseModelOutputWithPast:
        """
        Forward pass function.

        Args:
            input_ids (Optional[torch.LongTensor], optional): input ids of size [Batch x Seq_Length]
            inputs_embeds (Optional[torch.Tensor], optional): input embeddings of size [Batch x Seq_Length x D_Model]
            past_key_values (Optional[List[List[torch.Tensor]]], optional): Previous KV cache for fast decoding or memory.
            max_key_values (int, optional): The max number of past states to keep during generation. Defaults to None.

        Raises:
            ValueError: when both input_ids and inputs_embeds are passed.
            ValueError: when neither input_ids or inputs_embeds are passed.

        Returns:
            BaseModelOutputWithPast: Model outputs
        """

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

        # RoPE embeddings
        rope_pos, rope_scale = self.rope_embedding( embeddings, past_key_values )

        for i in range( self.config.n_layers ):
            curr_key_values = past_key_values[i] if past_key_values is not None else None
            embeddings, new_key_values = self.blocks[i]( embeddings, curr_key_values, rope_pos, rope_scale )

            hidden_state_list.append( embeddings )
            past_key_value_list.append( self.trim_cache( new_key_values, max_key_values ) )

        # Final normalisation
        embeddings = self.output_norm( embeddings )

        return BaseModelOutputWithPast(
            last_hidden_state=embeddings,
            past_key_values=past_key_value_list, # type: ignore
            hidden_states=hidden_state_list, # type: ignore
            attentions=None,
        )



class LSWTForCausalLM( LSWTPreTrainedModel ):
    """
    Causal LM model class for the LSW Transformer.

    Contains an LSWTModel and a projection layer for the shared embedding LM head.
    """

    def __init__( self, config: LSWTConfig, parent_embeddings: torch.Tensor | None=None ):
        """
        Constructs a new LSWTForCausalLM.

        Args:
            config (LSWTConfig): Config for the LSWT architecture
            parent_embeddings (Optional[torch.Tensor], optional): Optinal warm start embeddings.
        """

        super().__init__( config )

        self.model = LSWTModel( config, parent_embeddings )
        self.head_proj = torch.nn.Linear( config.d_model, config.d_vocab, bias=False )
        self.post_init()

    def get_input_embeddings( self ):
        return self.model.get_input_embeddings()

    def embed_input( self, input_ids: torch.LongTensor ) -> torch.Tensor:
        """
        Embedds and projects inputs.

        Args:
            input_ids (torch.LongTensor): input ids of size [Batch x Seq_Length]

        Returns:
            torch.Tensor: input embeddings of size [Batch x Seq_Length x D_Model]
        """
        return self.model.embed_input( input_ids )

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        past_key_values: list[list[torch.Tensor]] | None = None,

        use_cache=True,
        return_dict=True,
        output_attentions=False,
        output_hidden_states=True,

        max_key_values: int | None = None,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass function.

        Args:
            input_ids (Optional[torch.LongTensor], optional): input ids of size [Batch x Seq_Length]
            inputs_embeds (Optional[torch.Tensor], optional): input embeddings of size [Batch x Seq_Length x D_Model]
            past_key_values (Optional[List[List[torch.Tensor]]], optional): Previous KV cache for fast decoding or memory.

            use_cache (bool, optional): If set to `True`, returns KV cache for fast decoding or memory. Defaults to True.
            return_dict (bool, optional): Whether or not to return a CausalLMOutputWithPast. Must be True.
            output_attentions (bool, optional): Returns attentions for all layers. Must be False.
            output_hidden_states (bool, optional): Whether or not to return the hidden states of all layers. Defaults to True.

            max_key_values (int, optional): The max number of past states to keep during generation. Defaults to None.

        Returns:
            CausalLMOutputWithPast: Model outputs
        """

        assert return_dict, "Must always return_dict"
        assert not output_attentions, "Must never output_attentions"

        base_outputs = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            max_key_values=max_key_values,
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

    # TODO: do this legit + remove pylint ignores # pylint: disable=W0511
    def prepare_inputs_for_generation( # type: ignore
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs
    ):
        # pylint: disable=W0613
        # pylint: disable=W0612
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Trim the past key values if a max_key_values model arg is passed
            max_key_values = kwargs.get( 'max_key_values', 0 ) or 0
            past_key_values = self.cache_to( past_key_values, device=input_ids.device, trim=max_key_values )
            input_ids = input_ids[:, -1 : ]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get( 'use_cache' ),
                'max_key_values': kwargs.get( 'max_key_values', self.config.rope_positions )
            }
        )
        return model_inputs

    def _reorder_cache( self, past_key_values, beam_idx ):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past


@dataclass
class SAELoss( ModelOutput ):
    """ Base class for SAE outputs.
    """
    
    reconstruction_loss: torch.Tensor
    l1_penalty: torch.Tensor

class LSWTSparseAutoEncoder( torch.nn.Module ):
    def __init__( self, pooler_config: LSWTPoolerConfig, d_model: int ):
        super().__init__()
        
        self.pooler_config = pooler_config
        
        n_dim = d_model
        m_dim = pooler_config.embedding_size
        
        assert m_dim
        assert pooler_config.pooler_activation in [ 'relu', 'prolu_ste', 'prolu_relu' ] # TODO: raise instead of assert
        assert not pooler_config.pooler_activation_gated # TODO: raise instead of assert
        
        self.activation = pooler_config.pooler_activation
        
        self.encoder_weight = torch.nn.Parameter( torch.empty( m_dim, n_dim ) )
        self.encoder_bias = torch.nn.Parameter( torch.empty( m_dim ) )
        
        self.decoder_weight = torch.nn.Parameter( torch.empty( n_dim, m_dim ) )
        self.decoder_bias = torch.nn.Parameter( torch.empty( n_dim ) )
    
    def activate( self, x_prime: torch.Tensor, w: torch.Tensor, b: torch.Tensor ) -> torch.Tensor:
        match self.activation:
            case 'relu':
                return torch.relu( F.linear( x_prime, w, b ) ) # pylint: disable=E1102
            
            case 'prolu_ste':
                return prolu_ste( F.linear( x_prime, w, None ), b ) # pylint: disable=E1102
            
            case 'prolu_relu':
                return prolu_relu( F.linear( x_prime, w, None ), b ) # pylint: disable=E1102
            
            case _:
                raise ValueError( 'Invalid SAE activation!' )
    
    def forward( self, hidden_states: torch.Tensor, output_auxiliary = False ):
        x_prime = hidden_states - self.decoder_bias
        
        latent = self.activate( x_prime, self.encoder_weight, self.encoder_bias )
        
        if output_auxiliary:
            x_hat = F.linear( latent, self.decoder_weight, self.decoder_bias ) # pylint: disable=E1102
            
            l2_penalty = ( hidden_states.detach() - x_hat ).pow( 2 ).mean()
            l1_penalty = torch.norm( latent, p=1, dim=-1 ).mean()
            
            return latent, SAELoss( l2_penalty, l1_penalty )
        else:
            return latent
            
            


@dataclass
class DPHOutput( ModelOutput ):
    """ Base class for DPH reward outputs.
    """
    
    rewards: Mapping[str, torch.Tensor]
    """ Mapping of head name to the rewards computed on the <|im_end|> token."""
    
    latent_states: torch.Tensor | None = None
    """ Latent states computed on the <|im_end|> token. Returns None when `output_latent_states=False` """
    
    aux_loss: SAELoss | None = None
    """ Auxiliary loss for SAEs """

class LSWTPooler( torch.nn.Module ):
    def __init__( self, pooler_config: LSWTPoolerConfig, d_model: int, enable_bias: bool ):
        super().__init__()
        
        self.pooler_config = pooler_config
        
        if pooler_config is None:
            raise ValueError( 'pooler_config must be defined!' )
        
        if pooler_config.reward_heads is None:
            raise ValueError( 'reward_heads must be defined. If no heads are desired please use an empty list.' )
        
        self.pooler_pipeline = torch.nn.Sequential()
        self.embedding_dropout = torch.nn.Dropout( p=pooler_config.embedding_dropout )
        self.layer_dropout = torch.nn.Dropout( p=pooler_config.layer_dropout )
        
        self.layer_norm_pre = torch.nn.LayerNorm( d_model ) if pooler_config.layer_pooling_norm in [ 'pre', 'both' ] else torch.nn.Identity()
        self.layer_norm_post = torch.nn.LayerNorm( d_model ) if pooler_config.layer_pooling_norm in [ 'post', 'both' ] else torch.nn.Identity()

        self.token_norm_pre = torch.nn.LayerNorm( d_model ) if pooler_config.token_pooling_norm in [ 'pre', 'both' ] else torch.nn.Identity()
        self.token_norm_post = torch.nn.LayerNorm( d_model ) if pooler_config.token_pooling_norm in [ 'post', 'both' ] else torch.nn.Identity()
        
        self.token_rotate = RotateLayer( d_model, d_model * pooler_config.token_pooling_rotation_expansion ) if pooler_config.token_pooling_rotation else None
        
        self.token_gate = torch.nn.Linear( d_model, d_model, True ) if pooler_config.token_pooling_gate else None
        self.token_gate_act = ACT2FN[pooler_config.token_pooling_gate] if pooler_config.token_pooling_gate else None

        if pooler_config.layer_pooling == 'weighted_sum':
            assert not isinstance( pooler_config.layer_select, int )
            self.layer_weighting = torch.nn.Parameter( torch.empty( len( pooler_config.layer_select ), 1, 1, 1 ), requires_grad=True )
            self.layer_weighting.data.fill_( 0.0 )
        else:
            self.layer_weighting = None
        
        if pooler_config.token_pooling == 'ema':
            assert pooler_config.token_pooling_ema_beta
            
            beta = pooler_config.token_pooling_ema_beta
            self.ema_beta = torch.nn.Parameter( torch.tensor( beta / ( 1.0 - beta ) ).log(), requires_grad=False )
            
            match pooler_config.token_pooling_ema_beta_learnable:
                case 'global':
                    self.ema_weight = torch.nn.Parameter( torch.empty( [ 1, 1, 1 ] ), requires_grad=True )
                case 'activation':
                    self.ema_weight = torch.nn.Parameter( torch.empty( [ 1, 1, d_model * pooler_config.token_pooling_rotation_expansion ] ), requires_grad=True )
                case None:
                    self.ema_weight = torch.nn.Parameter( torch.empty( [ 1, 1, 1 ] ), requires_grad=False )
                case _:
                    raise ValueError( 'Invalid `token_beta_learnable` value' )
            
            self.ema_weight.data.zero_()
        else:
            self.ema_weight = None
            self.ema_beta = None

        # Match the reward pooler type
        match pooler_config.pooler_function:
            
            # If identity we only do dropout
            case 'identity':
                # Set embedding size to that of d_model as pooler is passthrough
                embedding_size = d_model
                
                # Add dropout as final layer
                self.pooler_pipeline.append( torch.nn.Identity() )
            
            # If projection do: linear -> activation -> dropout
            case 'projection':
                # Assertions to make linter happy. We should have raised an error already.
                assert pooler_config.embedding_size is not None
                assert pooler_config.pooler_activation is not None
                
                # Set embedding size to that of config
                embedding_size = pooler_config.embedding_size
                
                # Set intermediate size to 2x if gated, otherwise 1x
                intermediate_size = pooler_config.embedding_size * ( 2 if pooler_config.pooler_activation_gated else 1 )
                
                # Set activation to SwiGLU if gated, otherwise get activation by name
                activation = ActGLU( pooler_config.pooler_activation ) if pooler_config.pooler_activation_gated else ACT2FN[pooler_config.pooler_activation]
                
                # Append linear -> activation -> dropout
                self.pooler_pipeline.append( torch.nn.Linear( d_model, intermediate_size, bias=enable_bias ) )
                self.pooler_pipeline.append( activation )
            
            case 'sae':
                self.pooler_pipeline.append( LSWTSparseAutoEncoder( pooler_config, d_model ) )
                
                 # Set intermediate size to 2x if gated, otherwise 1x
                intermediate_size = pooler_config.embedding_size
                
            case _:
                raise ValueError( 'Invalid pooler type.' )

        # Create dict of reward head projections
        self.reward_heads = torch.nn.ModuleDict( {
            name: torch.nn.Linear( embedding_size, 1, bias=pooler_config.reward_head_bias )
            for name in pooler_config.reward_heads
        } )
    
    def aggregate_states(
        self,
        hidden_states: tuple[torch.Tensor],
        input_ids: torch.LongTensor,
        start_id: int,
        end_id: int,
        return_all=False,
        prefix_type: str | None = 'assistant',
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        pooler_config = self.pooler_config
        assert pooler_config
        
        # Select the layer or subset of layers
        if isinstance( pooler_config.layer_select, int ):
            states = hidden_states[pooler_config.layer_select][ None, ... ]
        else:
            states = torch.stack( itemgetter( *pooler_config.layer_select )( hidden_states ) )
        
        # Pre normalise layers
        states: torch.Tensor = self.layer_norm_pre( states )
        
        # Perform layer pooling type
        match pooler_config.layer_pooling:
            case 'layer':
                states = states.squeeze( 0 )
            
            case 'mean':
                states = states.mean( 0 )
            
            case 'weighted_sum':
                assert self.layer_weighting is not None
                
                drop_mask = torch.ones(
                    states.size( 0 ),
                    states.size( 1 ),
                    1,
                    1,
                    dtype=states.dtype,
                    device=states.device
                )
                
                states = (
                    states *
                    self.layer_weighting.softmax( 0 ) *
                    self.layer_dropout( drop_mask )
                ).sum( 0 )

            case _:
                raise ValueError( 'Incorrect layer pooler type.' )
        
        # Post normalise layers
        states: torch.Tensor = self.layer_norm_post( states )
        
        
        # Get some useful stuff
        batch_size, seq_lengths = input_ids.shape[:2]
        batch_ids = torch.arange( batch_size, device=states.device )
        seq_ids = torch.arange( seq_lengths, device=input_ids.device )
        
        if prefix_type == None:
            prefix_count = 0
        else:
            prefix_count = pooler_config.prefix_sizes[prefix_type]
        
        start_idx = torch.where( input_ids == start_id, seq_ids, -1 ).max( -1 )[0] + prefix_count
        end_idx = torch.where( input_ids == end_id, seq_ids, -1 ).max( -1 )[0]
        segment_mask = ( start_idx[ :, None ] <= seq_ids[ None, : ] ) * ( seq_ids[ None, : ] <= end_idx[ :, None ] )
        segment_pos = segment_mask.float().cumsum( -1 ) * segment_mask
        
        segment_mask = segment_mask[ ..., None ]
        segment_pos = segment_pos[ ..., None ]
        
        # Pre normalise tokens
        states: torch.Tensor = self.token_norm_pre( states )
        
        # Compute gate before rotation
        if self.token_gate:
            assert self.token_gate_act is not None
            gate = self.token_gate_act( self.token_gate( states ) ).float()
        else:
            gate = None
        
        # Create rotation matrix if enabled
        rotation_weight = self.token_rotate( states.dtype ) if self.token_rotate else None
        
        # Perform rotation (if enabled)
        if rotation_weight is not None:
            states = torch.matmul( states, rotation_weight )
        
        # Cast to float
        states = states.float()
        
        # Perform layer token pooling
        match pooler_config.token_pooling:
            case 'cls':
                states = states * segment_mask
            
            case 'max':
                states = torch.where( segment_mask, states, -1e9 ).cummax( -2 )[0]
                states = states * segment_mask
            
            case 'mean':
                states = torch.where( segment_mask, states, 0 ).cumsum( -2 )
                states = states / ( segment_pos + 1e-9 )
                states = states * segment_mask
            
            case 'sgpt':
                states = torch.where( segment_mask, states * segment_pos, 0 ).cumsum( -2 )
                states = states / segment_pos.cumsum( -2 )
                states = states * segment_mask
            
            case 'ema':
                assert self.ema_beta is not None
                assert self.ema_weight is not None
                
                log_beta = F.logsigmoid( self.ema_beta + self.ema_weight ).float() # pylint: disable=E1102
                
                if gate is None:
                    states = complex_scan( segment_mask, states, log_beta, segment_pos )
                else:
                    segment_weight = segment_mask.float() * gate
                    states = complex_selective_scan( segment_mask, states, log_beta, segment_weight )
            
            case _:
                raise ValueError( 'Incorrect token pooler type.' )
        
        # Perform inverse rotation (if enabled)
        if rotation_weight is not None:
            states = torch.matmul( states, rotation_weight.T )
            
        # Post normalise tokens
        states: torch.Tensor = self.token_norm_post( states )
        
        
        if not return_all:
            return states[ batch_ids, end_idx ]
        else:
            return {
                'final_states': states[ batch_ids, end_idx ],
                'pooled_states': states,
                'segment_mask': segment_mask,
                'segment_pos': segment_pos,
            }
        
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        output_latent_states=False,
        compute_sae_loss=False
    ) -> DPHOutput:
        assert self.pooler_config
        
        if compute_sae_loss:
            assert self.pooler_config.pooler_function == 'sae'
            latent_states, sae_loss = self.pooler_pipeline( hidden_states, output_auxiliary=True )
        else:
            latent_states = self.pooler_pipeline( hidden_states )
            sae_loss = None
        
        dropped_states = self.embedding_dropout( latent_states )

        rewards = {
            name: module( dropped_states )
            for name, module
            in self.reward_heads.items()
        }
        
        return DPHOutput(
            rewards=rewards,
            latent_states=latent_states if output_latent_states else None,
            aux_loss=sae_loss,
        )


class LSWTForDPH( LSWTForCausalLM ):
    """
    Causal LM model class with DPH for the LSW Transformer.

    Inherits from LSWTForCausalLM and adds one or more DPHs.
    """

    def __init__( self, config: LSWTConfig, parent_embeddings: torch.Tensor | None=None ):
        """
        Constructs a new LSWTForDPH.

        Args:
            config (LSWTConfig): Config for the LSWT architecture
            parent_embeddings (Optional[torch.Tensor], optional): Optinal warm start embeddings.
        """

        super().__init__( config, parent_embeddings )

        self.pooler = LSWTPooler( config.pooler_config, config.d_model, config.enable_bias )

        self.post_init()
    
    def compute_final_rewards(
        self,
        hidden_states: tuple[torch.Tensor],
        input_ids: torch.LongTensor,
        start_id: int,
        end_id: int,
        prefix_type: str | None = 'assistant',
    ) -> Mapping[str, torch.Tensor]:
        dph_states = self.pooler.aggregate_states( hidden_states, input_ids, start_id, end_id, return_all=False, prefix_type=prefix_type )
        assert isinstance( dph_states, torch.Tensor )
        return self.pooler( dph_states, False, False )

    # def compute_final_rewards(
    #     self,
    #     last_hidden_states: torch.Tensor,
    #     input_ids: torch.LongTensor,
    #     cls_id: int,
    #     output_latent_states: bool = False,
    #     compute_sae_loss: bool = False
    # ) -> DPHOutput:
    #     """ Computes the final token rewards of all sequences in a batch.

    #     Args:
    #         last_hidden_states (torch.Tensor): Hidden states of size [Batch x Seq x Dim].
    #         input_ids (torch.LongTensor): Input ids of size [Batch x Seq].
    #         cls_id (int): id of the token used to aggregate rewards from. If multiple exist in the sequence the last one is used.
    #         output_latent_states (bool, optional): When true returns the intermediate latent state. Defaults to False.
    #         compute_sae_loss (bool, optional): When true computes the L1 and reconstruction losses. Only supported when in SAE mode. Defaults to False.

    #     Returns:
    #         DPHOutput: Model output
    #     """

    #     batch_size, seq_lengths = input_ids.shape[:2]
    #     batch_ids = torch.arange( batch_size, device=last_hidden_states.device )
    #     seq_ids = torch.arange( seq_lengths, device=input_ids.device )
    #     cls_idx = torch.where( input_ids == cls_id, seq_ids, -1 ).max( -1 )[0]

    #     # TODO: some sort of assertion that there at least is a CLS ID somewhere
    #     assert torch.all( cls_idx != -1 )

    #     pooled_states = last_hidden_states[ batch_ids, cls_idx ]
        
    #     return self.pooler( pooled_states, output_latent_states=output_latent_states, compute_sae_loss=compute_sae_loss )
        

    def get_param_groups(
        self,
        decay_mask: Sequence[str],
        dph_decay_mask: Sequence[str] = tuple(),
        dph_decay_init: bool = False,
        dph_weight_decay: float = 0.0,
        dph_lr_multiplier: float = 1.0,
    ) -> list[dict]:
        """
        Returns optimizer parameter groups with weight decay disabled for certain params.

        Args:
            decay_mask (Sequence[str]): list of string patterns which disable weight decay for model backbone
            dph_decay_mask (Sequence[str]): list of string patterns which disable weight decay for pooler
            dph_decay_init (bool): if prior regularization should be enabled for DPH weights. Defaults to False.
            dph_weight_decay (float): the weight/prior decay amount. Defaults to 0.0.
            dph_lr_multiplier (float): the learning rate multiplier for DPH pooler weights.

        Returns:
            List[Dict]: list of param groups to be used by the optimizer
        """

        decay_params = []
        non_decay_params = []

        dph_decay_params = []
        dph_non_decay_params = []

        # TODO: improve this
        def check_dph( p ):
            for pp in self.pooler.parameters():
                if p is pp:
                    return True

        for name, p in self.named_parameters():
            if p.requires_grad:               
                if check_dph( p ):
                    if any( i in name for i in dph_decay_mask ):
                        dph_non_decay_params.append( p )
                    else:
                        dph_decay_params.append( p )
                else:
                    if any( i in name for i in decay_mask ):
                        non_decay_params.append( p )
                    else:
                        decay_params.append( p )

        return [
            { 'params': decay_params },
            { 'params': non_decay_params, 'weight_decay': 0.0 },

            { 'params': dph_decay_params, 'decay_init': dph_decay_init, 'weight_decay': dph_weight_decay, 'lr_multiplier': dph_lr_multiplier },
            { 'params': dph_non_decay_params, 'decay_init': dph_decay_init, 'weight_decay': 0.0, 'lr_multiplier': dph_lr_multiplier },
        ]


class WrappedLSWTForDPH( LSWTForDPH ):
    def __init__( self, config: LSWTConfig, wrapped_model ):
        super().__init__( config )
        
        self.model = None
        self.head_proj = None
    
        self.wrapped_model = wrapped_model
    
    def forward( self, *args, **kwargs ):
        kwargs.pop( 'max_key_values', None )
        return self.wrapped_model( *args, **kwargs )
    
    def get_input_embeddings( self ):
        return self.wrapped_model.get_input_embeddings()