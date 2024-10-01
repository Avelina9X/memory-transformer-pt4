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

from .layers_pooling import LSWTLayerPoolerSingle, LSWTLayerPoolerWeighted, LSWTTokenPoolerAttention, LSWTTokenPoolerCLS

from .configuration import LSWTConfig, LSWTPoolerConfig
from .layers import SharedEmbeddings, RotaryEmbedding, LSWTBlock, ActGLU, prolu_ste, prolu_relu

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

        # elif isinstance( module, LSWTSparseAutoEncoder ):
        #     torch.nn.init.kaiming_uniform_( module.decoder_weight.data )
        #     torch.nn.init.kaiming_uniform_( module.encoder_weight.data )
        #     module.encoder_bias.data.zero_()
        #     module.decoder_bias.data.zero_()

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
class DPHOutput( ModelOutput ):
    """ Base class for DPH reward outputs.
    """

    rewards: Mapping[str, torch.Tensor]
    """ Mapping of head name to the rewards computed on the <|im_end|> token."""
    
    embeddings: torch.Tensor | None = None
    """ Latent states computed on the <|im_end|> token. Returns None when `output_embeddings=False` """

class LSWTPooler( torch.nn.Module ):
    def __init__( self, pooler_config: LSWTPoolerConfig, base_config: LSWTConfig ):
        super().__init__()

        self.pooler_config = pooler_config
        self.base_config = base_config

        if pooler_config.reward_heads is None:
            raise ValueError( 'reward_heads must be defined. If no heads are desired please use an empty list.' )
        
        self.layer_norm_post = torch.nn.LayerNorm( base_config.d_model ) if pooler_config.layer_pooling_norm in [ 'post' ] else torch.nn.Identity()
        self.token_norm_post = torch.nn.LayerNorm( base_config.d_model ) if pooler_config.layer_pooling_norm in [ 'post' ] else torch.nn.Identity()
        self.embedding_dropout = torch.nn.Dropout( pooler_config.embedding_dropout )

        match pooler_config.layer_pooling:
            case 'layer':
                self.layer_pooler = LSWTLayerPoolerSingle( pooler_config )
            case 'weighted_sum':
                self.layer_pooler = LSWTLayerPoolerWeighted( pooler_config )
            case _:
                raise ValueError( f'`{pooler_config.layer_pooling}` is not a valid value for pooler_config.layer_pooling' )
        
        match pooler_config.token_pooling:
            case 'cls':
                self.token_pooler = LSWTTokenPoolerCLS( pooler_config )
            case 'attn':
                self.token_pooler = LSWTTokenPoolerAttention( pooler_config, base_config )
            case _:
                raise ValueError( f'`{pooler_config.token_pooling}` is not a valid value for pooler_config.token_pooling' )

        # Create dict of reward head projections
        self.reward_heads = torch.nn.ModuleDict( {
            name: torch.nn.Linear( base_config.d_model, 1, bias=pooler_config.reward_head_bias )
            for name in pooler_config.reward_heads
        } )


    def forward(
        self,
        hidden_states: tuple[torch.Tensor],
        input_ids: torch.Tensor,
        output_embeddings=False,
        return_final=True,
    ) -> DPHOutput:
        assert self.pooler_config
        
        # Perform layer pooling and normalise
        layer_states: torch.Tensor = self.layer_pooler( hidden_states )
        layer_states: torch.Tensor = self.layer_norm_post( layer_states )
        
        # Perform token pooling and normalise
        embeddings: torch.Tensor = self.token_pooler( layer_states, input_ids, return_final )
        embeddings: torch.Tensor = self.token_norm_post( embeddings )

        # Perform dropout for the heads
        dropped_states: torch.Tensor = self.embedding_dropout( embeddings )

        # Compute rewards
        rewards = {
            name: module( dropped_states )
            for name, module
            in self.reward_heads.items()
        }

        return DPHOutput(
            rewards=rewards,
            embeddings=embeddings if output_embeddings else None,
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

        self.pooler = LSWTPooler( config.pooler_config, config )

        self.post_init()

    def compute_final_rewards(
        self,
        hidden_states: tuple[torch.Tensor],
        input_ids: torch.Tensor,
    ) -> Mapping[str, torch.Tensor]:
        return self.pooler( hidden_states, input_ids, output_embeddings=False, return_final=True ).rewards

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
    
    def parameters_split( self, pooler: bool ):
        def check_dph( p ):
            for pp in self.pooler.parameters():
                if p is pp:
                    return True
        
        return [ p for p in self.parameters() if p.requires_grad and ( check_dph( p ) == pooler ) ]


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
