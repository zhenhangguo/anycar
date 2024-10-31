import argparse
from functools import partial
from typing import Sequence

import os
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from typing import Any, Callable, Dict, Tuple, Sequence, Union, Iterable, Optional
from enum import Enum

# from datasets import load_dataset
from flax import linen as nn
from flax.linen import partitioning as nn_partitioning
from flax.training import train_state

from jax.interpreters import pxla
from jax.sharding import PartitionSpec
from jax import lax, vmap
from flax.linen.attention import combine_masks
from jax.ad_checkpoint import checkpoint_name
from jax import random as jax_random
from reprlib import recursive_repr

import transformer_engine.jax.flax as te_flax

#from .transformormer_engine_code import *

#import transformer_engine.jax.flax as te_flax

import warnings
warnings.filterwarnings('ignore')

class JaxLearnedPositionalEncoding(nn.Module):
    d_model: int
    max_len: int
    dropout_rate: float
    flip: bool = False
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x, deterministic=False):
        pos_enc = self.param('pos_enc', lambda key, shape: jax.random.uniform(key, shape, self.dtype),
                             (self.max_len, self.d_model))
        if self.flip:
            x = x + jnp.flip(pos_enc, axis=0)
        else:
            x = x + pos_enc

        x = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(x)
        return x
   
class JaxTransformerEncoder(nn.Module):
    state_dim: int
    action_dim: int
    output_dim: int
    latent_dim: int
    num_heads: int
    num_layers: int
    dropout_rate: float
    history_length: int
    action_length: int
    dtype: jnp.dtype = jnp.float32
    # attn_mask_type: str = 'causal'

    @nn.compact
    def __call__(self, history, action, history_padding_mask=None, action_padding_mask=None, tgt_mask=None, deterministic=False):
        action_encoder = nn.DenseGeneral(self.latent_dim, dtype=self.dtype, name='linear_input_action')
        history_emb = jnp.zeros((history.shape[0], history.shape[1] * 2 - 1, self.latent_dim), dtype=self.dtype)
        history_emb = history_emb.at[:, ::2].set(nn.DenseGeneral(self.latent_dim, dtype=self.dtype, name='linear_input_state')(history[:, :, :self.state_dim])) # shape: [batch_size, seq_length, latent_dim]
        history_emb = history_emb.at[:, 1::2].set(action_encoder(history[:, :-1, self.state_dim:self.state_dim+self.action_dim])) # shape: [batch_size, seq_length-1, latent_dim]
        
        history_emb = JaxLearnedPositionalEncoding(self.latent_dim, self.history_length * 2 - 1, self.dropout_rate, flip=True, dtype=self.dtype)(history_emb, deterministic=deterministic)

        action_emb = action_encoder(action)
        action_emb = JaxLearnedPositionalEncoding(self.latent_dim, self.action_length, self.dropout_rate, dtype=self.dtype)(action_emb, deterministic=deterministic)

        decoder_output = action_emb
        for i in range(self.num_layers):
            te_Decoder = partial(
                te_flax.TransformerLayer,
                hidden_size=self.latent_dim,
                mlp_hidden_size=self.latent_dim * 4,
                num_attention_heads=self.num_heads,
                hidden_dropout=self.dropout_rate,
                attention_dropout=self.dropout_rate,
                intermediate_dropout=self.dropout_rate,
                dropout_rng_name='dropout',
                mlp_activations=('gelu',),
                layer_type=TransformerLayerType.ENCODER,
                self_attn_mask_type="causal",
                enable_relative_embedding=False,
                dtype=self.dtype,
                name=f'transformer_layer_{i}',
            )
            decoder_output = te_Decoder()(inputs=decoder_output, attention_mask=None, encoder_decoder_mask=tgt_mask, encoded=history_emb, deterministic=deterministic)

        output = nn.DenseGeneral(self.output_dim, dtype=self.dtype, name='linear_output')(decoder_output)
        return output
    
     
class JaxTransformerDecoder(nn.Module):
    state_dim: int
    action_dim: int
    output_dim: int
    latent_dim: int
    num_heads: int
    num_layers: int
    dropout_rate: float
    history_length: int
    action_length: int
    dtype: jnp.dtype = jnp.float32
    # attn_mask_type: str = 'causal'

    @nn.compact
    def __call__(self, history, action, history_padding_mask=None, action_padding_mask=None, tgt_mask=None, deterministic=False):
        action_encoder = nn.DenseGeneral(self.latent_dim, dtype=self.dtype, name='linear_input_action')
        history_emb = jnp.zeros((history.shape[0], history.shape[1] * 2 - 1, self.latent_dim), dtype=self.dtype)
        history_emb = history_emb.at[:, ::2].set(nn.DenseGeneral(self.latent_dim, dtype=self.dtype, name='linear_input_state')(history[:, :, :self.state_dim])) # shape: [batch_size, seq_length, latent_dim]
        history_emb = history_emb.at[:, 1::2].set(action_encoder(history[:, :-1, self.state_dim:self.state_dim+self.action_dim])) # shape: [batch_size, seq_length-1, latent_dim]
        
        history_emb = JaxLearnedPositionalEncoding(self.latent_dim, self.history_length * 2 - 1, self.dropout_rate, flip=True, dtype=self.dtype)(history_emb, deterministic=deterministic)

        action_emb = action_encoder(action)
        action_emb = JaxLearnedPositionalEncoding(self.latent_dim, self.action_length, self.dropout_rate, dtype=self.dtype)(action_emb, deterministic=deterministic)

        decoder_output = action_emb
        for i in range(self.num_layers):
            te_Decoder = partial(
                te_flax.TransformerLayer,
                hidden_size=self.latent_dim,
                mlp_hidden_size=self.latent_dim * 4,
                num_attention_heads=self.num_heads,
                hidden_dropout=self.dropout_rate,
                attention_dropout=self.dropout_rate,
                intermediate_dropout=self.dropout_rate,
                dropout_rng_name='dropout',
                mlp_activations=('gelu',),
                layer_type=te_flax.TransformerLayerType.DECODER,
                self_attn_mask_type="causal",
                enable_relative_embedding=False,
                dtype=self.dtype,
                name=f'transformer_layer_{i}',
            )
            decoder_output = te_Decoder()(inputs=decoder_output, attention_mask=None, encoder_decoder_mask=tgt_mask, encoded=history_emb, deterministic=deterministic)

        output = nn.DenseGeneral(self.output_dim, dtype=self.dtype, name='linear_output')(decoder_output)
        return output
    
class JaxTransformerDecoderVis(nn.Module):
    state_dim: int
    action_dim: int
    output_dim: int
    latent_dim: int
    num_heads: int
    num_layers: int
    dropout_rate: float
    history_length: int
    action_length: int
    dtype: jnp.dtype = jnp.float32
    # attn_mask_type: str = 'causal'

    @nn.compact
    def __call__(self, history, action, history_padding_mask=None, action_padding_mask=None, tgt_mask=None, deterministic=False):
        action_encoder = nn.DenseGeneral(self.latent_dim, dtype=self.dtype, name='linear_input_action')
        history_emb = jnp.zeros((history.shape[0], history.shape[1] * 2 - 1, self.latent_dim), dtype=self.dtype)
        history_emb = history_emb.at[:, ::2].set(nn.DenseGeneral(self.latent_dim, dtype=self.dtype, name='linear_input_state')(history[:, :, :self.state_dim])) # shape: [batch_size, seq_length, latent_dim]
        history_emb = history_emb.at[:, 1::2].set(action_encoder(history[:, :-1, self.state_dim:self.state_dim+self.action_dim])) # shape: [batch_size, seq_length-1, latent_dim]
        
        history_emb = JaxLearnedPositionalEncoding(self.latent_dim, self.history_length * 2 - 1, self.dropout_rate, flip=True, dtype=self.dtype)(history_emb, deterministic=deterministic)

        action_emb = action_encoder(action)
        action_emb = JaxLearnedPositionalEncoding(self.latent_dim, self.action_length, self.dropout_rate, dtype=self.dtype)(action_emb, deterministic=deterministic)

        decoder_output = action_emb
        all_attention_weights = []
        for i in range(self.num_layers):
            te_Decoder = partial(
                te_flax.TransformerLayer,
                hidden_size=self.latent_dim,
                mlp_hidden_size=self.latent_dim * 4,
                num_attention_heads=self.num_heads,
                hidden_dropout=self.dropout_rate,
                attention_dropout=self.dropout_rate,
                intermediate_dropout=self.dropout_rate,
                dropout_rng_name='dropout',
                mlp_activations=('gelu',),
                layer_type=te_flax.TransformerLayerType.DECODER,
                self_attn_mask_type="causal",
                enable_relative_embedding=False,
                dtype=self.dtype,
                name=f'transformer_layer_{i}',
            )
            decoder_output, attn_weights = te_Decoder()(inputs=decoder_output, attention_mask=None, encoder_decoder_mask=tgt_mask, encoded=history_emb, deterministic=deterministic)
            all_attention_weights.append(attn_weights)
        mean_attention_weights = jnp.mean(jnp.stack(all_attention_weights), axis=0)

        output = nn.DenseGeneral(self.output_dim, dtype=self.dtype, name='linear_output')(decoder_output)
        return output, mean_attention_weights
    
class JaxMLP(nn.Module):
    hidden_sizes: Sequence[int]
    output_dim: int
    dropout_rate: float

    @nn.compact
    def __call__(self, history, action, history_padding_mask=None, action_padding_mask=None, tgt_mask=None, deterministic=False):
        bs = history.shape[0]
        x = jnp.concatenate([history.reshape(bs, -1), action.reshape(bs, -1)], axis=1)
        for i, hidden_size in enumerate(self.hidden_sizes):
            x = nn.Dense(hidden_size)(x)
            x = nn.relu(x)
            x = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(x)
        x = nn.Dense(self.output_dim * action.shape[1])(x)
        return x.reshape((x.shape[0], -1, self.output_dim))
    
class JaxCNN(nn.Module):
    conv_layers: Sequence[int]  # List of filter sizes for convolutional layers
    kernel_sizes: Sequence[int]  # Corresponding kernel sizes
    pool_sizes: Sequence[int]  # Corresponding pool sizes
    output_dim: int
    dropout_rate: float

    @nn.compact
    def __call__(self, history, action, history_padding_mask=None, action_padding_mask=None, tgt_mask=None, deterministic=False):
        bs = history.shape[0]
        x = jnp.concatenate([history.reshape(bs, -1), action.reshape(bs, -1)], axis=1)  # Concatenate history and action
        x = x[:, :, None]
        # Apply the convolutional layers
        for i, (filters, kernel_size, pool_size) in enumerate(zip(self.conv_layers[:-1], self.kernel_sizes[:-1], self.pool_sizes[:-1])):
            x = nn.Conv(features=filters, kernel_size=(kernel_size,), padding='SAME')(x)
            x = nn.relu(x)
            x = nn.max_pool(x, window_shape=(pool_size, ), strides=(pool_size, ), padding='SAME')
            x = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(x)

        x = x.reshape((bs, -1))
        x = nn.Dense(self.output_dim * action.shape[1])(x)

        return x.reshape((x.shape[0], -1, self.output_dim))


class JaxGRU(nn.Module):
    hidden_size: int
    output_dim: int
    num_layers: int
    dropout_rate: float

    @nn.compact
    def __call__(self, history, action, history_padding_mask=None, action_padding_mask=None, tgt_mask=None, deterministic=False):
        bs = history.shape[0]
        
        actions = action

        x = history.reshape(bs, -1)

        # Initialize GRU layers
        gru_cell = nn.GRUCell(features = self.hidden_size, name=f'gru_layer')
        
        # Initialize hidden state to zeros
        hidden_state = nn.Dense(self.hidden_size)(x)
        hidden_state = nn.relu(hidden_state)

        # Apply GRU layers iteratively
        outputs = []
        for t in range(actions.shape[1]):
            hidden_state, output = gru_cell(hidden_state, actions[:, t, :])
            outputs.append(output)
        
        x = jnp.stack(outputs, axis=1)  # Stack the outputs along the time dimension

        x = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(x)

        x = nn.Dense(self.output_dim)(x)
        
        return x

class JaxLSTM(nn.Module):
    hidden_size: int
    output_dim: int
    dropout_rate: float

    @nn.compact
    def __call__(self,  history, action, history_padding_mask=None, action_padding_mask=None, tgt_mask=None, deterministic=False):
        bs = history.shape[0]
        # Initialize the LSTM cell
        lstm_cell = nn.LSTMCell(features=self.hidden_size)
        
        # Initialize carry (hidden and cell states)
        carry = lstm_cell.initialize_carry(jax.random.PRNGKey(0), (action.shape[0],))

        carry = (nn.relu(nn.Dense(self.hidden_size)(history.reshape(bs, -1))), nn.relu(nn.Dense(self.hidden_size)(history.reshape(bs, -1))))
        
        outputs = []
        
        # Iterate through each time step
        for t in range(action.shape[1]):
            # Update carry and compute the output for each time step
            carry, output = lstm_cell(carry, action[:, t, :])
            outputs.append(output)
        
        # Stack the outputs along the time axis
        x = jnp.stack(outputs, axis=1)

        x = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(x)
        
        # Apply a final dense layer to map to output dimension
        x = nn.Dense(self.output_dim)(x)
        
        return x
    