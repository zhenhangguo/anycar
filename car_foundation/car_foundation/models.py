import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from jax import grad, jit, vmap
from functools import partial
from car_planner.fast_spline_trajectory_generation import interpolate_action_sequence
from car_foundation.utils import generate_subsequences_hf
from transformers import GPT2Config, GPT2Model, FlaxGPT2Model, modeling_flax_pytorch_utils
from transformers.models.gpt2.modeling_flax_gpt2 import FlaxGPT2Module
from flax import linen

from typing import Sequence, Optional
import os
import time
import math
import tqdm
import matplotlib.pyplot as plt

import torch
from torch import nn, Tensor

from collections import OrderedDict

class MLP(linen.Module):
    features: Sequence[int]

    @linen.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = linen.relu(linen.Dense(feat)(x))
        return linen.Dense(self.features[-1])(x)
    
class TorchMLP(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, output_size)
        )
    
    def forward(self, x):
        return self.layers(x)
    
class TorchTransformerEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, action_dim, embed_dim, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.output_dim = output_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim

        self.input_dim = output_dim + action_dim
        self.embedding = nn.Linear(self.input_dim, embed_dim)
        
        self.pos_encoder = nn.Parameter(torch.zeros(input_dim, embed_dim))
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True),
            num_layers=num_layers
        )
        self.output_layer = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)  # x shape: [batch_size, seq_length, input_dim]
        x += self.pos_encoder[:x.size(1)]  # add positional encoding
        x = self.transformer_encoder(x)  # pass through transformer
        x = self.output_layer(x[:, -1, :])  # take the last timestep output and predict next state
        return x
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
    
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, flip=False):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(max_len, d_model))
        self.dropout = nn.Dropout(0.1)
        self.flip = flip

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        if self.flip:
            x = x + torch.flip(self.pe[:x.size(1)], [0])
        else:
            x = x + self.pe[:x.size(1)]

        return self.dropout(x)

class TorchTransformerDecoder(nn.Module):
    def __init__(self, state_dim, action_dim, output_dim, latent_dim, num_heads, 
                num_layers, device, dropout=0.1, history_length=250, prediction_length=50, compressed_history_length = 42):
        super().__init__()
        # 初始化维度参数
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.history_length = history_length
        self.prediction_length = prediction_length
        self.compressed_history_length = compressed_history_length

        self.embedding = nn.ModuleDict({
            'state': nn.Linear(state_dim, latent_dim),
            'action': nn.Linear(action_dim, latent_dim),
            'output': nn.Linear(latent_dim, output_dim)
        }).to(device)

        # compress output size = 250->42
        self.compressor = nn.ModuleDict({
            'state': nn.Sequential(
                nn.Conv1d(state_dim, state_dim, kernel_size=5, stride=3, padding=2),   # if history input is 125, padding should be 1 to keep the same size with action
                nn.ReLU(),
                nn.Conv1d(state_dim, state_dim, kernel_size=3, stride=2, padding=1),
            ).to(device),
            'action': nn.Sequential(
                nn.Conv1d(action_dim, action_dim, kernel_size=5, stride=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(action_dim, action_dim, kernel_size=3, stride=2, padding=1),
            ).to(device)
        })

        self.register_buffer('tgt_mask', 
                           nn.Transformer.generate_square_subsequent_mask(prediction_length).to(device))
        
        # 位置编码
        self.position_encoding = nn.ModuleDict({
            'history': LearnedPositionalEncoding(latent_dim, compressed_history_length * 2 - 1, flip=True).to(device) ,
            'action': LearnedPositionalEncoding(latent_dim, prediction_length).to(device) 
        })
        
        # Transformer 结构
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=latent_dim,
                nhead=num_heads,
                dim_feedforward=512,
                dropout=dropout,
                batch_first=True,
                device=device
            ), 
            num_layers=num_layers
        ).to(device)

        self.register_buffer('dummy', torch.tensor(0, device=device), persistent=False)
        self.to(device)

    def _build_history_emb(self, history: torch.Tensor) -> torch.Tensor:
        """向量化的历史序列构建"""
        if history.device != self.device:
            history = history.to(self.device, non_blocking=True)

        # 优化内存布局，减少transpose操作
        batch_size = history.size(0)
        state = history[..., :self.state_dim].permute(0, 2, 1).contiguous()
        action = history[..., self.state_dim:].permute(0, 2, 1).contiguous()

        with torch.cuda.stream(torch.cuda.current_stream()):  # 使用单独的CUDA流

            # 直接在GPU上进行卷积计算
            state_compressed = self.compressor['state'](state.cuda())
            action_compressed = self.compressor['action'](action.cuda())
        
            # 优化内存布局转换
            state_emb = self.embedding['state'](state_compressed.transpose(1, 2))
            action_emb = self.embedding['action'](action_compressed.transpose(1, 2))

            interleaved = torch.stack([state_emb, action_emb], dim=2)
            interleaved = interleaved.view(interleaved.size(0), -1, interleaved.size(-1))
            return interleaved[:, :-1, :]

    def forward(self, history, action, history_padding_mask=None, action_padding_mask=None):
        if not self.training:  
            history_first_batch = history[0:1,:,:].contiguous()
            history_emb_first_batch = self._build_history_emb(history_first_batch)
            history_emb_first_batch  = self.position_encoding['history'](history_emb_first_batch)
            history_emb = history_emb_first_batch.repeat(history.size(0), 1, 1)
        else:
            history_emb = self._build_history_emb(history)
            history_emb = self.position_encoding['history'](history_emb)
        
        action_emb = self.position_encoding['action'](
            self.embedding['action'](action)
        )
        
        out = self.transformer_decoder(
            tgt=action_emb,
            memory=history_emb,
            tgt_mask=self.tgt_mask,
            tgt_key_padding_mask=action_padding_mask.to(self.device) if action_padding_mask is not None else None,
            memory_key_padding_mask=history_padding_mask.to(self.device) if history_padding_mask is not None else None,
        )
        return self.embedding['output'](out)

class TorchTransformerDecoderCurrentState(TorchTransformerDecoder):
    def __init__(self, state_dim, action_dim, output_dim, latent_dim, num_heads,
                num_layers, device, dropout=0.1, history_length=250, prediction_length=50,
                compressed_history_length=42, current_dim=None):
        super().__init__(
            state_dim,
            action_dim,
            output_dim,
            latent_dim,
            num_heads,
            num_layers,
            device,
            dropout,
            history_length,
            prediction_length,
            compressed_history_length,
        )
        self.current_dim = current_dim if current_dim is not None else state_dim + action_dim
        self.action_fusion = nn.Linear(action_dim + self.current_dim, latent_dim).to(device)

    def init_fusion_from_action_embedding(self):
        with torch.no_grad():
            self.action_fusion.weight.zero_()
            self.action_fusion.bias.copy_(self.embedding['action'].bias)
            self.action_fusion.weight[:, :self.action_dim].copy_(self.embedding['action'].weight)

    def _build_action_emb(self, history, action, current_state=None):
        if current_state is None:
            current_state = history[:, -1, :self.current_dim]
        current_state = current_state.to(action.device)
        current = current_state[:, None, :].expand(-1, action.shape[1], -1)
        return self.action_fusion(torch.cat([action, current], dim=-1))

    def forward(self, history, action, current_state=None, history_padding_mask=None, action_padding_mask=None):
        if not self.training:
            history_first_batch = history[0:1, :, :].contiguous()
            history_emb_first_batch = self._build_history_emb(history_first_batch)
            history_emb_first_batch = self.position_encoding['history'](history_emb_first_batch)
            history_emb = history_emb_first_batch.repeat(history.size(0), 1, 1)
        else:
            history_emb = self._build_history_emb(history)
            history_emb = self.position_encoding['history'](history_emb)

        action_emb = self.position_encoding['action'](
            self._build_action_emb(history, action, current_state)
        )

        out = self.transformer_decoder(
            tgt=action_emb,
            memory=history_emb,
            tgt_mask=self.tgt_mask,
            tgt_key_padding_mask=action_padding_mask.to(self.device) if action_padding_mask is not None else None,
            memory_key_padding_mask=history_padding_mask.to(self.device) if history_padding_mask is not None else None,
        )
        return self.embedding['output'](out)

class TorchTransformerDecoderCurrentStateMLP(TorchTransformerDecoderCurrentState):
    def __init__(self, state_dim, action_dim, output_dim, latent_dim, num_heads,
                num_layers, device, dropout=0.1, history_length=250, prediction_length=50,
                compressed_history_length=42, current_dim=None, fusion_hidden_dim=128):
        super().__init__(
            state_dim,
            action_dim,
            output_dim,
            latent_dim,
            num_heads,
            num_layers,
            device,
            dropout,
            history_length,
            prediction_length,
            compressed_history_length,
            current_dim,
        )
        self.fusion_hidden_dim = fusion_hidden_dim
        self.action_fusion = nn.Sequential(
            nn.Linear(action_dim + self.current_dim, fusion_hidden_dim),
            nn.SiLU(),
            nn.Linear(fusion_hidden_dim, latent_dim),
        ).to(device)
        self.init_fusion_from_action_embedding()

    def init_fusion_from_action_embedding(self):
        """Start as the baseline action embedding and learn a residual correction."""
        with torch.no_grad():
            nn.init.zeros_(self.action_fusion[-1].weight)
            nn.init.zeros_(self.action_fusion[-1].bias)

    def _build_action_emb(self, history, action, current_state=None):
        if current_state is None:
            current_state = history[:, -1, :self.current_dim]
        current_state = current_state.to(action.device)
        current = current_state[:, None, :].expand(-1, action.shape[1], -1)
        fusion_input = torch.cat([action, current], dim=-1)
        return self.embedding['action'](action) + self.action_fusion(fusion_input)


class TorchTransformerDecoderKinematicQueryMLP(TorchTransformerDecoderCurrentStateMLP):
    """Add a zero-initialized DyTR-style nominal kinematic query branch.

    The existing action/current-state query remains unchanged.  Nominal future
    states and transitions are fused through an additive MLP whose final layer
    starts at zero, so loading a ``TorchTransformerDecoderCurrentStateMLP``
    checkpoint preserves its predictions exactly at initialization.
    """

    def __init__(self, state_dim, action_dim, output_dim, latent_dim, num_heads,
                 num_layers, device, dropout=0.1, history_length=250,
                 prediction_length=50, compressed_history_length=42,
                 current_dim=None, fusion_hidden_dim=128,
                 nominal_state_dim=5, nominal_transition_dim=4,
                 query_hidden_dim=128):
        super().__init__(
            state_dim,
            action_dim,
            output_dim,
            latent_dim,
            num_heads,
            num_layers,
            device,
            dropout,
            history_length,
            prediction_length,
            compressed_history_length,
            current_dim,
            fusion_hidden_dim,
        )
        self.nominal_state_dim = nominal_state_dim
        self.nominal_transition_dim = nominal_transition_dim
        self.query_hidden_dim = query_hidden_dim
        query_input_dim = (
            action_dim
            + self.current_dim
            + nominal_state_dim
            + nominal_transition_dim
        )
        self.kinematic_query_fusion = nn.Sequential(
            nn.Linear(query_input_dim, query_hidden_dim),
            nn.SiLU(),
            nn.Linear(query_hidden_dim, latent_dim),
        ).to(device)
        with torch.no_grad():
            nn.init.zeros_(self.kinematic_query_fusion[-1].weight)
            nn.init.zeros_(self.kinematic_query_fusion[-1].bias)

    def _build_action_emb(
        self,
        history,
        action,
        current_state=None,
        nominal_state=None,
        nominal_transition=None,
    ):
        base_embedding = super()._build_action_emb(history, action, current_state)
        if nominal_state is None and nominal_transition is None:
            return base_embedding
        if nominal_state is None or nominal_transition is None:
            raise ValueError(
                "nominal_state and nominal_transition must be provided together"
            )
        if nominal_state.shape[:-1] != action.shape[:-1]:
            raise ValueError(
                "nominal_state leading dimensions must match future action"
            )
        if nominal_transition.shape[:-1] != action.shape[:-1]:
            raise ValueError(
                "nominal_transition leading dimensions must match future action"
            )
        if nominal_state.shape[-1] != self.nominal_state_dim:
            raise ValueError(
                f"Expected nominal_state dim {self.nominal_state_dim}, "
                f"got {nominal_state.shape[-1]}"
            )
        if nominal_transition.shape[-1] != self.nominal_transition_dim:
            raise ValueError(
                f"Expected nominal_transition dim {self.nominal_transition_dim}, "
                f"got {nominal_transition.shape[-1]}"
            )

        if current_state is None:
            current_state = history[:, -1, :self.current_dim]
        current = current_state.to(action.device)[:, None, :].expand(
            -1, action.shape[1], -1
        )
        query_input = torch.cat(
            (
                action,
                current,
                nominal_state.to(action.device),
                nominal_transition.to(action.device),
            ),
            dim=-1,
        )
        return base_embedding + self.kinematic_query_fusion(query_input)

    def forward(
        self,
        history,
        action,
        current_state=None,
        nominal_state=None,
        nominal_transition=None,
        history_padding_mask=None,
        action_padding_mask=None,
    ):
        if not self.training:
            history_first_batch = history[0:1, :, :].contiguous()
            history_emb_first_batch = self._build_history_emb(history_first_batch)
            history_emb_first_batch = self.position_encoding['history'](
                history_emb_first_batch
            )
            history_emb = history_emb_first_batch.repeat(history.size(0), 1, 1)
        else:
            history_emb = self._build_history_emb(history)
            history_emb = self.position_encoding['history'](history_emb)

        action_emb = self.position_encoding['action'](
            self._build_action_emb(
                history,
                action,
                current_state,
                nominal_state,
                nominal_transition,
            )
        )
        out = self.transformer_decoder(
            tgt=action_emb,
            memory=history_emb,
            tgt_mask=self.tgt_mask,
            tgt_key_padding_mask=(
                action_padding_mask.to(self.device)
                if action_padding_mask is not None
                else None
            ),
            memory_key_padding_mask=(
                history_padding_mask.to(self.device)
                if history_padding_mask is not None
                else None
            ),
        )
        return self.embedding['output'](out)

class TorchTransformer(nn.Module):
    def __init__(self, history_dim, action_dim, output_dim, latent_dim, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.history_dim = history_dim
        self.action_dim = action_dim
        self.output_dim = output_dim

        self.history_embedding = nn.Linear(history_dim, latent_dim)
        self.action_embedding = nn.Linear(action_dim, latent_dim)
        self.output_embedding = nn.Linear(latent_dim, self.output_dim)
        
        self.pos_emb = PositionalEncoding(latent_dim)
        self.transformer = nn.Transformer(
            d_model=latent_dim, nhead=num_heads, num_encoder_layers=num_layers, num_decoder_layers=num_layers,
            dim_feedforward=128, dropout=dropout, batch_first=True
        )

    def forward(self, history, action, history_padding_mask=None, action_padding_mask=None):
        history_emb = self.history_embedding(history)
        history_emb = self.pos_emb(history_emb)
        action_emb = self.action_embedding(action)
        action_emb = self.pos_emb(action_emb)

        x = self.transformer(history_emb, action_emb,
                             src_is_causal=True, tgt_is_causal=True, memory_is_causal=True,
                             src_mask = nn.Transformer.generate_square_subsequent_mask(history_emb.size(1), device=history_emb.device),
                             tgt_mask = nn.Transformer.generate_square_subsequent_mask(action_emb.size(1), device=action_emb.device),
                             memory_mask = nn.Transformer.generate_square_subsequent_mask(history_emb.size(1), device=history_emb.device),
                             src_key_padding_mask=history_padding_mask,
                             tgt_key_padding_mask=action_padding_mask,
                             memory_key_padding_mask=history_padding_mask)
        x = self.output_embedding(x)
        return x
    

class TorchGPT2(GPT2Model):
    def __init__(self, state_dim, action_dim, output_dim, latent_dim, num_heads, num_layers, dropout=0.1):
        config = GPT2Config(
            vocab_size=state_dim + action_dim,
            n_positions=550,
            n_embd=latent_dim,
            n_layer=num_layers,
            n_head=num_heads,
            n_inner=4 * latent_dim,
            activation_function='gelu_new',
            resid_pdrop = dropout,
            embd_pdrop = dropout,
            attn_pdrop = dropout,
            use_cache = False, # setting to true will be useful later for inference
        )
        super().__init__(config)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim

        self.state_embedding = nn.Linear(state_dim, latent_dim)
        self.action_embedding = nn.Linear(action_dim, latent_dim)
        self.output_embedding = nn.Linear(latent_dim, self.output_dim)
        self.seperator_token = nn.Parameter(torch.randn(1, 1, latent_dim))

    def forward(self, history, action, history_padding_mask=None, action_padding_mask=None):
        history_emb = torch.zeros(history.size(0), history.size(1) * 2 - 1, self.latent_dim, device=history.device)
        history_emb[:, ::2] = self.state_embedding(history[:, :, :self.state_dim]) # shape: [batch_size, seq_length, latent_dim]
        history_emb[:, 1::2] = self.action_embedding(history[:, :-1, self.state_dim:self.state_dim+self.action_dim]) # shape: [batch_size, seq_length-1, latent_dim]
        action_emb = self.action_embedding(action)
        seperator_token = self.seperator_token.expand(action_emb.size(0), 1, self.latent_dim)
        inputs_embeds = torch.cat([history_emb, seperator_token, action_emb], dim=1)
        if history_padding_mask is None:
            history_padding_mask = torch.ones(history_emb.size(0), history_emb.size(1), dtype=torch.float32, device=history_emb.device)
        if action_padding_mask is None:
            action_padding_mask = torch.ones(action_emb.size(0), action_emb.size(1), dtype=torch.float32, device=action_emb.device)
        seperator_token_mask = torch.ones(action_emb.size(0), 1, dtype=torch.float32, device=action_emb.device)
        attn_mask = torch.cat([history_padding_mask, seperator_token_mask, action_padding_mask], dim=1)

        x = super().forward(inputs_embeds=inputs_embeds, attention_mask=attn_mask)
        x = self.output_embedding(x.last_hidden_state[:, -action.size(1):, :])
        return x
    
class CustomFlaxGPT2Module(FlaxGPT2Module):
    state_dim: int = 6
    action_dim: int = 2
    output_dim: int = 6
    latent_dim: int = 128
    num_heads: int = 4
    num_layers: int = 3
    dropout_percent: float = 0.1

    def setup(self):
        FlaxGPT2Module.setup(self)
        self.state_embedding = linen.Dense(self.latent_dim, name='state_embedding')
        self.action_embedding = linen.Dense(self.latent_dim, name='action_embedding')
        self.output_embedding = linen.Dense(self.output_dim, name='output_embedding')
        self.seperator_token = self.param('seperator_token', linen.initializers.xavier_uniform(), (1, 1, self.latent_dim))

    def forward(self, history, action, history_padding_mask=None, action_padding_mask=None, **kwargs):
        history_emb = jnp.zeros((history.shape[0], history.shape[1] * 2 - 1, self.latent_dim))
        history_emb = history_emb.at[:, ::2].set(self.state_embedding(history[:, :, :self.state_dim])) # shape: [batch_size, seq_length, latent_dim]
        history_emb = history_emb.at[:, 1::2].set(self.action_embedding(history[:, :-1, self.state_dim:self.state_dim+self.action_dim])) # shape: [batch_size, seq_length-1, latent_dim]
        action_emb = self.action_embedding(action)
        seperator_token = jnp.repeat(self.seperator_token, action_emb.shape[0], axis=0)
        inputs_embeds = jnp.concatenate([history_emb, seperator_token, action_emb], axis=1)
        if history_padding_mask is None:
            history_padding_mask = jnp.ones((history_emb.shape[0], history_emb.shape[1]))
        if action_padding_mask is None:
            action_padding_mask = jnp.ones((action_emb.shape[0], action_emb.shape[1]))
        seperator_token_mask = jnp.ones((action_emb.shape[0], 1))
        attn_mask = jnp.concatenate([history_padding_mask, seperator_token_mask, action_padding_mask], axis=1)

        x = FlaxGPT2Module.__call__(self, inputs_embeds=inputs_embeds, attention_mask=attn_mask, **kwargs)
        x = self.output_embedding(x.last_hidden_state[:, -action.shape[1]:, :])
        return x
    
    def load_from_pytorch(self, pt_model: TorchGPT2, batch_size: int, history_length: int, action_length: int, flax_gpt2):
        # prepare dummy jax inputs and pass through the model
        dummy_history = jnp.zeros((batch_size, history_length, self.state_dim + self.action_dim))
        dummy_action = jnp.zeros((batch_size, action_length, self.action_dim))
        dummy_history_padding_mask = jnp.ones((batch_size, history_length))
        dummy_action_padding_mask = jnp.ones((batch_size, action_length))
        # params = self.init(key, dummy_history, dummy_action, dummy_history_padding_mask, dummy_action_padding_mask)

        # for name, module in pt_gpt2.named_modules():
        #     # for the linear layers, copy the weights and biases
        #     if isinstance(module, nn.Linear):
        #         kernel = module.weight.detach().cpu().numpy().T
        #         bias = module.bias.detach().cpu().numpy()
        #         params = params.set(f'{name}.kernel', kernel)
        #         params = params.set(f'{name}.bias', bias)
        #     # for gpt-2, use the huggingface conversion function

        params = modeling_flax_pytorch_utils.convert_pytorch_state_dict_to_flax(pt_model.state_dict(), flax_gpt2)
        params = {'params': params}

        out = self.apply(params, dummy_history, dummy_action, dummy_history_padding_mask, dummy_action_padding_mask, method=CustomFlaxGPT2Module.forward)
        return params
    
class FlaxGPT2(FlaxGPT2Model):
    module_class = CustomFlaxGPT2Module     
    
def main():
    # model = ParamTest()
    # rng = random.PRNGKey(0)
    # x = jnp.ones((1, 128))
    # params = model.init(rng, x)
    # y = model.apply(params, x)
    # print(y)

    config = GPT2Config(
        vocab_size=6 + 2,
        n_positions=550,
        n_embd=128,
        n_layer=3,
        n_head=4,
        n_inner=4 * 128,
        activation_function='gelu_new',
        resid_pdrop = 0.1,
        embd_pdrop = 0.1,
        attn_pdrop = 0.1,
        use_cache = False, # setting to true will be useful later for inference
    )
    model = FlaxGPT2(config=config, input_shape=(1, 550), state_dim=6, action_dim=2, output_dim=6, latent_dim=128, num_heads=4, num_layers=3)
    pt_model = TorchGPT2(state_dim=6, action_dim=2, output_dim=6, latent_dim=128, num_heads=4, num_layers=3)
    pt_model.load_state_dict(torch.load('/home/ubuntu/lecar-car/model_checkpoint.pth'))
    params = model.module.load_from_pytorch(pt_model, 256, 250, 50, model)
    
if __name__ == '__main__':
    main()
