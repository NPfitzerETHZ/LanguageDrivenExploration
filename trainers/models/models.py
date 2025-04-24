# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import abc
from copy import deepcopy
from textwrap import indent
from typing import Optional, Sequence, Tuple, Type, Union

import numpy as np

import torch
from tensordict import TensorDict
from torch import nn
from torch.nn import functional as F
from torchrl.data.utils import DEVICE_TYPING
from torchrl.modules.models.multiagent import MultiAgentNetBase
from torchrl.modules.models import MLP
from torchrl.modules.models.utils import create_on_device

class GridAttention(nn.Module):
    def __init__(self, num_object_types, sentence_embedding_dim, D_obs=9, D_model=64):
        super().__init__()
        self.embedding = nn.Embedding(num_object_types, D_obs)

        # Project grid embeddings to keys and values
        self.key_proj = nn.Linear(D_obs, D_model)
        self.value_proj = nn.Linear(D_obs, D_model)

        # Project sentence embedding to query
        self.query_proj = self.query_proj = nn.Sequential(
            nn.Linear(sentence_embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, D_model)
        )

    def forward(self, grid_targets, sentence_embedding):
        """
        grid_targets: [B, A, N]      - batch of grids (e.g. flattened 3x3 = 9 cells)
        sentence_embedding: [B,A,E] - one embedding per instruction (E = sentence_embedding_dim)
        """

        grid_embeds = self.embedding(grid_targets.int())

        # [B', N, D_model]
        K = self.key_proj(grid_embeds)
        #V = self.value_proj(grid_embeds)

        # [B', 1, D_model]
        Q = self.query_proj(sentence_embedding).unsqueeze(-2)

        # Attention scores: [B', 1, N]
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (K.size(-1) ** 0.5)

        # Attention weights: [B', N]
        attn_weights = F.softmax(attn_scores.squeeze(-2), dim=-1)
        
        # Apply attention weights to values
        # V: [B', N, D_model]
        # attn_output: [B', 1, D_model]
        #attn_output = torch.matmul(attn_weights.unsqueeze(1), V).squeeze(1)

        return attn_weights  # can treat this as a soft attention mask

class TaskHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, activation_class, device):
        activation = create_on_device(activation_class, device)
        super(TaskHead, self).__init__()
        layers = [nn.Linear(input_dim, hidden_dim), activation]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), activation]
        layers += [nn.Linear(hidden_dim, output_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class SplitMLP(nn.Module):
    def __init__(self, embedding_size, local_grid_dim, task_heads, mlp, n_agents, n_agent_inputs, centralized=False):
        super().__init__()
        self.embedding_size = embedding_size
        self.local_grid_dim = local_grid_dim
        self.task_heads = task_heads
        self.mlp = mlp
        self.n_agents = n_agents
        self.n_agent_inputs = n_agent_inputs
        self.centralized = centralized
        
    def forward(self, x):
        # Task Encoders
        x_embed = x[..., :self.embedding_size]
        head_outputs = []

        if self.centralized:
            # Centralized: single embedding, rest is agent-specific extras
            x_rest_parts = []
            x_target_grid_parts = []
            for i in range(self.n_agents):
                start = i * self.n_agent_inputs + self.embedding_size
                end = start + self.local_grid_dim
                x_target_grid_parts.append(x[...,start:end])
                start = end
                end = (i + 1) * self.n_agent_inputs
                x_rest_parts.append(x[..., start:end])
            x_target_grid = torch.cat(x_target_grid_parts, dim=-1)
            x_rest = torch.cat(x_rest_parts, dim=-1)
        else:
            # Decentralized: one embedding and rest already per agent
            x_target_grid = x[..., self.embedding_size:self.embedding_size+self.local_grid_dim]
            x_rest = x[..., self.embedding_size+self.local_grid_dim:]
        
        for task_name, head in self.task_heads.items():
            if task_name == "class":
                # For target classes, we need to use the GridAttention module
                encoded = head(x_target_grid, x_embed)
            else:
                # For other tasks, we use the standard TaskHead
                encoded = head(x_embed)
            head_outputs.append(encoded)
        
        # Combine all head outputs into one tensor along the last dimension
        encoded = torch.cat(head_outputs, dim=-1)
        combined = torch.cat([encoded, x_rest], dim=-1)
        return self.mlp(combined)

class MultiAgentMLP_Custom(MultiAgentNetBase):

    def __init__(
        self,
        n_agent_inputs: int | None,
        n_agent_outputs: int,
        n_agents: int,
        *,
        centralized: bool | None = None,
        share_params: bool | None = None,
        device: Optional[DEVICE_TYPING] = None,
        depth: Optional[int] = None,
        num_cells: Optional[Union[Sequence, int]] = None,
        activation_class: Optional[Type[nn.Module]] = nn.Tanh,
        use_td_params: bool = True,
        embedding_size: int,
        local_grid_dim: int,
        encoder_depth: int,
        latent_dim: int,
        target_classes: int,
        task_dict,
        **kwargs,
    ):
        self.n_agents = n_agents
        self.n_agent_inputs = n_agent_inputs
        self.n_agent_outputs = n_agent_outputs
        self.share_params = share_params
        self.centralized = centralized
        self.num_cells = num_cells
        self.activation_class = activation_class
        self.depth = depth
        self.embedding_size = embedding_size
        self.encoder_depth = encoder_depth
        self.latent_dim = latent_dim
        self.task_dict = task_dict
        self.local_grid_dim = local_grid_dim
        self.target_classes = target_classes

        super().__init__(
            n_agents=n_agents,
            centralized=centralized,
            share_params=share_params,
            device=device,
            agent_dim=-2,
            use_td_params=use_td_params,
            **kwargs,
        )

    def _pre_forward_check(self, inputs):
        if inputs.shape[-2] != self.n_agents:
            raise ValueError(
                f"Multi-agent network expected input with shape[-2]={self.n_agents},"
                f" but got {inputs.shape}"
            )
        # If the model is centralized, agents have full observability
        if self.centralized:
            inputs = inputs.flatten(-2, -1)
        return inputs

    def _build_single_net(self, *, device, **kwargs):
        # Adjust agent input size if centralized
        n_agent_inputs = self.n_agent_inputs
        if self.centralized and n_agent_inputs is not None:
            n_agent_inputs *= self.n_agents  # full concatenated input per agent

        task_heads = nn.ModuleDict()
        for task_name, num_layers in self.task_dict.items():
            if task_name == "class":
                # For target classes, we need to use the GridAttention module
                task_heads[task_name] = GridAttention(
                    num_object_types=self.target_classes,
                    sentence_embedding_dim=self.embedding_size,
                    D_obs=8,
                    D_model=16
                )
            else:
                # For other tasks, we use the standard TaskHead
                task_heads[task_name] = TaskHead(
                    input_dim=self.embedding_size,
                    hidden_dim=self.num_cells,
                    output_dim=self.latent_dim,
                    num_layers=num_layers,
                    activation_class=self.activation_class,
                    device=device
                )

        if self.centralized:
            # Shared input: includes all agents' local grids + latent + shared agent features
            n_agent_features = self.embedding_size + self.local_grid_dim
            shared_agent_inputs = (self.n_agent_inputs - n_agent_features) * self.n_agents
            mlp_input_dim = (
                self.n_agents * self.local_grid_dim +
                self.latent_dim +
                shared_agent_inputs
            )
        else:
            # Single agent: one agent's local grid + latent + individual features
            single_agent_features = self.embedding_size + self.local_grid_dim
            agent_specific_inputs = self.n_agent_inputs - single_agent_features
            mlp_input_dim = (
                self.latent_dim +
                self.local_grid_dim +
                agent_specific_inputs
            )

        mlp = MLP(
            in_features=mlp_input_dim,
            out_features=self.n_agent_outputs,
            depth=self.depth,
            num_cells=self.num_cells,
            activation_class=self.activation_class,
            device=device,
            **kwargs,
        )

        return SplitMLP(
            embedding_size=self.embedding_size,
            local_grid_dim=self.local_grid_dim,
            task_heads=task_heads,
            mlp=mlp,
            n_agents=self.n_agents,
            n_agent_inputs=self.n_agent_inputs,
            centralized=self.centralized,
        )
    
