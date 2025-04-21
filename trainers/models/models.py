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
from torchrl.data.utils import DEVICE_TYPING
from torchrl.modules.models.multiagent import MultiAgentNetBase
from torchrl.modules.models import MLP

class SplitMLP(nn.Module):
    def __init__(self, embedding_size, encoder, mlp, n_agents, n_agent_inputs, centralized=False):
        super().__init__()
        self.embedding_size = embedding_size
        self.encoder = encoder
        self.mlp = mlp
        self.n_agents = n_agents
        self.n_agent_inputs = n_agent_inputs
        self.centralized = centralized

    def forward(self, x):
        if self.centralized:
            # Centralized: single embedding, rest is agent-specific extras
            x_embed = x[..., :self.embedding_size]
            x_rest_parts = []
            for i in range(self.n_agents):
                start = i * self.n_agent_inputs + self.embedding_size
                end = (i + 1) * self.n_agent_inputs
                x_rest_parts.append(x[..., start:end])
            x_rest = torch.cat(x_rest_parts, dim=-1)
        else:
            # Decentralized: one embedding and rest already per agent
            x_embed = x[..., :self.embedding_size]
            x_rest = x[..., self.embedding_size:]

        encoded = self.encoder(x_embed)
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
        encoder_depth: int,
        latent_dim: int,
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

        # Encoder always takes only the embedding part
        encoder = MLP(
            in_features=self.embedding_size,
            out_features=self.latent_dim,
            depth=self.encoder_depth,
            num_cells=self.num_cells,
            activation_class=self.activation_class,
            device=device,
            **kwargs
        )

        # Adjust MLP input dim based on whether centralized or not
        if self.centralized:
            # All agent inputs minus one embedding (shared)
            mlp_input_dim = self.latent_dim + (n_agent_inputs - self.n_agents*self.embedding_size)
        else:
            # Single agent inputs
            mlp_input_dim = self.latent_dim + (self.n_agent_inputs - self.embedding_size)

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
            encoder=encoder,
            mlp=mlp,
            n_agents=self.n_agents,
            n_agent_inputs=self.n_agent_inputs,
            centralized=self.centralized,
        )
    
