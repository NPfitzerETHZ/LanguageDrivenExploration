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
from torchrl.modules.models.multiagent import MultiAgentMLP
import torch_geometric
from torch_geometric.nn import MessagePassing                # base class
from torch_geometric.nn.pool import radius_graph, knn_graph
from torch_geometric.utils import add_self_loops
from torchrl.modules.models import MLP
from torchrl.modules.models.utils import create_on_device
from math import prod
import importlib
import inspect
import warnings
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

class MultiAgentMLP_Efficient(MultiAgentMLP):
    def __init__(
        self,
        n_agent_inputs: int | None,
        n_agent_outputs: int,
        n_agents: int,
        *,
        centralized: bool | None = None,
        share_params: bool | None = None,
        embedding_size: int,
        device: Optional[DEVICE_TYPING] = None,
        depth: Optional[int] = None,
        num_cells: Optional[Union[Sequence, int]] = None,
        activation_class: Optional[Type[nn.Module]] = nn.Tanh,
        use_td_params: bool = True,
    
        **kwargs,
    ):
        self.embedding_size = embedding_size
        super().__init__(
            n_agent_inputs=n_agent_inputs,
            n_agent_outputs=n_agent_outputs,
            n_agents=n_agents,
            centralized=centralized,
            share_params=share_params,
            device=device,
            depth=depth,
            num_cells=num_cells,
            activation_class=activation_class,
            use_td_params=use_td_params,
            **kwargs
        )
    
    def _pre_forward_check(self, inputs):
        if inputs.shape[-2] != self.n_agents:
            raise ValueError(
                f"Multi-agent network expected input with shape[-2]={self.n_agents},"
                f" but got {inputs.shape}"
            )
        # If the model is centralized, agents have full observability
        if self.centralized:
            # Extracrt the first embedding size from the input
            inpus_embed = inputs[...,0,:self.embedding_size]
            # Extract the rest of the inputs
            inputs_rest = inputs[..., self.embedding_size:]
            # Flatten the rest of the inputs
            inputs_rest = inputs_rest.flatten(-2, -1)
            # Concatenate the embedding with the rest of the inputs
            inputs = torch.cat([inpus_embed, inputs_rest], dim=-1)
        return inputs
    
    def _build_single_net(self, *, device, **kwargs):
        n_agent_inputs = self.n_agent_inputs
        if self.centralized and n_agent_inputs is not None:
            n_agent_inputs = self.embedding_size + self.n_agents * (self.n_agent_inputs - self.embedding_size)
        return MLP(
            in_features=n_agent_inputs,
            out_features=self.n_agent_outputs,
            depth=self.depth,
            num_cells=self.num_cells,
            activation_class=self.activation_class,
            device=device,
            **kwargs,
        )

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

"""
Multi‑agent GNN with FiLM conditioning
=====================================
Broadcasts a single *team‑level* `sentence_embedding` to all agents via
Feature‑wise Linear Modulation (FiLM).  Supports an arbitrary number
of GNN layers (default 1) while applying the same γ / β gates after
each layer.

Key additions compared to the original class
-------------------------------------------
* **FiLMGen** module: 2‑layer MLP adapter + projection → (γ, β).
* **n_gnn_layers** argument: builds a `ModuleList` of `gnn_class` layers.
* FiLM gate applied after every GNN layer (no extra nodes or edges).
* `sentence_key` argument so the name of the instruction tensor can be
  configured (defaults to "sentence_embedding").
"""

from typing import Optional, Type

import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import radius_graph, knn_graph

# --------------------------------------------------
# FiLM utilities
# --------------------------------------------------

class FiLMGen(nn.Module):
    """Generate FiLM parameters γ and β from a sentence embedding."""

    def __init__(self, *, sent_dim: int, hidden: int, feat_dim: int):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.LayerNorm(sent_dim),
            nn.Linear(sent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.to_gamma_beta = nn.Linear(hidden, 2 * feat_dim)

    def forward(self, sent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (γ, β) each of shape (B, feat_dim)."""
        cond = self.adapter(sent)
        gamma, beta = self.to_gamma_beta(cond).chunk(2, dim=-1)
        return gamma, beta


def film(h: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """Apply FiLM: h * (1 + γ) + β (all shapes (B*N, F))."""
    return h * (1.0 + gamma) + beta


# --------------------------------------------------
# Multi‑agent GNN with FiLM
# --------------------------------------------------
_has_torch_geometric = importlib.util.find_spec("torch_geometric") is not None
if _has_torch_geometric:
    import torch_geometric
    from torch_geometric.transforms import BaseTransform

    class _RelVel(BaseTransform):
        """Transform that reads graph.vel and writes node1.vel - node2.vel in the edge attributes"""

        def __init__(self):
            pass

        def __call__(self, data):
            (row, col), vel, pseudo = data.edge_index, data.vel, data.edge_attr

            cart = vel[row] - vel[col]
            cart = cart.view(-1, 1) if cart.dim() == 1 else cart

            if pseudo is not None:
                pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
                data.edge_attr = torch.cat([pseudo, cart.type_as(pseudo)], dim=-1)
            else:
                data.edge_attr = cart
            return data

class MultiAgentGNN(nn.Module):
    """Graph‑based multi‑agent policy head with optional multi‑layer FiLM."""

    def __init__(
        self,
        *,
        n_agents: int,
        node_input_dim: int,
        action_dim: int,
        sentence_dim: int,
        topology: str = "full",
        self_loops: bool = True,
        gnn_class: Type[MessagePassing] | None = None,
        gnn_kwargs: Optional[dict] = None,
        n_gnn_layers: int = 1,
        position_key: Optional[str] = None,
        pos_features: Optional[int] = None,
        exclude_pos_from_node_features: bool = False,
        velocity_key: Optional[str] = None,
        vel_features: Optional[int] = None,
        edge_radius: Optional[float] = None,
        emb_dim: int = 32,
        sentence_key: str = "sentence_embedding",
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        assert topology in {"full", "empty", "from_pos"}
        assert n_gnn_layers >= 1, "Need at least one GNN layer"

        gnn_kwargs = gnn_kwargs or {}
        gnn_class = gnn_class or self._default_gnn()

        self.n_agents = n_agents
        self.topology = topology
        self.self_loops = self_loops
        self.position_key = position_key
        self.velocity_key = velocity_key
        self.exclude_pos_from_node_features = exclude_pos_from_node_features
        self.edge_radius = edge_radius
        #self.device = torch.device(device)
        self.sentence_key = sentence_key
        self.emb_dim = emb_dim
        
        # self.edge_index = _get_edge_index(
        #     topology=self.topology,
        #     self_loops=self.self_loops,
        #     device=self.device,
        #     n_agents=self.n_agents,
        # )
        self._full_position_key = None
        self._full_velocity_key = None

        # ---------------- feature dimensions ---------------------------
        base_input_dim = node_input_dim  # keep original for checks
        if position_key and not exclude_pos_from_node_features:
            assert pos_features is not None
            node_input_dim += pos_features
        if velocity_key:
            assert vel_features is not None
            node_input_dim += vel_features
        self._base_obs_dim = base_input_dim  # for shape check later
        self._pos_features = pos_features
        self._vel_features = vel_features

        # whether edge features are used
        self._edge_feat_dim = 0
        if position_key:
            self._edge_feat_dim += (pos_features or 0) + 1  # Δpos + dist
        if velocity_key:
            self._edge_feat_dim += vel_features or 0
            
        self.gnn_supports_edge_attrs = (
            "edge_dim" in inspect.getfullargspec(gnn_class).args
        )
        
        if (
            self.position_key is not None or self.velocity_key is not None
        ) and not self.gnn_supports_edge_attrs:
            warnings.warn(
                "Position key or velocity key provided but GNN class does not support edge attributes. "
                "These keys will not be used for computing edge features."
            )
        if (
            position_key is not None or velocity_key is not None
        ) and self.gnn_supports_edge_attrs:
            gnn_kwargs.update({"edge_dim": self._edge_feat_dim})

        # ---------------- learnable modules ----------------------------
        self.pre_embed = nn.Linear(node_input_dim, emb_dim)
        
        # self.sentence_encoder = nn.Sequential(
        #     nn.Linear(sentence_dim, emb_dim), nn.ReLU(),
        #     nn.Linear(emb_dim, emb_dim),
        # )

        self.graph_convs = nn.ModuleList([
            gnn_class(in_channels=4, out_channels=emb_dim, **gnn_kwargs)
            for _ in range(n_gnn_layers)
        ])
        
        self.space_conv =  nn.Conv2d(in_channels=1, out_channels=emb_dim, kernel_size=3, padding=1)

        #self.film_gen = FiLMGen(sent_dim=sentence_dim, hidden=emb_dim, feat_dim=emb_dim)

        self.policy_head = nn.Sequential(
            nn.Linear(sentence_dim + node_input_dim + emb_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 2 * action_dim)
        )

        # --------------- edge index template ---------------------------
        # if topology == "full":
        #     src, dst = zip(*[(i, j) for i in range(n_agents) for j in range(n_agents) if i != j])
        #     edge_fc = torch.tensor([src, dst], dtype=torch.long)
        #     if self_loops:
        #         edge_fc = add_self_loops(edge_fc, num_nodes=n_agents)[0]
        #     self.register_buffer("_edge_template", edge_fc)
        # elif topology == "empty" and self_loops:
        #     loops = torch.arange(n_agents, dtype=torch.long).repeat(2, 1)
        #     self.register_buffer("_edge_template", loops)
        # else:
        #     self.register_buffer("_edge_template", torch.empty(2, 0, dtype=torch.long))

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(self, td) -> torch.Tensor:  # noqa: F821
        # -------- shapes & keys check ----------------------------------
        obs = td["obs"]  # (B, N, D_obs)
        B, N, _ = obs.shape
        assert N == self.n_agents, f"Expected {self.n_agents} agents, got {N}"

        pos = td.get(self.position_key) if self.position_key else None
        vel = td.get(self.velocity_key) if self.velocity_key else None

        # -------- build node features ----------------------------------
        node_feat = [obs]
        if pos is not None and not self.exclude_pos_from_node_features:
            node_feat.append(pos)
        if vel is not None:
            node_feat.append(vel)
        x = torch.cat(node_feat, dim=-1)  # (B, N, F)
        
        grid = td["grid_obs"]
        G = grid.shape[-1]
        grid = grid.view(B*N,1,G, G)
        visits = self.space_conv(grid).mean(dim=(2, 3)).view(B, N, -1)  # (B, N, emb_dim)
        #x = self.pre_embed(x)  # (B, N, emb_dim)
        
        # -------- FiLM parameters --------------------------------------
        # sentence = td[self.sentence_key][:,0,:]  # (B, sentence_dim)
        # gamma, beta = self.film_gen(sentence)  # (B, emb_dim)
        # gamma = gamma.unsqueeze(1).expand(-1, N, -1).reshape_as(x)
        # beta = beta.unsqueeze(1).expand(-1, N, -1).reshape_as(x)
        
        # sentence = td[self.sentence_key]
        # sentence = self.sentence_encoder(sentence)
        # x = torch.cat([x, sentence], dim=-1) 

        # graph = _batch_from_dense_to_ptg(
        #     x=x,
        #     edge_index=self.edge_index,
        #     pos=pos,
        #     vel=vel,
        #     self_loops=self.self_loops,
        #     edge_radius=self.edge_radius,
        # )
        # forward_gnn_params = {
        #     "x": graph.x,
        #     "edge_index": graph.edge_index,
        # }
        # if (
        #     self.position_key is not None or self.velocity_key is not None
        # ) and self.gnn_supports_edge_attrs:
        #     forward_gnn_params.update({"edge_attr": graph.edge_attr})
            
        # for conv in self.convs:
        #     #x = film(x, gamma, beta)
        #     x = conv(**forward_gnn_params).view(B, N, self.emb_dim)
        
        sentence = td[self.sentence_key]
        x = torch.cat([x, visits, sentence], dim=-1) 

        logits = self.policy_head(x).view(B, N, -1)  # (B, N, 2*action_dim)
        return logits

    # ------------------------------------------------------------------
    def _build_pos_graph(self, pos_flat: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
        if self.edge_radius is not None:
            return radius_graph(x=pos_flat, r=self.edge_radius, batch=batch_idx, loop=False)
        # else k‑NN fully connected (k = N‑1)
        return knn_graph(x=pos_flat, k=self.n_agents - 1, batch=batch_idx, loop=False)

    # ------------------------------------------------------------------
    @staticmethod
    def _default_gnn() -> Type[MessagePassing]:
        from torch_geometric.nn import GATv2Conv
        return GATv2Conv

def _get_edge_index(topology: str, self_loops: bool, n_agents: int, device: str):
        
    if topology == "full":
        adjacency = torch.ones(n_agents, n_agents, device=device, dtype=torch.long)
        edge_index, _ = torch_geometric.utils.dense_to_sparse(adjacency)
        if not self_loops:
            edge_index, _ = torch_geometric.utils.remove_self_loops(edge_index)
    elif topology == "empty":
        if self_loops:
            edge_index = (
                torch.arange(n_agents, device=device, dtype=torch.long)
                .unsqueeze(0)
                .repeat(2, 1)
            )
        else:
            edge_index = torch.empty((2, 0), device=device, dtype=torch.long)
    elif topology == "from_pos":
        edge_index = None
    else:
        raise ValueError(f"Topology {topology} not supported")

    return edge_index


def _batch_from_dense_to_ptg(
    x: torch.Tensor,
    edge_index: Optional[torch.Tensor],
    self_loops: bool,
    pos: torch.Tensor = None,
    vel: torch.Tensor = None,
    edge_radius: Optional[float] = None,
) -> torch_geometric.data.Batch:
    batch_size = prod(x.shape[:-2])
    n_agents = x.shape[-2]
    x = x.view(-1, x.shape[-1])
    if pos is not None:
        pos = pos.view(-1, pos.shape[-1])
    if vel is not None:
        vel = vel.view(-1, vel.shape[-1])

    b = torch.arange(batch_size, device=x.device)

    graphs = torch_geometric.data.Batch()
    graphs.ptr = torch.arange(0, (batch_size + 1) * n_agents, n_agents)
    graphs.batch = torch.repeat_interleave(b, n_agents)
    graphs.x = x
    graphs.pos = pos
    graphs.vel = vel
    graphs.edge_attr = None

    if edge_index is not None:
        n_edges = edge_index.shape[1]
        # Tensor of shape [batch_size * n_edges]
        # in which edges corresponding to the same graph have the same index.
        batch = torch.repeat_interleave(b, n_edges)
        # Edge index for the batched graphs of shape [2, n_edges * batch_size]
        # we sum to each batch an offset of batch_num * n_agents to make sure that
        # the adjacency matrices remain independent
        batch_edge_index = edge_index.repeat(1, batch_size) + batch * n_agents
        graphs.edge_index = batch_edge_index
    else:
        if pos is None:
            raise RuntimeError("from_pos topology needs positions as input")
        graphs.edge_index = torch_geometric.nn.pool.radius_graph(
            graphs.pos, batch=graphs.batch, r=edge_radius, loop=self_loops
        )

    graphs = graphs.to(x.device)
    if pos is not None:
        graphs = torch_geometric.transforms.Cartesian(norm=False)(graphs)
        graphs = torch_geometric.transforms.Distance(norm=False)(graphs)
    if vel is not None:
        graphs = _RelVel()(graphs)

    return graphs


class SimpleConcatCritic(nn.Module):
    """Value function that just concatenates all agent observations and the sentence.

    • Flattens agents: (B, N, D) → (B, N*D)
    • Concatenates the raw sentence embedding.
    • Passes through a shallow MLP to output V(s).
    """

    def __init__(
        self,
        *,
        n_agents: int,
        node_input_dim: int,
        sentence_dim: int,
        sentence_key: str = "sentence_embedding",
        hidden_dim: int = 256,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.n_agents = n_agents
        flat_dim = n_agents * (node_input_dim + 32) + sentence_dim

        self.net = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim * 2), nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2), nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        self.space_conv =  nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1).to(device)

    def forward(self, td: "torchrl.data.TensorDictBase") -> torch.Tensor:  # (B)
        obs = td["obs"]                     # (B,T, N, D)
        flat_obs = obs.view(*obs.shape[:-2], -1)                     # (B, T, N*D)
        
        grid = td["grid_obs"]
        G = grid.shape[-1]
        batch_grid = grid.view(-1, 1, G, G)
        conv_out = self.space_conv(batch_grid)
        visits = conv_out.mean(dim=(-2, -1))
        visits = visits.view(*grid.shape[:-3], grid.shape[-3], -1)
        flat_visits = visits.view(*visits.shape[:-2], -1)
        
        sent = td["sentence_embedding"][...,0,:]                # (B, T, S)
        x = torch.cat([flat_obs,flat_visits,sent], dim=-1)        # (B, T, N*D + S)
        output = self.net(x)              # (B, T, 1)
        
        # Centralized
        n_agent_outputs = output.shape[-1]
        output = output.view(*output.shape[:-1], n_agent_outputs)
        output = output.unsqueeze(-2)
        output = output.expand(
            *output.shape[:-2], self.n_agents, n_agent_outputs
                )
        return output
