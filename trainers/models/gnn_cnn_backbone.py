from __future__ import annotations

from dataclasses import dataclass, MISSING
from typing import Optional, Sequence, Type

import inspect
import warnings
from math import prod
import torch
torch.autograd.set_detect_anomaly(True)
from tensordict import TensorDictBase
from torch import nn, Tensor
from torchrl.modules import MLP, MultiAgentMLP

from tensordict.utils import _unravel_key_to_tuple, NestedKey
from benchmarl.models.common import Model, ModelConfig
import math, inspect, warnings, torch, torch.nn as nn
from torch_geometric.nn import MessagePassing

import importlib
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

class GNN_CNN_BackBone(Model):

    def __init__(
        self,
        topology: str,
        self_loops: bool,
        gnn_class: Type[torch_geometric.nn.MessagePassing],
        gnn_kwargs: Optional[dict],
        position_key: Optional[str],
        exclude_pos_from_node_features: Optional[bool],
        velocity_key: Optional[str],
        rotation_key: Optional[str],
        grid_key: Optional[str],
        edge_radius: Optional[float],
        pos_features: Optional[int],
        rot_features: Optional[int],
        vel_features: Optional[int],
        gnn_emb_dim: Optional[int],
        use_conv_2d: bool,
        cnn_filters: Optional[int],
        cnn_spatial: Optional[int],
        cnn_emb_dim: Optional[int],
        **kwargs,
    ):
        self.topology =     topology
        self.self_loops =   self_loops
        self.position_key = position_key
        self.rotation_key = rotation_key
        self.velocity_key = velocity_key
        self.grid_key =     grid_key
        self.edge_radius =  edge_radius
        self.pos_features = pos_features
        self.vel_features = vel_features
        self.rot_features = rot_features
        self.use_conv_2d =  use_conv_2d
        self.gnn_emb_dim =  gnn_emb_dim
        self.cnn_emb_dim =  cnn_emb_dim
        self.exclude_pos_from_node_features = exclude_pos_from_node_features
                
        super().__init__(
            input_spec=kwargs.pop("input_spec"),
            output_spec=kwargs.pop("output_spec"),
            agent_group=kwargs.pop("agent_group"),
            input_has_agent_dim=kwargs.pop("input_has_agent_dim"),
            n_agents=kwargs.pop("n_agents"),
            centralised=kwargs.pop("centralised"),
            share_params=kwargs.pop("share_params"),
            device=kwargs.pop("device"),
            action_spec=kwargs.pop("action_spec"),
            model_index=kwargs.pop("model_index"),
            is_critic=kwargs.pop("is_critic"),
        )
        
        G = self.input_spec[('agents', 'observation', self.grid_key)].shape[-1]
        flat_grid = G * G

        if use_conv_2d:
            C =             cnn_filters
            S =             cnn_spatial                           
            stride =        G // S
            kernel_size =   G - (S - 1) * stride            
            padding =       ((S - 1) * stride + kernel_size - G) // 2
            conv_flat =     C * S * S

            self.conv_2d = nn.Conv2d(in_channels=2, out_channels=C,
                                     kernel_size=kernel_size,
                                     stride=stride, padding=padding).to(self.device)                   
            self.linear_from_conv = nn.Linear(conv_flat, cnn_emb_dim).to(self.device)

        else:                                   
            self.cnn_emb_dim = flat_grid

        if gnn_kwargs is None:
            gnn_kwargs = {}
            
        gnn_kwargs.update({"in_channels": self._n_node_in(), "out_channels": self.gnn_emb_dim})
        if self._edge_attr_dim() and "edge_dim" in inspect.signature(gnn_class).parameters:
            gnn_kwargs["edge_dim"] = self._edge_attr_dim()
        self.gnn_supports_edge_attrs = (
            "edge_dim" in inspect.getfullargspec(gnn_class).args
        )
        
        self.edge_index = _get_edge_index(
            topology=self.topology,
            self_loops=self.self_loops,
            device=self.device,
            n_agents=self.n_agents,
        )
            
        self.gnn = gnn_class(**gnn_kwargs).to(self.device)
    
    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
            
        # Gather in_key
        pos = rot = vel = None
        if self.position_key is not None:
            pos = tensordict.get(('agents','observation',self.position_key))
        if self.rotation_key is not None:
            rot = tensordict.get(('agents','observation',self.rotation_key))
        if self.velocity_key is not None:    
            vel = tensordict.get(('agents','observation',self.velocity_key))
            
        grid_obs =      tensordict.get(('agents','observation',self.grid_key))
        obs =           tensordict.get(('agents','observation','obs'))
        batch_size =    obs.shape[:-2]
        
        if self.use_conv_2d:
            G =             grid_obs.shape[-1]
            batch_grid =    grid_obs.view(-1,2,G, G)
            conv_out =      self.conv_2d(batch_grid)        # (B, cnn_filters, H, W)
            conv_flat =     conv_out.view(*grid_obs.shape[:-3], -1)  # (B, cnn_filters * H * W)
            grid_obs =      self.linear_from_conv(conv_flat)    # (B, cnn_emb_dim)
        
        node_feat = [grid_obs, obs]
        if pos is not None and not self.exclude_pos_from_node_features:
            node_feat.append(pos)
        if rot is not None:
            node_feat.append(rot)
        if vel is not None:
            node_feat.append(vel)
            
        x = torch.cat(node_feat, dim=-1)
        
        graph = _batch_from_dense_to_ptg(
            x=x,
            edge_index=self.edge_index,
            pos=pos,
            vel=vel,
            self_loops=self.self_loops,
            edge_radius=self.edge_radius,
        )
        forward_gnn_params = {
            "x": graph.x,
            "edge_index": graph.edge_index,
        }
        if (
            self.position_key is not None or self.velocity_key is not None
        ) and self.gnn_supports_edge_attrs:
            forward_gnn_params.update({"edge_attr": graph.edge_attr})
        
        x = self.gnn(**forward_gnn_params).view(
            *batch_size, self.n_agents, self.gnn_emb_dim
        )
        
        return x
    
    def _n_node_in(self) -> int:
        """Number of input features for each node passed to the GNN."""
        n = 0

        # 1. grid_obs embedding coming from the CNN/linear layer
        n += self.cnn_emb_dim            

        # 2. plain observation vector ("obs")
        n += self.input_spec[('agents', 'observation', 'obs')].shape[-1]

        # 3. optional positional features
        if self.position_key is not None and not self.exclude_pos_from_node_features:
            n += self.pos_features          

        # 4. optional rotation features
        if self.rotation_key is not None:
            n += self.rot_features

        # 5. optional velocity features
        if self.velocity_key is not None:
            n += self.vel_features

        return n
    
    def _edge_attr_dim(self) -> int:
        """Length of the edge-attribute vector (distance, Δv, …)."""
        return (self.pos_features + 1 + self.vel_features 
                if (self.position_key or self.velocity_key) else 0)


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
    x: Tensor,
    edge_index: Optional[Tensor],
    self_loops: bool,
    pos: Tensor = None,
    vel: Tensor = None,
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


@dataclass
class GNN_CNN_BackBoneConfig(ModelConfig):

    use_conv_2d: bool = MISSING

    gnn_kwargs: Optional[dict] = None

    topology: Optional[str] = None
    self_loops: Optional[bool] = None
    num_cells: Optional[Sequence[int]] = None
    layer_class: Optional[Type[nn.Module]] = None
    activation_class: Optional[Type[nn.Module]] = None
    
    gnn_emb_dim: Optional[int] = None
    gnn_class: Optional[Type[torch_geometric.nn.MessagePassing]] = None
    
    cnn_emb_dim: Optional[int] = None
    cnn_filters: Optional[int] = None
    cnn_spatial: Optional[int] = None
    
    position_key: Optional[str] = None
    pos_features: Optional[int] = 0
    rotation_key: Optional[str] = None
    rot_features: Optional[int] = 0
    velocity_key: Optional[str] = None
    vel_features: Optional[int] = 0
    grid_key: Optional[str] = None

    exclude_pos_from_node_features: Optional[bool] = None
    edge_radius: Optional[float] = None
    activation_kwargs: Optional[dict] = None

    norm_class: Type[nn.Module] = None
    norm_kwargs: Optional[dict] = None
    
    model = None
    
    def associated_class(self) -> Type[GNN_CNN_BackBone]:
        
        return GNN_CNN_BackBone

    def associated_model(self , kwargs=None) -> GNN_CNN_BackBone:
        if self.model is None:
            self.model = self.get_model(
                input_spec=kwargs.pop("input_spec", None),
                output_spec=kwargs.pop("output_spec", None),
                agent_group=kwargs.pop("agent_group", None),
                input_has_agent_dim=kwargs.pop("input_has_agent_dim", False),
                n_agents=kwargs.pop("n_agents", 1),
                centralised=kwargs.pop("centralised", False),
                share_params=kwargs.pop("share_params", False),
                device=kwargs.pop("device", "cpu"),
                action_spec=kwargs.pop("action_spec", None),
                model_index=kwargs.pop("model_index", 0),
            )
            return self.model
        else:
            return self.model