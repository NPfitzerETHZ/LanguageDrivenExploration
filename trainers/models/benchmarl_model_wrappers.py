from __future__ import annotations

from dataclasses import dataclass, MISSING
from typing import Optional, Sequence, Type

import inspect
import warnings
from math import prod
import torch
from tensordict import TensorDictBase
from torch import nn, Tensor
from torchrl.modules import MLP, MultiAgentMLP

from tensordict.utils import _unravel_key_to_tuple, NestedKey
from benchmarl.models.common import Model, ModelConfig

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



class MyModel(Model):
    """Multi layer perceptron model.

    Args:
        num_cells (int or Sequence[int], optional): number of cells of every layer in between the input and output. If
            an integer is provided, every layer will have the same number of cells. If an iterable is provided,
            the linear layers out_features will match the content of num_cells.
        layer_class (Type[nn.Module]): class to be used for the linear layers;
        activation_class (Type[nn.Module]): activation class to be used.
        activation_kwargs (dict, optional): kwargs to be used with the activation class;
        norm_class (Type, optional): normalization class, if any.
        norm_kwargs (dict, optional): kwargs to be used with the normalization layers;

    """

    def __init__(
        self,
        topology: str,
        self_loops: bool,
        gnn_class: Type[torch_geometric.nn.MessagePassing],
        gnn_kwargs: Optional[dict],
        position_key: Optional[str],
        exclude_pos_from_node_features: Optional[bool],
        velocity_key: Optional[str],
        sentence_key: Optional[str],
        grid_key: Optional[str],
        target_key: Optional[str],
        edge_radius: Optional[float],
        pos_features: Optional[int],
        vel_features: Optional[int],
        emb_dim: Optional[int],
        use_gnn: bool,
        use_sentence_encoder: bool,
        use_conv_2d: bool,
        **kwargs,
    ):
        self.topology = topology
        self.self_loops = self_loops
        self.position_key = position_key
        self.velocity_key = velocity_key
        self.sentence_key = sentence_key
        self.grid_key = grid_key
        self.target_key = target_key
        self.exclude_pos_from_node_features = exclude_pos_from_node_features
        self.edge_radius = edge_radius
        self.pos_features = pos_features
        self.vel_features = vel_features
        self.use_gnn = use_gnn
        self.use_sentence_encoder = use_sentence_encoder
        self.use_conv_2d = use_conv_2d
        self.emb_dim = emb_dim
                
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
        
        conv_out_channel_dim = 3
        
        #==== Setup Conv layer ====#
        if self.use_conv_2d:
            G = self.input_spec[('agents', 'observation', self.grid_key)].shape[-1]
            stride = G // conv_out_channel_dim
            kernel_size = G - 2 * stride
            padding = ((conv_out_channel_dim - 1) * stride + kernel_size - G) // 2
            self.conv_2d =  nn.Conv2d(in_channels=1, out_channels=emb_dim, kernel_size=kernel_size, stride=stride, padding=padding).to(self.device)
        
        
        #==== Setup GNN ====#
        if use_gnn:
            if self.pos_features > 0:
                self.pos_features += 1  # We will add also 1-dimensional distance
            self.edge_features = self.pos_features + self.vel_features
            self.input_features = conv_out_channel_dim ** 2
            # Input keys
            if self.position_key is not None and not self.exclude_pos_from_node_features:
                self.input_features += self.pos_features - 1
            if self.velocity_key is not None:
                self.input_features += self.vel_features

            if gnn_kwargs is None:
                gnn_kwargs = {}
            gnn_kwargs.update(
                {"in_channels": self.input_features, "out_channels": self.emb_dim}
            )
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
                gnn_kwargs.update({"edge_dim": self.edge_features})

            self.gnns = nn.ModuleList(
                [
                    gnn_class(**gnn_kwargs).to(self.device)
                    for _ in range(self.n_agents if not self.share_params else 1)
                ]
            )
            self.edge_index = _get_edge_index(
                topology=self.topology,
                self_loops=self.self_loops,
                device=self.device,
                n_agents=self.n_agents,
            )
        
        #=== Setup sentence encoder ===#
        if use_sentence_encoder:
            self.sentence_encoder = MLP(
                in_features=self.input_spec[('agents', 'observation', self.sentence_key)].shape[-1],
                out_features=128,
                device=self.device,
                num_cells=[256,256],
                activation_class=kwargs.get('activation_class', nn.ReLU),
                layer_class=kwargs.get('layer_class', nn.Linear)
            )
        
        #==== Setup Policy MLP ====#
        if use_gnn:
            self.input_features = (128 if self.use_sentence_encoder else self.input_spec[('agents', 'observation', self.sentence_key)].shape[-1]) + emb_dim + sum(
            [spec.shape[-1] for key, spec in self.input_spec.items(True, True)
            if _unravel_key_to_tuple(key)[-1] in (target_key,'obs')]
            )
        else:
            self.input_features = sum(
                [spec.shape[-1] for key, spec in self.input_spec.items(True, True)
                if _unravel_key_to_tuple(key)[-1] in (self.position_key, self.velocity_key, self.target_key, 'obs')])
            self.input_features += (128 if self.use_sentence_encoder else self.input_spec[('agents', 'observation', self.sentence_key)].shape[-1])
            self.input_features += (9 if self.use_conv_2d else self.input_spec[('agents', 'observation', self.grid_key)].shape[-1] * self.input_spec[('agents', 'observation', self.grid_key)].shape[-2])

        
        self.output_features = self.output_leaf_spec.shape[-1]

        if self.input_has_agent_dim:
            self.mlp = MultiAgentMLP(
                n_agent_inputs=self.input_features,
                n_agent_outputs=self.output_features,
                n_agents=self.n_agents,
                centralised=self.centralised,
                share_params=self.share_params,
                device=self.device,
                **kwargs,
            )
        else:
            self.mlp = nn.ModuleList(
                [
                    MLP(
                        in_features=self.input_features,
                        out_features=self.output_features,
                        device=self.device,
                        **kwargs,
                    )
                    for _ in range(self.n_agents if not self.share_params else 1)
                ]
            )


    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        # Gather in_key
        pos = tensordict.get(('agents','observation',self.position_key))
        vel = tensordict.get(('agents','observation',self.velocity_key))
        grid_visit = tensordict.get(('agents','observation',self.grid_key))
        sentence = tensordict.get(('agents','observation',self.sentence_key))
        grid_target = tensordict.get(('agents','observation',self.target_key))
        obs = tensordict.get(('agents','observation','obs'))
        batch_size = obs.shape[:-2]
        
        if self.use_conv_2d:
            G = grid_visit.shape[-1]
            batch_grid = grid_visit.view(-1,1,G, G)
            grid_visit = self.conv_2d(batch_grid).mean(dim=(-3)).view(*grid_visit.shape[:-3], grid_visit.shape[-3], -1)
        else:
            grid_visit = grid_visit.view(*grid_visit.shape[:-2], -1)
        
        node_feat = [grid_visit]
        if pos is not None and not self.exclude_pos_from_node_features:
            node_feat.append(pos)
        if vel is not None:
            node_feat.append(vel)
        x = torch.cat(node_feat, dim=-1)
        
        if self.use_gnn:
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
            
            if not self.share_params:
                if not self.centralised:
                    x = torch.stack(
                        [
                            gnn(**forward_gnn_params).view(
                                *batch_size,
                                self.n_agents,
                                self.emb_dim,
                            )[..., i, :]
                            for i, gnn in enumerate(self.gnns)
                        ],
                        dim=-2,
                    )
                else:
                    x = torch.stack(
                        [
                            gnn(**forward_gnn_params)
                            .view(
                                *batch_size,
                                self.n_agents,
                                self.emb_dim,
                            )
                            .mean(dim=-2)  # Mean pooling
                            for i, gnn in enumerate(self.gnns)
                        ],
                        dim=-2,
                    )

            else:
                x = self.gnns[0](**forward_gnn_params).view(
                    *batch_size, self.n_agents, self.emb_dim
                )
                if self.centralised:
                    x = res.mean(dim=-2)  # Mean pooling
                    
        # Pre-embed sentence
        if self.use_sentence_encoder:
            sentence = self.sentence_encoder(sentence)
        
        # Stack all inputs
        x = torch.cat([x, obs, grid_target, sentence], dim=-1)

        # Has multi-agent input dimension
        if self.input_has_agent_dim:
            res = self.mlp.forward(x)
            if not self.output_has_agent_dim:
                # If we are here the module is centralised and parameter shared.
                # Thus the multi-agent dimension has been expanded,
                # We remove it without loss of data
                res = res[..., 0, :]

        # Does not have multi-agent input dimension
        else:
            if not self.share_params:
                res = torch.stack(
                    [net(x) for net in self.mlp],
                    dim=-2,
                )
            else:
                res = self.mlp[0](input)

        tensordict.set(self.out_key, res)
        return tensordict

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
class MyModelConfig(ModelConfig):
    """Dataclass config for a :class:`~benchmarl.models.Mlp`."""

    use_gnn: bool = MISSING
    use_sentence_encoder: bool = MISSING
    use_conv_2d: bool = MISSING

    gnn_kwargs: Optional[dict] = None

    topology: Optional[str] = None
    self_loops: Optional[bool] = None
    num_cells: Optional[Sequence[int]] = None
    layer_class: Optional[Type[nn.Module]] = None
    activation_class: Optional[Type[nn.Module]] = None
    emb_dim: Optional[int] = None
    gnn_class: Optional[Type[torch_geometric.nn.MessagePassing]] = None
    position_key: Optional[str] = None
    pos_features: Optional[int] = 0
    velocity_key: Optional[str] = None
    vel_features: Optional[int] = 0
    sentence_key: Optional[str] = None
    grid_key: Optional[str] = None
    target_key: Optional[str] = None
    
    exclude_pos_from_node_features: Optional[bool] = None
    edge_radius: Optional[float] = None
    activation_kwargs: Optional[dict] = None

    norm_class: Type[nn.Module] = None
    norm_kwargs: Optional[dict] = None

    @staticmethod
    def associated_class():
        return MyModel