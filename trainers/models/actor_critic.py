from __future__ import annotations

from dataclasses import dataclass, MISSING
from typing import Optional, Sequence, Type

from math import prod
import torch
from tensordict import TensorDictBase
from torchrl.modules import MLP, MultiAgentMLP
from benchmarl.models.common import Model, ModelConfig
import torch, torch.nn as nn
from trainers.models.gnn_cnn_backbone import GNN_CNN_BackBoneConfig, GNN_CNN_BackBone
from torchrl.data import Composite, DEVICE_TYPING
from dataclasses import asdict


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
        sentence_key: Optional[str],
        target_key: Optional[str],
        obstacle_key: Optional[str],
        gnn_emb_dim: int,
        encoder_dim: Optional[int],
        encoder_num_cells: Optional[Sequence[int]],
        use_encoder: bool,
        state_model: GNN_CNN_BackBone,
        **kwargs,
    ):

        self.sentence_key = sentence_key
        self.target_key = target_key
        self.obstacle_key = obstacle_key
        self.gnn_emb_dim = gnn_emb_dim
        self.use_encoder = use_encoder
        self.encoder_dim = encoder_dim
        self.encoder_num_cells = encoder_num_cells
                
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
        self.state_model = state_model

        if self.use_encoder:
            if encoder_dim is None:
                raise ValueError("encoder_dim must be specified when use_encoder is True")
            self.encoder = MLP(
                in_features=self.input_spec[('agents', 'observation', self.sentence_key)].shape[-1],
                out_features=encoder_dim,
                device=self.device,
                num_cells=encoder_num_cells,
            )
            

        self.mlp_in = self._environment_obs_dim() 
        self.mlp_in += self.gnn_emb_dim
        self.output_features = self.output_leaf_spec.shape[-1] 
        if self.input_has_agent_dim:
            self.mlp = MultiAgentMLP(
                n_agent_inputs=self.mlp_in,
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
                        in_features=self.mlp_in,
                        out_features=self.output_features,
                        device=self.device,
                        **kwargs,
                    )
                    for _ in range(self.n_agents if not self.share_params else 1)
                ]
            )
    
    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        
        sentence =      tensordict.get(('agents','observation',self.sentence_key))
        grid_target =   tensordict.get(('agents','observation',self.target_key))
        grid_obstacle = tensordict.get(('agents','observation',self.obstacle_key))
        
        if self.is_critic:
            print("is critic")
        
        x = self.state_model(tensordict)
        
        if self.use_encoder:
            sentence = self.encoder(sentence)
        
        # Stack all inputs
        x = torch.cat([x , grid_target, grid_obstacle, sentence], dim=-1)
        if self.input_has_agent_dim:
            res = self.mlp.forward(x)
            if not self.output_has_agent_dim:
                res = res[..., 0, :]
        else:
            if not self.share_params:
                res = torch.stack(
                    [net(x) for net in self.mlp],
                    dim=-2,
                )
            else:
                res = self.mlp[0](x)

        tensordict.set(self.out_key, res)
        return tensordict
        
    
    def _environment_obs_dim(self) -> int:
        
        """Number of input features collected from the environment and going straight to the mlp"""
        n = 0
        # Sensor observations
        # 1. Target
        n += self.input_spec[('agents', 'observation', self.target_key)].shape[-1]
        # 2. Obstacles
        n += self.input_spec[('agents', 'observation', self.obstacle_key)].shape[-1]
        # Sentence embedding
        if self.use_encoder:
            n += self.encoder_dim
        else:   
            n += self.input_spec[('agents', 'observation', self.sentence_key)].shape[-1]
        
        return n


@dataclass
class MyModelConfig(ModelConfig):
    """Dataclass config for a :class:`~benchmarl.models.Mlp`."""

    use_encoder: bool = MISSING
    state_model_config: GNN_CNN_BackBoneConfig = None
    state_model: Optional[GNN_CNN_BackBone] = None
    
    num_cells: Optional[Sequence[int]] = None
    layer_class: Optional[Type[nn.Module]] = None
    activation_class: Optional[Type[nn.Module]] = None
    
    encoder_dim: Optional[int] = None
    encoder_num_cells: Optional[Sequence[int]] = None
    
    gnn_emb_dim: Optional[int] = None

    sentence_key: Optional[str] = None
    target_key: Optional[str] = None
    obstacle_key: Optional[str] = None
    
    activation_kwargs: Optional[dict] = None
    norm_class: Type[nn.Module] = None
    norm_kwargs: Optional[dict] = None

    @staticmethod
    def associated_class():
        return MyModel
    
    def get_model(
        self,
        input_spec: Composite,
        output_spec: Composite,
        agent_group: str,
        input_has_agent_dim: bool,
        n_agents: int,
        centralised: bool,
        share_params: bool,
        device: DEVICE_TYPING,
        action_spec: Composite,
        model_index: int = 0,
    ) -> Model:
        
        backbone_kwargs  = asdict(self.state_model_config)
        backbone_kwargs["input_spec"] = input_spec
        backbone_kwargs["output_spec"] = output_spec
        backbone_kwargs["agent_group"] = agent_group
        backbone_kwargs["input_has_agent_dim"] = input_has_agent_dim
        backbone_kwargs["n_agents"] = n_agents
        backbone_kwargs["centralised"] = centralised
        backbone_kwargs["share_params"] = share_params
        backbone_kwargs["device"] = device
        backbone_kwargs["action_spec"] = action_spec
        backbone_kwargs["model_index"] = model_index
        backbone = self.state_model_config.associated_model(backbone_kwargs)
    
        kwargs = asdict(self)
        kwargs.pop("state_model_config", None)
        kwargs["state_model"] = backbone

        return self.associated_class()(
            **kwargs,
            input_spec=input_spec,
            output_spec=output_spec,
            agent_group=agent_group,
            input_has_agent_dim=input_has_agent_dim,
            n_agents=n_agents,
            centralised=centralised,
            share_params=share_params,
            device=device,
            action_spec=action_spec,
            model_index=model_index,
            is_critic=self.is_critic,
        )