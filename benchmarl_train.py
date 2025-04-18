import copy
import torch
import torch.nn as nn
from typing import Callable, Optional
from benchmarl.environments import VmasTask
from benchmarl.utils import DEVICE_TYPING
from torchrl.envs import EnvBase, VmasEnv  
from grid_maps import MyGridMapScenario
#from llm_heading_scenario import MyLanguageScenario
from llm_target_class_scenario import MyLanguageScenario

from benchmarl.models import GnnConfig, SequenceModelConfig
import torch_geometric
from benchmarl.environments import VmasTask, Smacv2Task, PettingZooTask, MeltingPotTask
from benchmarl.experiment import ExperimentConfig, Experiment
from benchmarl.algorithms import MappoConfig
from benchmarl.models.mlp import MlpConfig

import os
from pathlib import Path

def get_env_fun(
    self,
    num_envs: int,
    continuous_actions: bool,
    seed: Optional[int],
    device: DEVICE_TYPING,
) -> Callable[[], EnvBase]:
    config = copy.deepcopy(self.config)
    if self is VmasTask.NAVIGATION: # This is the only modification we make ....
        scenario = MyLanguageScenario() # .... ends here
    else:
        scenario = self.name.lower()
    return lambda: VmasEnv(
        scenario=scenario,
        num_envs=num_envs,
        continuous_actions=continuous_actions,
        seed=seed,
        device=device,
        categorical_actions=True,
        clamp_actions=True,
        **config,
    )

comms_radius = 0.3
use_gnn = False

VmasTask.get_env_fun = get_env_fun

# Loads from "benchmarl/conf/experiment/base_experiment.yaml"
experiment_config = ExperimentConfig.get_from_yaml()
# Loads from "benchmarl/conf/task/vmas/balance.yaml"
task = VmasTask.NAVIGATION.get_from_yaml()
task.config = {
        "max_steps": 200,
        "n_agents": 3,
        "n_targets_per_class": 3,
        "n_target_classes": 3,
        "comms_radius": comms_radius,
        "use_gnn": use_gnn,
        "n_obstacles": 0,
        "global_heading_objective": False,
        "num_grid_cells": 400,
        "data_json_path": 'data/language_data_complete_single_target_color_simple.json',
        "decoder_model_path": 'decoders/llm0_decoder_model_grid_single_target_color.pth',
        "use_decoder": False
}

# Loads from "benchmarl/conf/algorithm/mappo.yaml"
algorithm_config = MappoConfig.get_from_yaml()
algorithm_config.entropy_coef = 0.0001
# Loads from "benchmarl/conf/model/layers/mlp.yaml"
# model_config = MlpConfig.get_from_yaml()
# model_config.num_cells = [256, 256, 256]
model_config = MlpConfig(num_cells=[256,256,256],layer_class=nn.Linear,activation_class=nn.ReLU)
#model_config.norm_class = nn.LayerNorm(normalized_shape=)
#model_config.activation_class = nn.LeakyReLU()
#critic_model_config = MlpConfig.get_from_yaml()
#critic_model_config.num_cells = [256, 256, 256]
critic_model_config = MlpConfig(num_cells=[256,256,256],layer_class=nn.Linear,activation_class=nn.ReLU)

if use_gnn:
    gnn_config = GnnConfig(
        topology="from_pos", # Tell the GNN to build topology from positions and edge_radius
        edge_radius=comms_radius, # The edge radius for the topology
        self_loops=False,
        gnn_class=torch_geometric.nn.conv.GATv2Conv,
        gnn_kwargs={"add_self_loops": False, "residual": True}, # kwargs of GATv2Conv, residual is helpful in RL
        position_key="pos",
        pos_features=2,
        velocity_key="vel",
        vel_features=2,
        exclude_pos_from_node_features=True, # Do we want to use pos just to build edge features or also keep it in node features? Here we remove it as we want to be invariant to system translations (we do not use absolute positions)
    )
    # We add an MLP layer to process GNN output node embeddings into actions
    mlp_config = MlpConfig.get_from_yaml()
    model_config = SequenceModelConfig(model_configs=[gnn_config, mlp_config], intermediate_sizes=[256])

train_device = "cpu" # @param {"type":"string"}
vmas_device = "cpu" # @param {"type":"string"}
experiment_config.sampling_device = vmas_device
experiment_config.train_device = train_device

experiment_config.render = True
experiment_config.evaluation = True
experiment_config.share_policy_params = True # Policy parameter sharing on
experiment_config.loggers = ["csv"]
experiment_config.max_n_frames = 20_000_000 # Runs one iteration, change to 50_000_000 for full training
experiment_config.evaluation_interval = 200_000
experiment_config.on_policy_collected_frames_per_batch = 50_000
experiment_config.on_policy_n_envs_per_worker = 250
experiment_config.on_policy_minibatch_size = 4_000  # closer to RLlib’s 4096
experiment_config.on_policy_n_minibatch_iters = 45

experiment_config.save_folder = Path(os.path.dirname(os.path.realpath(__file__))) / "experiments"
experiment_config.save_folder.mkdir(parents=True, exist_ok=True)
# Checkpoint at every 3rd iteration
experiment_config.checkpoint_interval = (
    experiment_config.on_policy_collected_frames_per_batch * 3
)

print(experiment_config)
print(algorithm_config)
print(task)
experiment = Experiment(
    task=task,
    algorithm_config=algorithm_config,
    model_config=model_config,
    critic_model_config=critic_model_config,
    seed=0,
    config=experiment_config,
)
experiment.run()