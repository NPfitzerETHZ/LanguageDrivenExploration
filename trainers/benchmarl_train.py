import copy
import torch
import torch.nn as nn
from typing import Callable, Optional
from benchmarl.environments import VmasTask
from benchmarl.utils import DEVICE_TYPING
from torchrl.envs import EnvBase, VmasEnv
import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 
#from scenarios.old.grid_maps import MyGridMapScenario
#from llm_heading_scenario import MyLanguageScenario
#from scenarios.decentralized.decentralized_exploration import MyLanguageScenario
from scenarios.centralized.multi_agent_llm_exploration import MyLanguageScenario

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

# Comms_radius in normalized Frame
comms_radius = 1.0
use_gnn = True

VmasTask.get_env_fun = get_env_fun

# Loads from "benchmarl/conf/experiment/base_experiment.yaml"
experiment_config = ExperimentConfig.get_from_yaml()
# Loads from "benchmarl/conf/task/vmas/balance.yaml"
task = VmasTask.NAVIGATION.get_from_yaml()
task.config = {
    # === Map & Scenario Layout ===
    "x_semidim": 3.0,
    "y_semidim": 3.0,
    "covering_range": 0.15,
    "agent_radius": 0.16,
    "n_obstacles": 0,

    # === Agent/Target Counts & Behavior ===
    "n_agents": 2,
    "agents_per_target": 1,
    "n_targets_per_class": 4,
    "n_target_classes": 1,
    "n_targets": 4,  # 4 per class * 1 class
    "done_at_termination": True,

    # === Rewards ===
    "shared_target_reward": True,
    "shared_final_reward": True,
    "agent_collision_penalty": -1.5,
    "obstacle_collision_penalty": -0.5,
    "covering_rew_coeff": 7.0,
    "false_covering_penalty_coeff": -0.25,
    "time_penalty": 0.00,
    "terminal_rew_coeff": 15.0,
    "exponential_search_rew_coeff": 1.5,
    "termination_penalty_coeff": -5.0,

    # === Exploration Rewards ===
    "use_expo_search_rew": True,
    "grid_visit_threshold": 4,
    "exploration_rew_coeff": -0.05,
    "new_cell_rew_coeff": 0.05,
    "heading_exploration_rew_coeff": 20.0,

    # === Lidar & Sensing ===
    "use_lidar": False,
    "n_lidar_rays_entities": 8,
    "n_lidar_rays_agents": 12,
    "use_velocity_controller": True,

    # === Agent Communication & GNNs ===
    "use_gnn": use_gnn,
    "comm_dim": 0,
    "comms_radius": comms_radius,

    # === Observation Settings ===
    "observe_grid": True,
    "observe_targets": False,
    "observe_pos_history": False,
    "observe_vel_history": False,
    "use_grid_data": True,
    "use_class_data": False,
    "use_max_targets_data": False,

    # === Grid Settings ===
    "num_grid_cells": 400,
    "mini_grid_radius": 1,

    # === Movement & Dynamics ===
    "agent_weight": 1.0,
    "agent_v_range": 1.0,
    "agent_a_range": 1.0,
    "min_collision_distance": 0.1,
    "linear_friction": 0.1,

    # === Histories ===
    "history_length": 0,
    
    # === Language & LLM Goals ===
    "embedding_size": 1024,
    "llm_activate": True,

    # === External Inputs ===
    "data_json_path": "data/language_data_complete_multi_target_color_scale.json",
    "decoder_model_path": "decoders/llm0_decoder_model_grid_single_target_color.pth",
    "use_decoder": False,

    # === Visuals ===
    "viewer_zoom": 1,

    # === Additional Scenario ===
    "max_steps": 250
}

# Loads from "benchmarl/conf/algorithm/mappo.yaml"
algorithm_config = MappoConfig.get_from_yaml()
algorithm_config.entropy_coef = 0.0000
model_config = MlpConfig(num_cells=[256,256,256],layer_class=nn.Linear,activation_class=nn.ReLU)
critic_model_config = MlpConfig(num_cells=[256,256,256],layer_class=nn.Linear,activation_class=nn.ReLU)

if use_gnn:
    
    gnn_config = GnnConfig(
        topology="from_pos", # Tell the GNN to build topology from positions and edge_radius
        edge_radius=comms_radius, # The edge radius for the topology
        self_loops=True,
        gnn_class=torch_geometric.nn.conv.GATv2Conv,
        gnn_kwargs={"add_self_loops": True, "residual": True}, # kwargs of GATv2Conv, residual is helpf>
        position_key="pos",
        pos_features=2,
        velocity_key="vel",
        vel_features=2,
        exclude_pos_from_node_features=False, # Do we want to use pos just to build edge features or al>
    )

    # We add an MLP layer to process GNN output node embeddings into actions
    mlp_config = MlpConfig(num_cells=[256,256],layer_class=nn.Linear,activation_class=nn.ReLU)
    model_config = SequenceModelConfig(model_configs=[gnn_config, mlp_config], intermediate_sizes=[256])
    critic_model_config = MlpConfig(num_cells=[512,256,256],layer_class=nn.Linear,activation_class=nn.ReLU)

train_device = "cpu" # @param {"type":"string"}
vmas_device = "cpu" # @param {"type":"string"}
experiment_config.sampling_device = vmas_device
experiment_config.train_device = train_device

experiment_config.render = True
experiment_config.evaluation = True
experiment_config.share_policy_params = True # Policy parameter sharing on
experiment_config.loggers = ["wandb"]
experiment_config.max_n_frames = 18_000_000 # Runs one iteration, change to 50_000_000 for full training
experiment_config.evaluation_interval = 120_000
experiment_config.on_policy_collected_frames_per_batch = 30_000
experiment_config.on_policy_n_envs_per_worker = 125
experiment_config.on_policy_minibatch_size = 3_000  # closer to RLlib’s 4096
experiment_config.on_policy_n_minibatch_iters = 45


# Catastrophic Reward Decay Counter-Measures:
algorithm_config.critic_coef = 0.5
algorithm_config.loss_critic_type = 'smooth_l1'
experiment_config.clip_grad_val = 2.
experiment_config.on_policy_n_minibatch_iters = 20


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