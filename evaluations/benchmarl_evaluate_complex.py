#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

import os
from pathlib import Path

from benchmarl.algorithms import MappoConfig
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig

import copy
import torch
from typing import Callable, Optional
from benchmarl.environments import VmasTask
from benchmarl.utils import DEVICE_TYPING
from torchrl.envs import EnvBase, VmasEnv  
from scenarios.old.grid_maps import MyGridMapScenario
from scenarios.llm_heading_scenario import MyLanguageScenario

from benchmarl.models import GnnConfig, SequenceModelConfig
import torch_geometric
from benchmarl.environments import VmasTask, Smacv2Task, PettingZooTask, MeltingPotTask
from benchmarl.experiment import ExperimentConfig, Experiment
from benchmarl.algorithms import MappoConfig
from benchmarl.models.mlp import MlpConfig
import numpy as np

import os
from pathlib import Path

NUM_EVALUATION_STAGES = 8


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

if __name__ == "__main__":


    comms_radius = 0.3
    use_gnn = False

    VmasTask.get_env_fun = get_env_fun

    # Loads from "benchmarl/conf/experiment/base_experiment.yaml"
    experiment_config = ExperimentConfig.get_from_yaml()
    # Loads from "benchmarl/conf/task/vmas/balance.yaml"
    task = VmasTask.NAVIGATION.get_from_yaml()
    for i in range(NUM_EVALUATION_STAGES):
        # Keep constant cell area
        u = 0.0025
        l = 20
        h = 1
        c = (h + i * 0.5 - np.sqrt(u) * l) / np.sqrt(u)
        task.config = {
            "max_steps": 200,
            "n_agents": 1,
            "comms_radius": comms_radius,
            "use_gnn": use_gnn,
            "n_obstacles": 20,
            "global_heading_objective": False,
            "num_grid_cells": (l + c)**2,
            "x_semidim": h + i * 0.5,
            "y_semidim": h + i * 0.5,
            "use_lidar": False,
            "use_obstacle_lidar": False
        }

        # Loads from "benchmarl/conf/algorithm/mappo.yaml"
        algorithm_config = MappoConfig.get_from_yaml()
        algorithm_config.entropy_coef = 0.00
        # Loads from "benchmarl/conf/model/layers/mlp.yaml"
        model_config = MlpConfig.get_from_yaml()
        model_config.num_cells = [256, 256, 256]
        critic_model_config = MlpConfig.get_from_yaml()
        critic_model_config.num_cells = [256, 256, 256]

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
        experiment_config.render = True
        experiment_config.save_folder = Path(os.path.dirname(os.path.realpath(__file__))) / "evaluations"
        experiment_config.save_folder.mkdir(parents=True, exist_ok=True)
        experiment_config.evaluation_episodes = 200


        experiment = Experiment(
            task=task,
            algorithm_config=algorithm_config,
            model_config=model_config,
            critic_model_config=critic_model_config,
            seed=0,
            config=experiment_config,
        )

        # Now we tell it where to restore from
        experiment_config.restore_file = ("experiments/easy_exploration_with_language_1_agent/checkpoints/checkpoint_4140000.pt")
        # The experiment will be saved in the same folder as the one it is restoring from
        experiment_config.save_folder = None
        # Let's do 3 more iters

        # We can also change part of the configuration (algorithm, task). For example to evaluate in a new task.
        print(experiment_config)
        experiment = Experiment(
            algorithm_config=algorithm_config,
            model_config=model_config,
            seed=0,
            config=experiment_config,
            task=task,
        )

        # We can also evaluate
        experiment.evaluate()