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

from debug_policy.navigation import NavScenario

from benchmarl.environments import VmasTask
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
        scenario =NavScenario() # .... ends here
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


VmasTask.get_env_fun = get_env_fun

# Loads from "benchmarl/conf/experiment/base_experiment.yaml"
experiment_config = ExperimentConfig.get_from_yaml()
# Loads from "benchmarl/conf/task/vmas/balance.yaml"
task = VmasTask.NAVIGATION.get_from_yaml()

# Loads from "benchmarl/conf/algorithm/mappo.yaml"
algorithm_config = MappoConfig.get_from_yaml()

model_config = MlpConfig(num_cells=[256,256],layer_class=nn.Linear,activation_class=nn.ReLU)
critic_model_config = MlpConfig(num_cells=[256,256],layer_class=nn.Linear,activation_class=nn.ReLU)


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

experiment_config.off_policy_collected_frames_per_batch = 30_000
experiment_config.off_policy_n_envs_per_worker = 125
experiment_config.off_policy_train_batch_size = 3_000  # closer to RLlib’s 4096
experiment_config.off_policy_n_optimizer_steps = 45
experiment_config.off_policy_memory_size = 1_000_000
experiment_config.off_policy_init_random_frames = 0
experiment_config.off_policy_use_prioritized_replay_buffer = False
experiment_config.off_policy_prb_alpha = 0.6
experiment_config.off_policy_prb_beta = 0.4


# Catastrophic Reward Decay Counter-Measures:
#algorithm_config.critic_coef = 0.5
#algorithm_config.loss_critic_type = 'smooth_l1'
experiment_config.clip_grad_val = 1.0
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