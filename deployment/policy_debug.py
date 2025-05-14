import torch
import torch.nn as nn
from torch import tensor

from tensordict import TensorDict
from sentence_transformers import SentenceTransformer
from omegaconf import OmegaConf
import importlib
import copy

from benchmarl.experiment import ExperimentConfig, Experiment
from benchmarl.algorithms import MappoConfig
from benchmarl.models.mlp import MlpConfig
from benchmarl.models import GnnConfig, SequenceModelConfig
import torch_geometric
from benchmarl.environments import VmasTask

from typing import Callable, Optional
from benchmarl.environments import VmasTask
from benchmarl.utils import DEVICE_TYPING
from torchrl.envs import EnvBase, VmasEnv
import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  
from scenarios.centralized.multi_agent_llm_exploration import MyLanguageScenario
from scenarios.grids.world_occupancy_grid import WorldOccupancyGrid
import hydra
from omegaconf import DictConfig

from torchrl.envs.utils import ExplorationType, set_exploration_type


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

def _load_experiment_cpu(self):
    loaded_dict = torch.load(self.config.restore_file, map_location=torch.device("cpu"))
    self.load_state_dict(loaded_dict)
    return self

import importlib
def load_class(class_path: str):
    """
    Given a full class path string like 'torch.nn.modules.linear.Linear',
    dynamically load and return the class object.

    Args:
        class_path (str): Full path to the class.

    Returns:
        type: The loaded class.

    Raises:
        ImportError: If the module cannot be imported.
        AttributeError: If the class does not exist in the module.
    """
    try:
        module_path, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Cannot load class '{class_path}': {e}")


def get_experiment(config):
    
    VmasTask.get_env_fun = get_env_fun
    Experiment._load_experiment = _load_experiment_cpu
    OmegaConf.set_struct(config["model_config"].value.model_configs[0].gnn_kwargs, False)
    
    print(config["experiment_config"].value)
    
    experiment_config = ExperimentConfig(**config["experiment_config"].value)
    experiment_config.restore_file = str("/Users/nicolaspfitzer/ProrokLab/CustomScenarios/checkpoints/benchmarl/gnn_multi_agent_first/gnn_multi_agent_llm_deployment.pt")
    task = VmasTask.NAVIGATION.get_from_yaml()
    task.config = config["task_config"].value
    algorithm_config = MappoConfig(**config["algorithm_config"].value)
    
    use_gnn = config["task_config"].value.use_gnn
    
    if not use_gnn: 
        model_config = MlpConfig(**config["model_config"].value)
        model_config.activation_class = load_class(config["model_config"].value.activation_class)
        model_config.layer_class = load_class(config["model_config"].value.layer_class)
    else:
        gnn_cfg = config["model_config"].value.model_configs[0]
        mlp_cfg = config["model_config"].value.model_configs[1]
        gnn_config = GnnConfig(**gnn_cfg)
        gnn_config.gnn_class = load_class(gnn_cfg.gnn_class)
        # We add an MLP layer to process GNN output node embeddings into actions
        mlp_config = MlpConfig(**mlp_cfg)
        mlp_config.activation_class = load_class(mlp_cfg.activation_class)
        mlp_config.layer_class = load_class(mlp_cfg.layer_class)
        model_config = SequenceModelConfig(model_configs=[gnn_config, mlp_config], intermediate_sizes=config["model_config"].value.intermediate_sizes)
    
    experiment = Experiment(
        config=experiment_config,
        task=task,
        algorithm_config=algorithm_config,
        model_config=model_config,
        seed=0
    )
    
    return experiment

@hydra.main(config_path="../checkpoints/benchmarl/gnn_multi_agent_first/", 
            config_name="benchmarl_mappo", version_base="1.1")
def main(config: DictConfig):
    device = torch.device("cpu")
    experiment = get_experiment(config)
    policy = experiment.policy.to(device)

    llm = SentenceTransformer("thenlper/gte-large", device="cpu")

    use_llm = config["task_config"].value.llm_activate
    use_gnn = config["task_config"].value.use_gnn
    observe_targets = config["task_config"].value.observe_targets
    observe_pos_history = config["task_config"].value.observe_pos_history
    mini_grid_radius = config["task_config"].value.mini_grid_radius
    agent_weight = config["task_config"].value.agent_weight
    visit_threshold = config["task_config"].value.grid_visit_threshold
    num_agents = config["task_config"].value.n_agents
    history_length = config["task_config"].value.history_length

    occupancy_grid = WorldOccupancyGrid(
        batch_size=1,
        x_dim=2.0,
        y_dim=2.0,
        num_cells=1024,
        num_targets=4,
        num_targets_per_class=4,
        embedding_size=1024,
        visit_threshold=visit_threshold,
        world_grid=False,
        device=device
    )

    pos_dim = 2
    vel_dim = 2
    dt = 0.1

    print("\n=== Policy Manual Debug Interface ===\n")
    print(f"Initializing {num_agents} agents.")
    print("For each agent, enter a sentence, position, and velocity (normalized to [-1, 1]).")
    print("Press Enter to use defaults.\n")

    default_sentence = "Looking for a target in the north-east corner"
    sentences = []
    positions = []
    velocities = []

    for agent_id in range(num_agents):
        print(f"\n--- Agent {agent_id} ---")

        sentence = input(f"Sentence [default: '{default_sentence}'] (or 'exit'): ").strip()
        if sentence.lower() == "exit":
            return
        if sentence == "":
            sentence = default_sentence
        sentences.append(sentence)

        pos_str = input("Enter position_x, position_y [default: -0.9,-0.9]: ").strip()
        pos = [-0.9, -0.9] if pos_str == "" else [float(x) for x in pos_str.split(",")]
        positions.append(pos)

        vel_str = input("Enter velocity_x, velocity_y [default: 0.0,0.0]: ").strip()
        vel = [0.0, 0.0] if vel_str == "" else [float(x) for x in vel_str.split(",")]
        velocities.append(vel)

    positions = torch.tensor(positions, dtype=torch.float32)
    velocities = torch.tensor(velocities, dtype=torch.float32)

    embedded_sentences = (
        llm.encode(sentences, convert_to_tensor=True).to(device)
        if use_llm else None
    )

    print(f"LLM embeddings shape: {embedded_sentences.shape if embedded_sentences is not None else 'N/A'}")

    i = 0
    while i < 200:
        i += 1
        print(f"\nStep {i}:\n")
        try:
            obs_list = []
            for agent_idx in range(num_agents):
                components = []

                if use_llm:
                    components.append(embedded_sentences[agent_idx].unsqueeze(0))  # shape (1, 1024)

                if observe_targets:
                    tgt_obs = occupancy_grid.get_grid_target_observation(
                        positions[agent_idx:agent_idx+1], mini_grid_radius
                    )
                    components.append(tgt_obs)

                if observe_pos_history:
                    dummy_hist = torch.zeros((1, history_length * pos_dim), dtype=torch.float32)
                    components.append(dummy_hist)

                if use_llm:
                    components.append(torch.tensor([[0.0]], dtype=torch.float32))  # dummy covered count

                grid_obs = occupancy_grid.get_grid_visits_obstacle_observation(
                    positions[agent_idx:agent_idx+1], mini_grid_radius
                )
                components.append(grid_obs)

                if not use_gnn:
                    components.append(positions[agent_idx:agent_idx+1])
                    components.append(velocities[agent_idx:agent_idx+1])
                    agent_obs = torch.cat(components, dim=-1)
                    obs_list.append(agent_obs)
                else:
                    agent_obs = torch.cat(components, dim=-1)
                    obs_list.append(agent_obs)

            if not use_gnn:
                obs_tensor = torch.stack(obs_list).unsqueeze(0)  # shape: (1, num_agents, feat_dim)
                input_td = TensorDict({("agents", "observation"): obs_tensor}, batch_size=[1])
            else:
                obs_tensor = torch.stack(obs_list).squeeze(1)  # (num_agents, feat_dim)
                input_td = TensorDict({
                    ("agents", "observation", "obs"): obs_tensor,
                    ("agents", "observation", "pos"): positions,
                    ("agents", "observation", "vel"): velocities,
                }, batch_size=[num_agents])

            with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
                output_td = policy(input_td)

            action = output_td[("agents", "action")]
            log_prob = output_td[("agents", "log_prob")]

            velocities = action[:,:2]
            positions = positions + velocities * dt

            for j in range(num_agents):
                print(f"Agent {j} | Action: {action[j]} | Log prob: {log_prob[j]} | New Pos: {positions[j]} | New Vel: {velocities[j]}")

        except Exception as e:
            print(f"Error: {e}\n")




if __name__ == "__main__":
    # Adjust to your actual config path
    main()