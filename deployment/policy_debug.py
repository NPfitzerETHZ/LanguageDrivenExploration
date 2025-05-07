import torch
from torch import tensor
from tensordict import TensorDict
from sentence_transformers import SentenceTransformer
from omegaconf import OmegaConf
import importlib
import copy

from benchmarl.experiment import ExperimentConfig, Experiment
from benchmarl.algorithms import MappoConfig
from benchmarl.models.mlp import MlpConfig
from benchmarl.environments import VmasTask

from typing import Callable, Optional
from benchmarl.environments import VmasTask
from benchmarl.utils import DEVICE_TYPING
from torchrl.envs import EnvBase, VmasEnv
import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  
from scenarios.simple_language_deployment_scenario import MyLanguageScenario
import hydra
from omegaconf import DictConfig


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
    
    print(config["experiment_config"].value)
    
    experiment_config = ExperimentConfig(**config["experiment_config"].value)
    experiment_config.restore_sfile = str("/home/npfitzer/robomaster_ws/src/LanguageDrivenExploration/checkpoints/benchmarl/single_agent_first/single_agent_llm_deployment.pt")
    algorithm_config = MappoConfig(**config["algorithm_config"].value)
    model_config = MlpConfig(**config["model_config"].value)
    task = VmasTask.NAVIGATION.get_from_yaml()
    task.config = config["task_config"].value
    
    model_config.activation_class = load_class(model_config.activation_class)
    model_config.layer_class = load_class(model_config.layer_class)
    
    experiment = Experiment(
        config=experiment_config,
        task=task,
        algorithm_config=algorithm_config,
        model_config=model_config,
        seed=0
    )
    
    return experiment

@hydra.main(config_path="/home/npfitzer/robomaster_ws/src/LanguageDrivenExploration/checkpoints/benchmarl/single_agent_first/", 
            config_name="benchmarl_mappo", version_base="1.1")
def main(config: DictConfig):
    device = torch.device("cpu")
    experiment = get_experiment(config)
    policy = experiment.policy

    llm = SentenceTransformer("thenlper/gte-large", device="cpu")

    use_llm = config["task_config"].value.llm_activate
    observe_targets = config["task_config"].value.observe_targets
    observe_pos_history = config["task_config"].value.observe_pos_history
    use_gnn = config["task_config"].value.use_gnn
    mini_grid_radius = config["task_config"].value.mini_grid_radius
    agent_weight = config["task_config"].value.agent_weight

    pos_dim = 2
    vel_dim = 2
    sentence_dim = 1024  # gte-large output
    dummy_grid_obs_dim = (2 * mini_grid_radius + 1) ** 2  # visits + obstacle
    dummy_target_obs_dim = (2 * mini_grid_radius + 1) ** 2  # target map
    history_length = config["task_config"].value.history_length
    target_count_dim = 1
    dt = 0.1

    print("\n=== Policy Manual Debug Interface ===\n")
    print("Enter a sentence for LLM embedding.")
    print("Then enter position and velocity (normalized to [-1, 1]).\n")

    while True:
        try:
            sentence = input("Sentence [default: 'Looking for a target in the north-east corner'] (or 'exit'): ").strip()
            if sentence.lower() == "exit":
                break
            if sentence == "":
                sentence = "Looking for a target in the north-east corner"

            # Embed sentence
            if use_llm:
                embedded_sentence = llm.encode(sentence, convert_to_tensor=True).unsqueeze(0).to(device)
                print(f"LLM embedding shape: {embedded_sentence.shape}")
            else:
                embedded_sentence = None

            pos_str = input("Enter position_x, position_y [default: -0.9,-0.9]: ").strip()
            if pos_str == "":
                pos = torch.tensor([[-0.9, -0.9]], dtype=torch.float32)
            else:
                pos = torch.tensor([[float(x) for x in pos_str.split(",")]], dtype=torch.float32)

            vel_str = input("Enter velocity_x, velocity_y [default: 0.1,0.1]: ").strip()
            if vel_str == "":
                vel = torch.tensor([[0.1, 0.1]], dtype=torch.float32)
            else:
                vel = torch.tensor([[float(x) for x in vel_str.split(",")]], dtype=torch.float32)

            # Optional dummy parts
            components = []

            if use_llm:
                components.append(embedded_sentence)

            if observe_targets:
                dummy_target = torch.zeros((1, dummy_target_obs_dim), dtype=torch.float32)
                components.append(dummy_target)

            if observe_pos_history:
                dummy_hist = torch.zeros((1, history_length * pos_dim), dtype=torch.float32)
                components.append(dummy_hist)

            if use_llm:
                components.append(torch.tensor([[1]], dtype=torch.float32))  # covered target count

            # Grid obs
            dummy_grid = torch.zeros((1, dummy_grid_obs_dim), dtype=torch.float32)
            dummy_grid = torch.tensor([[0.7311, 0.5000, 0.0000, 0.0000, 0.2689, 0.8808, 0.0000, 0.0000, 0.0000]], dtype=torch.float32)
            components.append(dummy_grid)

            # Pose
            if not use_gnn:
                components.append(pos)
                components.append(vel)

            obs = torch.cat(components, dim=-1)
            print(f"Total observation shape: {obs.shape}")

            input_td = TensorDict({
                ("agents", "observation"): obs
            }, batch_size=[1])

            output_td = policy(input_td)
            print(output_td)
            action = output_td[("agents", "action")]
            log_prob = output_td[("agents", "log_prob")]
            
            action /= agent_weight
            ref_velocity = vel + action * dt

            print(f"→ Action: {action.tolist()[:2]} | Log Prob: {log_prob.item()}\n")

        except Exception as e:
            print(f"Error: {e}\n")



if __name__ == "__main__":
    # Adjust to your actual config path
    main()