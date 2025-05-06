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
    experiment_config.restore_file = str("/Users/nicolaspfitzer/ProrokLab/CustomScenarios/checkpoints/benchmarl/single_agent_first/single_agent_llm_deployment.pt")
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
        seed=1
    )
    
    return experiment

@hydra.main(config_path="/Users/nicolaspfitzer/ProrokLab/CustomScenarios/checkpoints/benchmarl/single_agent_first/", 
            config_name="benchmarl_mappo", version_base="1.1")
def main(config: DictConfig):
    device = torch.device("cpu")
    experiment = get_experiment(config)
    
    experiment.evaluate()
    
if __name__ == "__main__":
    # Adjust to your actual config path
    main()