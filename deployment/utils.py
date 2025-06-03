import sys
import copy
import importlib
from typing import List, Callable, Optional

import torch
from torchrl.envs import EnvBase, VmasEnv
from omegaconf import DictConfig, OmegaConf


from benchmarl.environments import VmasTask
from benchmarl.experiment import ExperimentConfig, Experiment
from benchmarl.algorithms import MappoConfig
from benchmarl.models import GnnConfig, SequenceModelConfig
from benchmarl.models.mlp import MlpConfig
from benchmarl.utils import DEVICE_TYPING

# Local Modules
from scenarios.centralized.multi_agent_llm_exploration import MyLanguageScenario
from deployment.debug.debug_policy.navigation import NavScenario


def convert_ne_to_xy(north: float, east: float) -> tuple[float, float]:
    """
    Convert coordinates from north–east (NE) ordering to x–y ordering.
    Here, x corresponds to east and y corresponds to north.
    """
    return east, north


def convert_xy_to_ne(x: float, y: float) -> tuple[float, float]:
    """
    Convert coordinates from x–y ordering to north–east (NE) ordering.
    Here, north corresponds to y and east corresponds to x.
    """
    return y, x

def get_env_fun_llm(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device: DEVICE_TYPING,
    ) -> Callable[[], EnvBase]:
        config = copy.deepcopy(self.config)
        # Override scenario for NAVIGATION task to use custom language-based scenario
        if self is VmasTask.NAVIGATION:
            scenario = MyLanguageScenario()
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

def get_env_fun_nav(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device: DEVICE_TYPING,
    ) -> Callable[[], EnvBase]:
        config = copy.deepcopy(self.config)
        # Override scenario for NAVIGATION task to use custom language-based scenario
        if self is VmasTask.NAVIGATION:
            scenario = NavScenario()
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

def _load_class(class_path: str):
    """
    Given a full class path string like 'torch.nn.modules.linear.Linear',
    dynamically load and return the class object.
    """
    try:
        module_path, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Cannot load class '{class_path}': {e}")

def get_experiment(config: DictConfig, restore_path: str, debug: bool = False) -> Experiment:
    
    if debug:
        VmasTask.get_env_fun = get_env_fun_nav
    else:
        VmasTask.get_env_fun = get_env_fun_llm

    Experiment._load_experiment = _load_experiment_cpu

    experiment_config = ExperimentConfig(**config["experiment_config"].value)
    experiment_config.restore_file = restore_path
    task = VmasTask.NAVIGATION.get_from_yaml()
    task.config = config["task_config"].value
    algorithm_config = MappoConfig(**config["algorithm_config"].value)
    
    use_gnn = config["task_config"].value.use_gnn
    
    if not use_gnn: 
        model_config = MlpConfig(**config["model_config"].value)
        model_config.activation_class = _load_class(config["model_config"].value.activation_class)
        model_config.layer_class = _load_class(config["model_config"].value.layer_class)
    else:
        OmegaConf.set_struct(config["model_config"].value.model_configs[0].gnn_kwargs, False)
        gnn_cfg = config["model_config"].value.model_configs[0]
        mlp_cfg = config["model_config"].value.model_configs[1]
        gnn_config = GnnConfig(**gnn_cfg)
        gnn_config.gnn_class = _load_class(gnn_cfg.gnn_class)
        # We add an MLP layer to process GNN output node embeddings into actions
        mlp_config = MlpConfig(**mlp_cfg)
        mlp_config.activation_class = _load_class(mlp_cfg.activation_class)
        mlp_config.layer_class = _load_class(mlp_cfg.layer_class)
        model_config = SequenceModelConfig(model_configs=[gnn_config, mlp_config], intermediate_sizes=config["model_config"].value.intermediate_sizes)
    
    experiment = Experiment(
        config=experiment_config,
        task=task,
        algorithm_config=algorithm_config,
        model_config=model_config,
        seed=0
    )
    
    return experiment