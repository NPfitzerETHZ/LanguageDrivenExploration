from omegaconf import DictConfig
from pathlib import Path
from benchmarl.environments import VmasTask
from torchrl.envs import VmasEnv
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.algorithms import MappoConfig
from hydra.utils import instantiate
from utils.utils import _load_class
import importlib

def _patch_env_creator(scenario_cls_path: str):
    """Monkey-patch VmasTask.get_env_fun so we can pass a dotted class path."""
    module_path, cls_name = scenario_cls_path.rsplit(".", 1)
    ScenarioCls = getattr(importlib.import_module(module_path), cls_name)

    def get_env_fun(self, num_envs, continuous_actions, seed, device):
        # clone the config dict to avoid side-effects
        cfg = dict(self.config)
        return lambda: VmasEnv(
            scenario=ScenarioCls(),  # instantiate custom scenario
            num_envs=num_envs,
            continuous_actions=continuous_actions,
            seed=seed,
            device=device,
            categorical_actions=True,
            clamp_actions=True,
            **cfg,
        )

    VmasTask.get_env_fun = get_env_fun
    
def benchmarl_setup_experiment(cfg: DictConfig) -> Experiment:
    
    # ---------- TASK ----------
    task_enum = VmasTask[cfg.task.name]
    task = task_enum.get_from_yaml()              # base defaults
    task.config = cfg.task.params          # override with YAML params

    _patch_env_creator(cfg.task.scenario_class)   # custom scenario

    # ---------- ALGORITHM ----------
    if cfg.algorithm.type.lower() == "mappo":
        algorithm_config = MappoConfig.get_from_yaml()
    else:
        raise ValueError("Only MAPPO implemented here")

    algorithm_config.entropy_coef = cfg.algorithm.params.entropy_coef

    # ----------- MODELS ----------
    actor_model = instantiate(cfg.model.actor_model)
    critic_model = instantiate(cfg.model.critic_model)

    # Load class objects dynamically
    for model in [actor_model, critic_model]:
        model.activation_class = _load_class(model.activation_class)
        model.layer_class = _load_class(model.layer_class)

    # Conditionally load GNN class if used
    if getattr(actor_model, "use_gnn", False):
        actor_model.gnn_class = _load_class(actor_model.gnn_class)
    else:
        actor_model.gnn_class = None

    # ---------- EXPERIMENT ----------
    exp_cfg = ExperimentConfig(**cfg.experiment)
    exp_cfg.save_folder = Path(__file__).parent / "experiments"
    exp_cfg.save_folder.mkdir(exist_ok=True, parents=True)

    return Experiment(
        task=task,
        algorithm_config=algorithm_config,
        model_config=actor_model,
        critic_model_config=critic_model,
        seed=cfg.seed,
        config=exp_cfg,
    )