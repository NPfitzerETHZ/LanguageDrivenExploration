# eval.py

from __future__ import annotations

import os
import sys
import torch
import hydra
from omegaconf import DictConfig

from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torchrl.envs import TransformedEnv, RewardSum
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs.utils import set_exploration_type, ExplorationType
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.modules.models.multiagent import MultiAgentMLP
from tensordict import TensorDict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.utils import DoneTransform
from scenarios.simple_language_deployment_scenario import MyLanguageScenario

def load_checkpoint(model, path="checkpoint.pth"):
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from {path} at iteration {checkpoint['iteration']}")
    return checkpoint

def compute_action(policy, observation, device="cpu"):
    """
    Computes the action for a single observation using the trained policy.
    
    Args:
        policy: the trained ProbabilisticActor
        observation: a tensor of shape [obs_dim] or [batch_size, obs_dim]
        device: device to move observation to (default: 'cpu')
    
    Returns:
        action: tensor of actions
    """
    if observation.ndim == 1:
        observation = observation.unsqueeze(0)  # make it batch size 1
    observation = observation.to(device)

    obs_td = TensorDict({("agents", "observation"): observation}, batch_size=[observation.shape[0]], device=device)

    with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
        action_td = policy(obs_td)

    action = action_td[("agents", "action")]
    return action.squeeze(0)  # remove batch dimension if single input

@hydra.main(version_base="1.1", config_path="", config_name="mappo_ippo")
def evaluate(cfg: DictConfig):  # noqa: F821
    # Device setup
    cfg.train.device = "cpu" if not torch.cuda.device_count() else "cuda:0"
    cfg.env.device = cfg.train.device

    torch.manual_seed(cfg.seed)

    # Env setup kwargs (same as training)
    comms_radius = 0.3
    use_gnn = False
    embedding_size = 1024
    encoder_depth = 2
    latent_dim = 64
    mini_grid_radius = 1
    n_target_classes = 0

    kwargs = {
        "n_agents": 1,
        "n_targets_per_class": 0,
        "n_target_classes": n_target_classes,
        "comms_radius": comms_radius,
        "use_gnn": use_gnn,
        "n_obstacles": 0,
        "global_heading_objective": False,
        "num_grid_cells": 400,
        "data_json_path": 'data/language_data_complete_single_target_color_medium.json',
        "decoder_model_path": 'decoders/llm0_decoder_model_grid_single_target_color.pth',
        "use_decoder": False,
        "use_grid_data": True,
        "use_class_data": False,
        "use_max_targets": False,
        "mini_grid_radius": mini_grid_radius
    }

    env = VmasEnv(
        scenario=MyLanguageScenario(),
        num_envs=cfg.eval.evaluation_episodes,
        continuous_actions=True,
        max_steps=cfg.env.max_steps,
        device=cfg.env.device,
        seed=cfg.seed,
        **kwargs,
    )

    env = TransformedEnv(
        env,
        RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")]),
    )

    # Build the policy (actor network)
    actor_net = nn.Sequential(
        MultiAgentMLP(
            n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
            n_agent_outputs=2 * env.action_spec.shape[-1],
            n_agents=env.n_agents,
            centralised=False,
            share_params=cfg.model.shared_parameters,
            device=cfg.train.device,
            depth=2,
            num_cells=256,
            activation_class=nn.Tanh
        ),
        NormalParamExtractor(),
    )

    policy_module = TensorDictModule(
        actor_net,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "loc"), ("agents", "scale")],
    )

    policy = ProbabilisticActor(
        module=policy_module,
        spec=env.full_action_spec_unbatched,
        in_keys=[("agents", "loc"), ("agents", "scale")],
        out_keys=[env.action_key],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": env.full_action_spec_unbatched[("agents", "action")].space.low,
            "high": env.full_action_spec_unbatched[("agents", "action")].space.high,
        },
        return_log_prob=True,
    )

    # Load checkpoint
    checkpoint_path = cfg.eval.checkpoint_path
    load_checkpoint(policy, checkpoint_path)

    # Evaluation rollouts
    policy.eval()
    env.frames = []

    with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
        rollouts = env.rollout(
            max_steps=cfg.env.max_steps,
            policy=policy,
            auto_cast_to_device=True,
            break_when_any_done=False,
        )

    rewards = rollouts.get(("agents", "episode_reward"))
    mean_reward = rewards.mean().item()
    print(f"Evaluation finished, mean episode reward: {mean_reward:.2f}")

    # Example of compute_action usage:
    dummy_obs = torch.randn(env.observation_spec["agents", "observation"].shape[-1])
    action = compute_action(policy, dummy_obs, device=cfg.train.device)
    print(f"Action for a random dummy observation: {action}")

    if not env.is_closed:
        env.close()

if __name__ == "__main__":
    evaluate()
