# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import time

import hydra
import torch
import sys
import os
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torchrl._utils import logger as torchrl_logger
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import RewardSum, TransformedEnv
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from models.models import MultiAgentGNN, SimpleConcatCritic
from torchrl.objectives import ClipPPOLoss, ValueEstimators

from torch_geometric.nn import GATv2Conv, GCNConv, GraphConv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.logging import init_logging, log_evaluation, log_training
from utils.utils import DoneTransform

from scenarios.centralized.multi_agent_llm_exploration import MyLanguageScenario
from omegaconf import DictConfig


def rendering_callback(env, td):
    env.frames.append(env.render(mode="rgb_array", agent_index_focus=None))

def save_checkpoint(model, optimizer, iteration, total_frames, path="torch_rl_checkpoint.pth"):
    dir_path = os.path.dirname(os.path.abspath(path))
    os.makedirs(dir_path, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
        "total_frames": total_frames,
    }
    torch.save(checkpoint, path)
    print(f"Saved checkpoint at iteration {iteration} to {path}")
    
@hydra.main(version_base="1.1", config_path="/Users/nicolaspfitzer/ProrokLab/CustomScenarios/configs", config_name="torchrl_mappo")
def train(cfg: DictConfig):  # noqa: F821
    # Device
    cfg.train.device = "cpu" if not torch.cuda.device_count() else "cuda:0"
    cfg.env.device = cfg.train.device

    # Seeding
    torch.manual_seed(cfg.seed)

    # Sampling
    cfg.env.vmas_envs = cfg.collector.frames_per_batch // cfg.env.max_steps
    cfg.collector.total_frames = cfg.collector.frames_per_batch * cfg.collector.n_iters
    cfg.buffer.memory_size = cfg.collector.frames_per_batch
    
    comms_radius = 0.5
    use_gnn = True
    kwargs = {
        # === Map & Scenario Layout ===
        "x_semidim": 1.0,
        "y_semidim": 1.0,
        "covering_range": 0.15,
        "agent_radius": 0.03,
        "n_obstacles": 5,

        # === Agent/Target Counts & Behavior ===
        "n_agents": 3,
        "agents_per_target": 1,
        "n_targets_per_class": 4,
        "n_target_classes": 1,
        "n_targets": 4,  # 4 per class * 1 class
        "done_at_termination": True,

        # === Rewards ===
        "shared_target_reward": False,
        "shared_final_reward": True,
        "agent_collision_penalty": -0.5,
        "obstacle_collision_penalty": -0.5,
        "covering_rew_coeff": 5.0,
        "false_covering_penalty_coeff": -0.25,
        "time_penalty": -0.05,
        "terminal_rew_coeff": 0.0,
        "exponential_search_rew_coeff": 1.5,
        "termination_penalty_coeff": -5.0,

        # === Exploration Rewards ===
        "use_expo_search_rew": True,
        "grid_visit_threshold": 2,
        "exploration_rew_coeff": -0.00,
        "new_cell_rew_coeff": 0.05,
        "heading_exploration_rew_coeff": 30, #30,

        # === Lidar & Sensing ===
        "use_lidar": False,
        "n_lidar_rays_entities": 8,
        "n_lidar_rays_agents": 12,
        "use_velocity_controller": True,
        "max_agent_observation_radius": 0.3,
        "prediction_horizon_steps": 1,

        # === Agent Communication & GNNs ===
        "use_gnn": use_gnn,
        "comm_dim": 0,
        "comms_radius": comms_radius,

        # === Observation Settings ===
        "observe_grid": True,
        "observe_targets": True,
        "observe_agents": False,
        "observe_pos_history": False,
        "observe_vel_history": False,
        "use_grid_data": True,
        "use_class_data": False,
        "use_max_targets_data": False,
        "use_confidence_data": False,

        # === Grid Settings ===
        "num_grid_cells": 400,
        "mini_grid_radius": 1,

        # === Movement & Dynamics ===
        "agent_weight": 1.0,
        "agent_v_range": 0.3,
        "agent_a_range": 0.3,
        "min_collision_distance": 0.03,
        "linear_friction": 0.1,

        # === Histories ===
        "history_length": 0,
        
        # === Language & LLM Goals ===
        "embedding_size": 1024,
        "llm_activate": True,

        # === External Inputs ===
        "data_json_path": "data/language_data_complete_multi_target_color_medium.json",
        "decoder_model_path": "decoders/llm0_decoder_model_grid_single_target_color.pth",
        "use_decoder": False,

        # === Visuals ===
        "viewer_zoom": 1,
    }
    
    # Create env and env_test
    env = VmasEnv(
        scenario=MyLanguageScenario(),
        num_envs=cfg.env.vmas_envs,
        continuous_actions=True,
        max_steps=cfg.env.max_steps,
        device=cfg.env.device,
        seed=cfg.seed,
        # Scenario kwargs
        **kwargs,
    )
    env = TransformedEnv(
        env,
        RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")]),
    )

    env_test = VmasEnv(
        scenario=MyLanguageScenario(),
        num_envs=cfg.eval.evaluation_episodes,
        continuous_actions=True,
        max_steps=cfg.env.max_steps,
        device=cfg.env.device,
        seed=cfg.seed,
        # Scenario kwargs
        **kwargs,
    )
    
    # Policy
    # actor_net = nn.Sequential(
    #     MultiAgentGNN(
    #         n_agents=env.n_agents,
    #         node_input_dim=env.observation_spec["agents", "observation","obs"].shape[-1],
    #         action_dim=env.action_spec.shape[-1],
    #         sentence_dim=cfg.model.embedding_size,
    #         topology="full", # from_pos: Tell the GNN to build topology from positions and edge_radius, full: Fully connected topology
    #         edge_radius=comms_radius, # The edge radius for the topology
    #         n_gnn_layers = 1,
    #         self_loops=True,
    #         gnn_class=GATv2Conv,
    #         gnn_kwargs={"add_self_loops": True, "residual": True}, # kwargs of GATv2Conv, residual is helpf>
    #         position_key="pos",
    #         pos_features=2,
    #         velocity_key="vel",
    #         vel_features=2,
    #         exclude_pos_from_node_features=False,
    #         sentence_key= "sentence_embedding",
    #         device=cfg.train.device,
    #         ),
    #         NormalParamExtractor()
    # )
    
    actor_net = nn.Sequential(
        MultiAgentGNN(
            n_agents=env.n_agents,
            node_input_dim=env.observation_spec["agents", "observation","obs"].shape[-1],
            action_dim=env.action_spec.shape[-1],
            sentence_dim=cfg.model.embedding_size,
            topology="from_pos", # from_pos: Tell the GNN to build topology from positions and edge_radius, full: Fully connected topology
            edge_radius=comms_radius, # The edge radius for the topology
            n_gnn_layers = 1,
            self_loops=True,
            gnn_class=GraphConv,
            position_key="pos",
            pos_features=2,
            velocity_key="vel",
            vel_features=2,
            exclude_pos_from_node_features=False,
            sentence_key= "sentence_embedding",
            device=cfg.train.device,
            ),
            NormalParamExtractor()
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

    # Critic
    module = SimpleConcatCritic(
        n_agents=env.n_agents,
        node_input_dim=env.observation_spec["agents", "observation","obs"].shape[-1],
        sentence_key="sentence_embedding",
        sentence_dim=cfg.model.embedding_size,
        hidden_dim=256,
        device=cfg.train.device,
    )
    
    value_module = ValueOperator(
        module=module,
        in_keys=[("agents", "observation")],
    )

    collector = SyncDataCollector(
        env,
        policy,
        device=cfg.env.device,
        storing_device=cfg.train.device,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
        postproc=DoneTransform(reward_key=env.reward_key, done_keys=env.done_keys),
    )

    replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(cfg.buffer.memory_size, device=cfg.train.device),
        sampler=SamplerWithoutReplacement(),
        batch_size=cfg.train.minibatch_size,
    )

    # Loss
    loss_module = ClipPPOLoss(
        actor_network=policy,
        critic_network=value_module,
        clip_epsilon=cfg.loss.clip_epsilon,
        entropy_coef=cfg.loss.entropy_coef,
        normalize_advantage=False,
    )
    loss_module.set_keys(
        reward=env.reward_key,
        action=env.action_key,
        done=("agents", "done"),
        terminated=("agents", "terminated"),
    )
    loss_module.make_value_estimator(
        ValueEstimators.GAE, gamma=cfg.loss.gamma, lmbda=cfg.loss.lmbda
    )
    optim = torch.optim.Adam(loss_module.parameters(), cfg.train.lr)

    # Logging
    if cfg.logger.backend:
        model_name = (
            ("Het" if not cfg.model.shared_parameters else "")
            + ("MA" if cfg.model.centralised_critic else "I")
            + "PPO"
        )
        logger = init_logging(cfg, model_name)

    total_time = 0
    total_frames = 0
    sampling_start = time.time()
    for i, tensordict_data in enumerate(collector):
        torchrl_logger.info(f"\nIteration {i}")

        sampling_time = time.time() - sampling_start

        with torch.no_grad():
            loss_module.value_estimator(
                tensordict_data,
                params=loss_module.critic_network_params,
                target_params=loss_module.target_critic_network_params,
            )
        current_frames = tensordict_data.numel()
        total_frames += current_frames
        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view)

        training_tds = []
        training_start = time.time()
        for _ in range(cfg.train.num_epochs):
            for _ in range(cfg.collector.frames_per_batch // cfg.train.minibatch_size):
                subdata = replay_buffer.sample()
                loss_vals = loss_module(subdata)
                training_tds.append(loss_vals.detach())

                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )

                loss_value.backward()

                total_norm = torch.nn.utils.clip_grad_norm_(
                    loss_module.parameters(), cfg.train.max_grad_norm
                )
                training_tds[-1].set("grad_norm", total_norm.mean())

                optim.step()
                optim.zero_grad()

        collector.update_policy_weights_()

        training_time = time.time() - training_start

        iteration_time = sampling_time + training_time
        total_time += iteration_time
        training_tds = torch.stack(training_tds)

        # More logs
        if cfg.logger.backend:
            log_training(
                logger,
                training_tds,
                tensordict_data,
                sampling_time,
                training_time,
                total_time,
                i,
                current_frames,
                total_frames,
                step=i,
            )

        if (
            cfg.eval.evaluation_episodes > 0
            and i % cfg.eval.evaluation_interval == 0
            and cfg.logger.backend
        ):
            evaluation_start = time.time()
            with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
                env_test.frames = []
                rollouts = env_test.rollout(
                    max_steps=cfg.env.max_steps,
                    policy=policy,
                    callback=rendering_callback,
                    auto_cast_to_device=True,
                    break_when_any_done=False,
                    # We are running vectorized evaluation we do not want it to stop when just one env is done
                )

                evaluation_time = time.time() - evaluation_start

                log_evaluation(logger, rollouts, env_test, evaluation_time, step=i)

        if cfg.logger.backend == "wandb":
            logger.experiment.log({}, commit=True)
        sampling_start = time.time()
        
        if i % cfg.train.checkpoint_interval == 0:
            checkpoint_path = os.path.join(cfg.train.checkpoint_dir, f"torch_rl_checkpoint_{i}.pth")
            save_checkpoint(loss_module, optim, i, total_frames, checkpoint_path)
            
    collector.shutdown()
    if not env.is_closed:
        env.close()
    if not env_test.is_closed:
        env_test.close()


if __name__ == "__main__":
    train()