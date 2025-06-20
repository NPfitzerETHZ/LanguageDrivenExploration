import torch
from typing import Union

#from deployment.my_deployment import Agent as RosAgent
#from scenarios.decentralized.agent.agent import DecentralizedAgent
#from scenarios.decentralized.decentralized_multi_agent_llm_exploration import MyLanguageScenario
#from deployment.my_deployment import VmasModelsROSInterface

def observation(agent, env):
    """
    Construct the observation vector for an agent using the provided env.

    Args:
        agent: An Agent object with required attributes:
            - state.pos: Tensor [2] for agent position
            - state.vel: Tensor [2] for agent velocity
            - sensors: Iterable of objects with .measure() -> Tensor
            - position_history / velocity_history: Objects with:
                - get_flattened(): returns history as Tensor
                - update(tensor): updates internal buffer
            - occupancy_grid: Object with methods:
                - observe_embeddings()
                - get_grid_target_observation(pos, radius)
                - get_grid_visits_obstacle_observation(pos, radius)

        env: An object with the following attributes:
            - x_semidim, y_semidim: Map half-dimensions for normalization
            - device: Target device for computation
            - use_lidar, observe_pos_history, observe_vel_history
            - llm_activate, observe_targets, max_target_objective
            - use_occupancy_grid_obs, use_expo_search_rew
            - mini_grid_radius: Radius for grid-based methods
            - max_target_count, num_covered_targets: Tensors [1]
            - use_gnn: Boolean

    Returns:
        If env.use_gnn is True:
            A dict with keys "obs", "pos", "vel"
        Else:
            A single concatenated observation tensor
    """
    # === Validation ===
    assert hasattr(agent, "occupancy_grid")
    assert hasattr(agent, "state") and hasattr(agent.state, "pos") and hasattr(agent.state, "vel") and hasattr(agent.state, "rot")
    assert hasattr(env, "x_semidim") and hasattr(env, "y_semidim") and hasattr(env, "device")

    obs_components = []
    obs_dict = {}

    # === Normalized position and velocity ===
    pos = agent.state.pos
    vel = agent.state.vel
    rot = agent.state.rot
    pos_norm = pos / torch.tensor([env.x_semidim, env.y_semidim], device=env.device)
    vel_norm = vel / torch.tensor([env.x_semidim, env.y_semidim], device=env.device)
    
    # === LLM sentence embedding ===
    if env.llm_activate:
        obs_dict["sentence_embedding"] = env.occupancy_grid.observe_embeddings()
    
    # === Histories ===
    if env.observe_pos_history:
        assert hasattr(agent, "position_history")
        pos_hist = agent.position_history.get_flattened()
        obs_components.append(pos_hist[:pos_norm.shape[0], :])
        agent.position_history.update(pos_norm)

    if env.observe_vel_history:
        assert hasattr(agent, "velocity_history")
        vel_hist = agent.velocity_history.get_flattened()
        obs_components.append(vel_hist[:vel_norm.shape[0], :])
        agent.velocity_history.update(vel_norm)
    
    # === LIDAR ===
    if env.use_lidar:
        assert hasattr(agent, "sensors")
        lidar_measures = torch.cat([sensor.measure() for sensor in agent.sensors], dim=-1)
        obs_components.append(lidar_measures)
    
    # === Exponential Search Reward / Max targets ===
    if env.use_expo_search_rew or env.use_max_targets_data:
        obs_components.append(agent.num_covered_targets.unsqueeze(1) / env.max_target_count.unsqueeze(1))

    # === Target Observation ===
    if env.observe_targets:
        obs_dict["target_obs"] = env.occupancy_grid.environment.get_grid_target_observation(pos, env.mini_grid_radius)
    
    # === Obstacle Observation ===
    obs_components.append(env.occupancy_grid.environment.get_grid_obstacle_observation(pos, env.mini_grid_radius))

    # === Occupancy Grid ===
    obs_dict["grid_obs"] = agent.occupancy_grid.get_grid_observation_2d(pos, env.mini_grid_radius * 5)

    # === Pose ===
    obs_dict["pos"] = pos_norm
    obs_dict["vel"] = vel_norm
    if env.use_kinematic_model:
        obs_dict["rot"] = rot
        
    # === Final Output ===
    obs = torch.cat([comp for comp in obs_components if comp is not None], dim=-1)
    obs_dict["obs"] = obs

    return obs_dict
