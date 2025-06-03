import torch

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
            - llm_activate, observe_targets
            - use_expo_search_rew
            - mini_grid_radius: Radius for grid-based methods
            - num_covered_targets: Tensor [1]
            - use_gnn: Boolean

    Returns:
        If env.use_gnn is True:
            A dict with keys "obs", "pos", "vel"
        Else:
            A single concatenated observation tensor
    """
    # === Validation ===
    assert hasattr(agent, "state") and hasattr(agent.state, "pos") and hasattr(agent.state, "vel")
    assert hasattr(env, "x_semidim") and hasattr(env, "y_semidim") and hasattr(env, "device")

    obs_components = []

    # === Normalized position and velocity ===
    pos = agent.state.pos
    vel = agent.state.vel
    pos_norm = pos / torch.tensor([env.x_semidim, env.y_semidim], device=env.device)
    vel_norm = vel / torch.tensor([env.x_semidim, env.y_semidim], device=env.device)

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

    # === LLM sentence embedding ===
    if env.llm_activate:
        obs_components.append(env.occupancy_grid.observe_embeddings())

    # === Target Observation ===
    if env.observe_targets:
        obs_components.append(env.occupancy_grid.get_grid_target_observation(pos, env.mini_grid_radius))

    # === Occupancy Grid ===
    obs_components.append(env.occupancy_grid.get_grid_visits_obstacle_observation(pos, env.mini_grid_radius))

    # === Exponential Search Reward / Redundant LLM Counter ===
    if env.use_expo_search_rew or env.llm_activate:
        obs_components.append(agent.num_covered_targets.unsqueeze(1))

    # === Pose ===
    if not env.use_gnn:
        
        if env.observe_agents:
            max_radius = env.max_agent_observation_radius
            horizon_dt = env.prediction_horizon_steps * env.world.dt

            # [B,2]
            agent_future_pos = agent.state.pos + agent.state.vel * horizon_dt

            other_agents = [a for a in env.world.agents if a is not agent]
            if other_agents:
                # [B,A,2]
                other_positions   = torch.stack([a.state.pos for a in other_agents],   dim=1)
                other_velocities  = torch.stack([a.state.vel for a in other_agents],   dim=1)
                other_future_pos  = other_positions + other_velocities * horizon_dt

                rel_pos      = other_future_pos - agent_future_pos.unsqueeze(1)            # [B,A,2]
                rel_pos_norm = rel_pos / torch.tensor([env.x_semidim, env.y_semidim], device=env.device)

                raw_distances = torch.norm(rel_pos_norm, dim=-1)  # [B,A]
                radius_norm = env.agent_radius / torch.tensor([env.x_semidim, env.y_semidim], device=env.device).norm()
                adjusted_distances = raw_distances - 2 * radius_norm
                adjusted_distances = torch.clamp(adjusted_distances, min=0.0, max=max_radius)
                distances = adjusted_distances

                angles = torch.atan2(rel_pos_norm[...,1], rel_pos_norm[...,0])             # [B,A]
                angles = torch.where(distances >= max_radius - 1e-6,
                                    torch.zeros_like(angles), angles)

                neighbor_polar_tensor = torch.stack([distances, angles], dim=-1)          # [B,A,2]

                # sort per batch by distance
                #sorted_idx  = torch.argsort(neighbor_polar_tensor[...,0], dim=1)           # [B,A]
                #idx_exp     = sorted_idx.unsqueeze(-1).expand(-1, -1, 2)                   # [B,A,2]
                #neighbor_polar_tensor = torch.gather(neighbor_polar_tensor, dim=1, index=idx_exp)

                # flatten to [B, A*2]
                neighbor_polar_tensor = neighbor_polar_tensor.flatten(start_dim=1)         # [B, A*2]
            else:
                B = agent.state.pos.size(0)
                neighbor_polar_tensor = torch.zeros((B, 0), device=env.device)

            obs_components.append(neighbor_polar_tensor)
        obs_components.extend([pos_norm, vel_norm])



    # === Final Output ===
    obs = torch.cat([comp for comp in obs_components if comp is not None], dim=-1)

    return {"obs": obs, "pos": pos_norm, "vel": vel_norm} if env.use_gnn else obs


def observation_torchrl(agent, env):
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
            - llm_activate, observe_targets
            - use_expo_search_rew
            - mini_grid_radius: Radius for grid-based methods
            - num_covered_targets: Tensor [1]
            - use_gnn: Boolean

    Returns:
        If env.use_gnn is True:
            A dict with keys "obs", "pos", "vel"
        Else:
            A single concatenated observation tensor
    """
    # === Validation ===
    assert hasattr(agent, "state") and hasattr(agent.state, "pos") and hasattr(agent.state, "vel") and hasattr(agent.state, "rot")
    assert hasattr(env, "x_semidim") and hasattr(env, "y_semidim") and hasattr(env, "device")
    agent_id = int(agent.name.split("_")[-1])

    obs_components = []
    obs_dict = {}

    # === Normalized position and velocity ===
    pos = agent.state.pos
    vel = agent.state.vel
    rot = agent.state.rot
    pos_norm = 1 / env.agent_radius * pos / torch.tensor([env.x_semidim, env.y_semidim], device=env.device)
    vel_norm = 1 / env.agent_radius * vel / torch.tensor([env.x_semidim, env.y_semidim], device=env.device)
        
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
    
    # # === Agent ID ===
    # one_hot = torch.nn.functional.one_hot(
    #     torch.tensor([agent_id], device=env.device),
    #     num_classes=env.n_agents
    # ).float()
    # id_tensor = one_hot.expand(env.world.batch_dim, -1)
    # obs_components.append(id_tensor)
    
    # === Target Observation ===
    if env.observe_targets:
        obs_dict["target_obs"] = env.occupancy_grid.get_grid_target_observation(pos, env.mini_grid_radius)
    
    # === Occupancy Grid ===
    #obs_dict["grid_obs"] = env.occupancy_grid.get_grid_visits_obstacle_observation_2d(pos, env.mini_grid_radius * 2)
    obs_dict["grid_obs"] = env.occupancy_grid.get_grid_visits_obstacle_observation_2d(pos, env.mini_grid_radius)

    # === Pose ===
    obs_dict["pos"] = pos_norm
    obs_dict["vel"] = vel_norm
    if env.use_kinematic_model:
        obs_dict["rot"] = rot
        
    if not env.use_gnn and env.observe_agents:
    
        max_radius = env.max_agent_observation_radius
        horizon_dt = env.prediction_horizon_steps * env.world.dt

        # [B,2]
        agent_future_pos = agent.state.pos + agent.state.vel * horizon_dt

        other_agents = [a for a in env.world.agents if a is not agent]
        if other_agents:
            # [B,A,2]
            other_positions   = torch.stack([a.state.pos for a in other_agents],   dim=1)
            other_velocities  = torch.stack([a.state.vel for a in other_agents],   dim=1)
            other_future_pos  = other_positions + other_velocities * horizon_dt

            rel_pos      = other_future_pos - agent_future_pos.unsqueeze(1)            # [B,A,2]
            rel_pos_norm = rel_pos / torch.tensor([env.x_semidim, env.y_semidim], device=env.device)

            raw_distances = torch.norm(rel_pos_norm, dim=-1)  # [B,A]
            radius_norm = env.agent_radius / torch.tensor([env.x_semidim, env.y_semidim], device=env.device).norm()
            adjusted_distances = raw_distances - 2 * radius_norm
            adjusted_distances = torch.clamp(adjusted_distances, min=0.0, max=max_radius)
            distances = adjusted_distances

            angles = torch.atan2(rel_pos_norm[...,1], rel_pos_norm[...,0])             # [B,A]
            angles = torch.where(distances >= max_radius - 1e-6,
                                torch.zeros_like(angles), angles)

            neighbor_polar_tensor = torch.stack([distances, angles], dim=-1)          # [B,A,2]

            # sort per batch by distance
            #sorted_idx  = torch.argsort(neighbor_polar_tensor[...,0], dim=1)           # [B,A]
            #idx_exp     = sorted_idx.unsqueeze(-1).expand(-1, -1, 2)                   # [B,A,2]
            #neighbor_polar_tensor = torch.gather(neighbor_polar_tensor, dim=1, index=idx_exp)

            # flatten to [B, A*2]
            neighbor_polar_tensor = neighbor_polar_tensor.flatten(start_dim=1)         # [B, A*2]
        else:
            B = agent.state.pos.size(0)
            neighbor_polar_tensor = torch.zeros((B, 0), device=env.device)

            obs_components.append(neighbor_polar_tensor)
        
    # === Final Output ===
    obs = torch.cat([comp for comp in obs_components if comp is not None], dim=-1)
    obs_dict["obs"] = obs

    return obs_dict