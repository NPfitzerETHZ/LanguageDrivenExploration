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
    vel_norm = vel / torch.tensor([2 * env.x_semidim, 2 * env.y_semidim], device=env.device)

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
        
        # Collect neighbor distances and angles
        neighbor_polar = []
        max_radius = 0.3
        for idx, other_agent in enumerate(env.world.agents):
            if other_agent is agent:
                continue
            rel_pos = other_agent.state.pos - agent.state.pos
            rel_pos_norm = rel_pos / torch.tensor([env.x_semidim, env.y_semidim], device=env.device)
            distance = torch.norm(rel_pos_norm,dim=-1)
            distance = torch.clamp(distance, max=max_radius)
            angle = torch.atan2(rel_pos_norm[:,1], rel_pos_norm[:,0])
            angle = torch.where(distance >= max_radius - 1e-6, torch.tensor(0.0, device=env.device), angle)
            neighbor_polar.append(torch.stack([distance, angle],dim=-1))

        if neighbor_polar:
            neighbor_polar_tensor = torch.cat(neighbor_polar, dim=-1)
        else:
            neighbor_polar_tensor = torch.zeros(0, device=env.device)

        obs_components.append(neighbor_polar_tensor)
        obs_components.extend([pos_norm, vel_norm])

    # === Final Output ===
    obs = torch.cat([comp for comp in obs_components if comp is not None], dim=-1)

    return {"obs": obs, "pos": pos_norm, "vel": vel_norm} if env.use_gnn else obs
