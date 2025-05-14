import torch

def observation(agent, config):
    """
    Construct the observation vector for an agent using the provided config.

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

        config: An object with the following attributes:
            - x_semidim, y_semidim: Map half-dimensions for normalization
            - device: Target device for computation
            - use_lidar, observe_pos_history, observe_vel_history
            - llm_activate, observe_targets, max_target_objective
            - use_occupancy_grid_obs, use_expo_search_rew
            - mini_grid_radius: Radius for grid-based methods
            - max_target_count, num_covered_targets: Tensors [1]
            - use_gnn: Boolean

    Returns:
        If config.use_gnn is True:
            A dict with keys "obs", "pos", "vel"
        Else:
            A single concatenated observation tensor
    """
    # === Validation ===
    assert hasattr(agent, "state") and hasattr(agent.state, "pos") and hasattr(agent.state, "vel")
    assert hasattr(agent, "occupancy_grid")
    assert hasattr(config, "x_semidim") and hasattr(config, "y_semidim") and hasattr(config, "device")

    obs_components = []

    # === Normalize position and velocity ===
    pos = agent.state.pos / torch.tensor([config.x_semidim, config.y_semidim], device=config.device)
    vel = agent.state.vel / torch.tensor([2 * config.x_semidim, 2 * config.y_semidim], device=config.device)

    # === Histories ===
    if config.observe_pos_history:
        assert hasattr(agent, "position_history")
        pos_hist = agent.position_history.get_flattened()
        obs_components.append(pos_hist[:pos.shape[0], :])
        agent.position_history.update(pos)

    if config.observe_vel_history:
        assert hasattr(agent, "velocity_history")
        vel_hist = agent.velocity_history.get_flattened()
        obs_components.append(vel_hist[:vel.shape[0], :])
        agent.velocity_history.update(vel)

    # === LIDAR ===
    if config.use_lidar:
        assert hasattr(agent, "sensors")
        lidar_measures = torch.cat([sensor.measure() for sensor in agent.sensors], dim=-1)
        obs_components.append(lidar_measures)

    # === LLM sentence embedding ===
    if config.llm_activate:
        obs_components.append(agent.occupancy_grid.observe_embeddings())

    # === Target Observation ===
    if config.observe_targets:
        obs_components.append(agent.occupancy_grid.get_grid_target_observation(pos, config.mini_grid_radius))

    # === Max target counts ===
    if config.max_target_objective:
        obs_components.extend([
            config.max_target_count.unsqueeze(1),
            config.num_covered_targets.unsqueeze(1)
        ])

    # === Occupancy Grid ===
    if config.use_occupancy_grid_obs:
        obs_components.append(agent.occupancy_grid.get_grid_visits_obstacle_observation(pos, config.mini_grid_radius))

    # === Exponential Search Reward / Redundant LLM Counter ===
    if config.use_expo_search_rew or config.llm_activate:
        obs_components.append(config.num_covered_targets.unsqueeze(1))

    # === Pose ===
    if not config.use_gnn:
        obs_components.extend([pos, vel])

    # === Final Output ===
    obs = torch.cat([comp for comp in obs_components if comp is not None], dim=-1)

    return {"obs": obs, "pos": pos, "vel": vel} if config.use_gnn else obs
