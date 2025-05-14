import torch

def compute_reward(agent, config):
    """
    Compute the reward for a given agent using the provided config.

    Returns:
        reward: Tensor of shape [batch_dim] for the given agent.
    """
    # === Validate Required Inputs ===
    assert hasattr(agent, "state") and hasattr(agent.state, "pos")
    assert hasattr(config, "world") and hasattr(config.world, "agents")

    is_first = agent == config.world.agents[0]
    is_last = agent == config.world.agents[-1]
    pos = agent.state.pos

    if config.n_targets > 0:
        # TODO: Refactor this to support full decentralization
        agent.num_covered_targets = config.all_time_covered_targets[
            torch.arange(0, config.world.batch_dim, device=config.device),
            config.target_class
        ].sum(dim=-1)

    # === Exponential Reward ===
    if config.use_expo_search_rew:
        config.covering_rew_val = torch.exp(
            config.exponential_search_rew * (agent.num_covered_targets + 1) / config.max_target_count
        ) + (config.covering_rew_coeff - 1)

    # === Initialize Reward Buffers ===
    reward = torch.zeros(config.world.batch_dim, device=config.world.device)
    agent.exploration_rew[:] = 0
    agent.coverage_rew[:] = 0
    agent.collision_rew[:] = 0
    agent.termination_rew[:] = 0

    # === Per-Agent Reward Components ===
    compute_collisions(agent,config)
    compute_exploration_rewards(agent, pos, config)
    compute_termination_rewards(agent, config)

    # === Team-Level Covering Rewards ===
    if is_first:
        config.shared_covering_rew[:] = 0
        compute_agent_distance_matrix(config)
        compute_covering_rewards(config)

    covering_rew = (
        agent.covering_reward if not config.shared_target_reward
        else config.shared_covering_rew
    )

    reward += agent.collision_rew + agent.termination_rew
    reward += (covering_rew + agent.exploration_rew + config.time_penalty) * (1 - agent.termination_signal)

    # === Handle Respawn Once ===
    if is_last:
        config._handle_target_respawn()

    return reward

def compute_agent_reward(agent, config):
    """
    Compute the covering reward for a specific agent.
    """
    batch_size, n_groups, _, n_targets = config.agents_targets_dists.shape
    agent_index = config.world.agents.index(agent)
    agent.covering_reward[:] = 0

    targets_covered_by_agent = (
        config.agents_targets_dists[:, :, agent_index, :] < config._covering_range  # [B, G, T]
    )
    num_covered = (
        targets_covered_by_agent * config.covered_targets  # [B, G, T]
    ).sum(dim=-1)  # [B, G]

    reward_mask = torch.arange(n_groups, device=config.target_class.device).unsqueeze(0)  # [1, G]
    reward_mask = reward_mask == config.target_class.unsqueeze(1)  # [B, G]

    group_rewards = num_covered * config.covering_rew_val.unsqueeze(1) * reward_mask  # [B, G]

    if config.target_attribute_objective or config.llm_activate:
        hinted_mask = config.occupancy_grid.searching_hinted_target.unsqueeze(1)  # [B, 1]
        group_rewards += (
            num_covered * config.false_covering_penalty_coeff * (~reward_mask) * hinted_mask
        )

    agent.covering_reward += group_rewards.sum(dim=-1)  # [B]
    return agent.covering_reward


def compute_agent_distance_matrix(config):
    """
    Compute agent-target and agent-agent distances and update related tensors in config.
    """
    config.agents_pos = torch.stack([a.state.pos for a in config.world.agents], dim=1)

    for i, targets in enumerate(config.target_groups):
        config.targets_pos[:, i, :, :] = torch.stack(
            [t.state.pos for t in targets], dim=1
        )

    config.agents_targets_dists = torch.cdist(
        config.agents_pos.unsqueeze(1),
        config.targets_pos
    )

    config.agents_covering_targets = config.agents_targets_dists < config._covering_range
    config.agents_per_target = config.agents_covering_targets.int().sum(dim=2)
    config.agent_is_covering = config.agents_covering_targets.any(dim=2)
    config.covered_targets = config.agents_per_target >= config._agents_per_target


def compute_collisions(agent, config):
    """
    Compute collision penalties for an agent against others and obstacles.
    """
    for other in config.world.agents:
        if other != agent:
            agent.collision_rew[
                config.world.get_distance(other, agent) < config.min_collision_distance
            ] += config.agent_collision_penalty

    pos = agent.state.pos
    if config.add_obstacles:
        for obstacle in config._obstacles:
            agent.collision_rew[
                config.world.get_distance(obstacle, agent) < config.min_collision_distance
            ] += config.obstacle_collision_penalty

        mask_x = (pos[:, 0] > config.x_semidim - config.agent_radius) | (pos[:, 0] < -config.x_semidim + config.agent_radius)
        mask_y = (pos[:, 1] > config.y_semidim - config.agent_radius) | (pos[:, 1] < -config.y_semidim + config.agent_radius)
        agent.collision_rew[mask_x] += config.obstacle_collision_penalty
        agent.collision_rew[mask_y] += config.obstacle_collision_penalty

def compute_covering_rewards(config):
    """
    Aggregate covering rewards for all agents into a shared reward tensor.
    """
    config.shared_covering_rew[:] = 0
    for agent in config.world.agents:
        config.shared_covering_rew += compute_agent_reward(agent, config)
    config.shared_covering_rew[config.shared_covering_rew != 0] /= 2

def compute_exploration_rewards(agent, pos, config):
    """
    Compute exploration and heading bonuses for the agent.
    """
    agent.exploration_rew += agent.occupancy_grid.compute_exploration_bonus(
        pos,
        exploration_rew_coeff=config.exploration_rew_coeff,
        new_cell_rew_coeff=config.new_cell_rew_coeff
    )

    if config.global_heading_objective or config.llm_activate:
        agent.exploration_rew += config.occupancy_grid.compute_region_heading_bonus_normalized(
            pos, heading_exploration_rew_coeff=config.heading_exploration_rew_coeff
        )
        config.occupancy_grid.update_heading_coverage_ratio()

        if config.comm_dim > 0:
            agent.coverage_rew = config.occupancy_grid.compute_coverage_ratio_bonus(
                config.coverage_action[agent.name]
            )

    config.occupancy_grid.update(pos)
    agent.occupancy_grid.update(pos)

def compute_termination_rewards(agent, config):
    """
    Compute termination reward and movement penalty after task completion.
    """
    reached_mask = agent.num_covered_targets >= config.max_target_count
    agent.termination_rew += reached_mask * (1 - agent.termination_signal) * config.terminal_rew_coeff

    if reached_mask.any():
        movement_penalty = (agent.state.vel[reached_mask] ** 2).sum(dim=-1) * config.termination_penalty_coeff
        agent.termination_rew[reached_mask] += movement_penalty
        agent.termination_signal[reached_mask] = 1.0





