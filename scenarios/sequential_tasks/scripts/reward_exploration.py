import torch

def compute_reward(agent, env):
    """
    Compute the reward for a given agent using the provided env.

    Returns:
        reward: Tensor of shape [batch_dim] for the given agent.
    """
    # === Validate Required Inputs ===
    assert hasattr(agent, "state") and hasattr(agent.state, "pos")
    assert hasattr(env, "world") and hasattr(env.world, "agents")

    is_first = agent == env.world.agents[0]
    is_last = agent == env.world.agents[-1]
    pos = agent.state.pos

    if env.n_targets > 0:
        agent.num_covered_targets = env.all_time_agent_covered_targets[
            torch.arange(0, env.world.batch_dim, device=env.device),
            env.target_class
        ].sum(dim=-1)

    # === Exponential Reward ===
    if env.use_expo_search_rew:
        
        team_coverage = env.all_time_covered_targets[
            torch.arange(0, env.world.batch_dim, device=env.device),
            env.target_class
        ].sum(dim=-1)
        
        env.covering_rew_val = torch.exp(
            env.exponential_search_rew * (team_coverage + 1) / 1
        ) + (env.covering_rew_coeff - 1)

    # === Initialize Reward Buffers ===
    reward = torch.zeros(env.world.batch_dim, device=env.world.device)
    agent.exploration_rew[:] = 0
    agent.coverage_rew[:] = 0
    agent.collision_rew[:] = 0
    agent.termination_rew[:] = 0

    # === Per-Agent Reward Components ===
    compute_collisions(agent,env)
    compute_exploration_rewards(agent, pos, env)
    compute_termination_rewards(agent, env)

    # === Team-Level Covering Rewards ===
    if is_first:
        env.shared_covering_rew[:] = 0
        compute_agent_distance_matrix(env)
        compute_covering_rewards(env)

    covering_rew = (
        agent.covering_reward if not env.shared_target_reward
        else env.shared_covering_rew
    )

    reward += agent.collision_rew + agent.termination_rew
    reward += (covering_rew + agent.exploration_rew + env.time_penalty) * (1 - agent.termination_signal)

    # === Handle Respawn Once ===
    if is_last:
        env._handle_target_respawn()

    return reward

def compute_agent_reward(agent, env):
    """
    Compute the covering reward for a specific agent.
    """
    _, n_groups, _, _ = env.agents_covering_targets.shape
    agent_index = env.world.agents.index(agent)
    agent.covering_reward[:] = 0

    targets_covered_by_agent = (
        env.agents_covering_targets[:,:,agent_index,:]  # [B, G, T]
    )
    num_covered = (
        targets_covered_by_agent * env.covered_targets  # [B, G, T]
    ).sum(dim=-1)  # [B, G]

    reward_mask = torch.arange(n_groups, device=env.target_class.device).unsqueeze(0)  # [1, G]
    reward_mask = reward_mask == env.target_class.unsqueeze(1)  # [B, G]

    group_rewards = num_covered * env.covering_rew_val.unsqueeze(1) * reward_mask  # [B, G]

    if env.llm_activate:
        hinted_mask = env.occupancy_grid.searching_hinted_target.unsqueeze(1)  # [B, 1]
        group_rewards += (
            num_covered * env.false_covering_penalty_coeff * (~reward_mask) * hinted_mask
        )

    agent.covering_reward += group_rewards.sum(dim=-1)  # [B]
    return agent.covering_reward

def compute_agent_distance_matrix(env):
    """
    Compute agent-target and agent-agent distances and update related tensors in env.
    """
    env.agents_pos = torch.stack([a.state.pos for a in env.world.agents], dim=1)
    delta = torch.abs(env.agents_pos.unsqueeze(1).unsqueeze(3) - env.targets_pos.unsqueeze(2))
    hx = env.occupancy_grid.cell_size_x  / 2   # half-width  in x
    hy = env.occupancy_grid.cell_size_y / 2   # half-height in y
    env.agents_covering_targets = (delta[..., 0] <= hx) & (delta[..., 1] <= hy)
    
    env.agents_per_target = env.agents_covering_targets.int().sum(dim=2)
    env.agent_is_covering = env.agents_covering_targets.any(dim=2)
    env.covered_targets = env.agents_per_target >= env._agents_per_target

def compute_collisions(agent, env):
    """
    Compute collision penalties for an agent against others and obstacles.
    """
    for other in env.world.agents:
        if other != agent:
            agent.collision_rew[
                env.world.get_distance(other, agent) < env.min_collision_distance
            ] += env.agent_collision_penalty

    pos = agent.state.pos
    for obstacle in env._obstacles:
        agent.collision_rew[
            env.world.get_distance(obstacle, agent) < env.min_collision_distance
        ] += env.obstacle_collision_penalty

    mask_x = (pos[:, 0] > env.x_semidim - env.agent_radius) | (pos[:, 0] < -env.x_semidim + env.agent_radius)
    mask_y = (pos[:, 1] > env.y_semidim - env.agent_radius) | (pos[:, 1] < -env.y_semidim + env.agent_radius)
    agent.collision_rew[mask_x] += env.obstacle_collision_penalty
    agent.collision_rew[mask_y] += env.obstacle_collision_penalty

def compute_covering_rewards(env):
    """
    Aggregate covering rewards for all agents into a shared reward tensor.
    """
    env.shared_covering_rew[:] = 0
    for agent in env.world.agents:
        env.shared_covering_rew += compute_agent_reward(agent, env)
    env.shared_covering_rew[env.shared_covering_rew != 0] /= 2

def compute_exploration_rewards(agent, pos: torch.Tensor, env):
    """
    Compute exploration and heading bonuses for the agent.
    """
    # Shared exploration reward - Agents must learn to communicate to maximize exploration
    agent.exploration_rew += env.occupancy_grid.internal_grid.compute_exploration_bonus(
        pos,
        exploration_rew_coeff=env.exploration_rew_coeff,
        new_cell_rew_coeff=env.new_cell_rew_coeff
    )

    if env.llm_activate:
        agent.exploration_rew += env.occupancy_grid.compute_region_heading_bonus_normalized(
            pos, heading_exploration_rew_coeff=env.heading_exploration_rew_coeff
        )
        env.occupancy_grid.update_heading_coverage_ratio()

        if env.comm_dim > 0:
            agent.coverage_rew = env.occupancy_grid.compute_coverage_ratio_bonus(
                env.coverage_action[agent.name]
            )

    grid_targets = env.occupancy_grid.environment.grid_targets
    env.occupancy_grid.internal_grid.update(pos, env.mini_grid_radius, grid_targets)

def compute_termination_rewards(agent, env):
    """
    Compute termination reward and movement penalty after task completion.
    """
    reached_mask = agent.num_covered_targets >= 1
    agent.termination_rew += reached_mask * (1 - agent.termination_signal) * env.terminal_rew_coeff

    if reached_mask.any():
        movement_penalty = (agent.state.vel[reached_mask] ** 2).sum(dim=-1) * env.termination_penalty_coeff
        agent.termination_rew[reached_mask] += movement_penalty
        agent.termination_signal[reached_mask] = 1.0





