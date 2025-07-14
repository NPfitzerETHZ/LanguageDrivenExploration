import torch
def compute_reward(agent, env):
    is_first = agent == env.world.agents[0]
    
    nav_pos_rew = torch.zeros((env.world.batch_dim,), device=env.device) 
    nav_final_rew = torch.zeros((env.world.batch_dim,), device=env.device)

    if is_first:
            
        for a in env.world.agents:
            nav_pos_rew += agent_reward(env,a)
            a.collision_rew[:] = 0

        env.all_base_reached = torch.all(
            torch.stack([a.on_base for a in env.world.agents], dim=-1),
            dim=-1,
        )

        nav_final_rew[env.all_base_reached] = env.nav_final_reward

        for i, a in enumerate(env.world.agents):
            for j, b in enumerate(env.world.agents):
                if i <= j:
                    continue
                if env.world.collides(a, b):
                    distance = env.world.get_distance(a, b)
                    a.collision_rew[
                        distance <= env.min_collision_distance
                    ] += env.agent_collision_penalty
                    b.collision_rew[
                        distance <= env.min_collision_distance
                    ] += env.agent_collision_penalty

    pos_reward = nav_pos_rew if env.nav_shared_rew else agent.nav_pos_rew
    return pos_reward + nav_final_rew + agent.collision_rew


def agent_reward(env, agent):
    agent.distance_to_base = torch.linalg.vector_norm(
        agent.state.pos - env.base.state.pos,
        dim=-1,
    )
    agent.on_base = agent.distance_to_base < env.base.shape.width / 2

    pos_shaping = agent.distance_to_base * env.nav_pos_shaping_factor
    agent.nav_pos_rew = agent.nav_pos_shaping - pos_shaping
    agent.nav_pos_shaping = pos_shaping
    return agent.nav_pos_rew