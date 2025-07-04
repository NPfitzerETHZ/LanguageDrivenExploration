import torch
from typing import Dict
from scenarios.sequential_tasks.scripts.language_grid import DEFEND_TIGHT, DEFEND_WIDE

def compute_reward(agent, env, def_type: int):

        assert DEFEND_TIGHT in env.desired_distance and DEFEND_WIDE in env.desired_distance, \
            f"def_type must contain both DEFEND_TIGHT and DEFEND_WIDE as keys"
        
        is_first = env.world.policy_agents.index(agent) == 0

        if is_first:

            # Avoid collisions with each other
            if env.agent_collision_penalty != 0:
                for a in env.world.policy_agents:
                    a.collision_rew[:] = 0

                for i, a in enumerate(env.world.agents):
                    for j, b in enumerate(env.world.agents):
                        if j <= i:
                            continue
                        collision = (
                            env.world.get_distance(a, b) <= env.min_collision_distance
                        )
                        if a.action_script is None:
                            a.collision_rew[collision] += env.collision_reward
                        if b.action_script is None:
                            b.collision_rew[collision] += env.collision_reward

        # stay close together (separation)
        agents_dist_shaping = (
            torch.stack(
                [
                    torch.linalg.vector_norm(agent.state.pos - a.state.pos, dim=-1)
                    for a in env.world.agents
                    if a != agent
                ],
                dim=1,
            )
            - env.desired_distance[def_type] * env.defend_behaviour_factor # Are we differentiating between wide vs tight defend?
        ).pow(2).mean(-1) * env.defend_dist_shaping_factor
        agent.dist_rew = agent.def_dist_shaping[def_type] - agents_dist_shaping
        agent.def_dist_shaping[def_type] = agents_dist_shaping
        
        # stay close to the target
        agent.target_distance = torch.linalg.vector_norm(
            agent.state.pos - env.flock_target,
            dim=-1,
        )
        # reward for being close to the target
        agent.dist_rew += (agent.target_distance * env.defend_dist_shaping_factor)

        return agent.collision_rew + agent.dist_rew