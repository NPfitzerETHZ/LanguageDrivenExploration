import typing
from typing import Callable, Dict, List

import torch
from torch import Tensor
import random

from vmas import render_interactively
from vmas.simulator.core import Agent, Entity, Landmark, Sphere, Box, World
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.sensors import Lidar
from vmas.simulator.utils import Color, ScenarioUtils, X, Y

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom

import torch
from histories import VelocityHistory, PositionHistory, JointPosHistory
from occupency_grid import OccupancyGrid
from rewards import CountBasedReward, EntropyBasedReward, JointEntropyBasedReward

class MyScenario(BaseScenario):

    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        """Initialize the world and entities."""
        self.device = device
        self._load_config(kwargs)
        world = self._create_world(batch_dim)
        self._create_agents(world, batch_dim)
        self._create_targets(world)
        self._create_obstacles(world)
        self._create_global_heading_vis(world)
        self._initialize_rewards(batch_dim)
        self._extras()
        return world
    
    def _load_config(self,kwargs):

        self.num_envs = kwargs.pop("num_envs", 96)
        self.n_agents = kwargs.pop("n_agents", 5)
        self.n_targets = kwargs.pop("n_targets", 5)
        self.n_obstacles = kwargs.pop("n_obstacle",5)
        self.x_semidim = kwargs.pop("x_semidim", 1.0)
        self.y_semidim = kwargs.pop("y_semidim", 1.0)
        self._min_dist_between_entities = kwargs.pop("min_dist_between_entities", 0.2)
        self._lidar_range = kwargs.pop("lidar_range", 0.35)
        self._covering_range = kwargs.pop("covering_range", 0.15)

        self.use_agent_lidar = kwargs.pop("use_agent_lidar", False)
        self.use_obstacle_lidar = kwargs.pop("use_obstacle_lidar", True)
        self.n_lidar_rays_entities = kwargs.pop("n_lidar_rays_entities", 15)
        self.n_lidar_rays_agents = kwargs.pop("n_lidar_rays_agents", 12)

        self._agents_per_target = kwargs.pop("agents_per_target", 1)
        self.agents_stay_at_target = kwargs.pop("agents_stay_at_target", False)
        self.targets_respawn = kwargs.pop("targets_respawn", False)
        self.shared_reward = kwargs.pop("shared_reward", False)  # Reward for finding a target
        self.shared_final_reward = kwargs.pop("shared_final_reward", True) # Reward for finding all targets, if targets_respawn false

        self.add_obstacles = kwargs.pop("add_obstacles", True)

        # Novelty rewards
        self.use_count_rew = kwargs.pop("use_count_rew", False)
        self.use_entropy_rew = kwargs.pop("use_entropy_rew", False)
        self.use_jointentropy_rew = kwargs.pop("use_jointentropy_rew", False)

        self.known_map = kwargs.pop("known_map", False)
        self.known_agent_pos = kwargs.pop("known_agents",False) # Not ure about this one. Do I need a GNN?

        self.agent_collision_penalty = kwargs.pop("agent_collision_penalty", -0.0)
        self.obstacle_collision_penalty = kwargs.pop("obstacle_collision_penalty", -0.25)
        self.covering_rew_coeff = kwargs.pop("covering_rew_coeff", 5.0) # Large reward for finding a target
        self.time_penalty = kwargs.pop("time_penalty", 0)
        ScenarioUtils.check_kwargs_consumed(kwargs)

        self._comms_range = self._lidar_range*0.001
        self.min_collision_distance = 0.005
        self.agent_radius = 0.05
        self.target_radius = self.agent_radius

        self.viewer_zoom = 1
        self.target_color = Color.GREEN
        self.obstacle_color = Color.BLUE

        # Histories
        self.pos_history_length = 30
        self.pos_dim = 2
        self.observe_pos_history = kwargs.pop("observe_pos_history", False)
        self.observe_jointpos_history = kwargs.pop("observe_jointpos_history", False)

        self.vel_history_length = 30
        self.vel_dim = 2
        self.observe_vel_history = kwargs.pop("observe_vel_history", False)

        #===================
        # Language Driven Goals
        # 1) Count
        self.max_target_objective = kwargs.pop("max_target_objective", False) # Enter as fraction of total number of targets
        # 2) Heading
        self.global_heading_objective = kwargs.pop("global_heading_objective", False)
        self.location_radius = kwargs.pop("location:radius", 0.5) 
        # 3) Attribute
        self.target_attribute_objective = kwargs.pop("target_attribute_objective", True)
        #===================

    def _create_world(self, batch_dim: int):
        """Create and return the simulation world."""
        return World(
            batch_dim,
            self.device,
            x_semidim=self.x_semidim,
            y_semidim=self.y_semidim,
            collision_force=500,
            substeps=2,
            drag=0.25,
        )
    
    def _create_agents(self, world, batch_dim):
        """Create agents and add them to the world."""
        for i in range(self.n_agents):
            agent = Agent(
                name=f"agent_{i}",
                collide=True,
                shape=Sphere(radius=0.05),
                sensors=self._create_agent_sensors(world),
            )
            self._initialize_agent_rewards(agent, batch_dim)
            self._create_agent_state_histories(agent)
            world.add_agent(agent)

    def _create_agent_sensors(self, world):
        """Create and return sensors for agents."""
        sensors = [
            Lidar(world, n_rays=self.n_lidar_rays_entities, max_range=self._lidar_range, entity_filter=lambda e: e.name.startswith("target"), render_color=Color.GREEN)
        ]
        if self.use_obstacle_lidar:
            sensors.append(Lidar(world, n_rays=self.n_lidar_rays_entities, max_range=self._lidar_range, entity_filter=lambda e: e.name.startswith("obstacle"), render_color=Color.BLUE))
        if self.use_agent_lidar:
            sensors.append(Lidar(world, n_rays=self.n_lidar_rays_agents, max_range=self._lidar_range, entity_filter=lambda e: e.name.startswith("agent"), render_color=Color.RED))
        if self.target_attribute_objective:
            sensors.append(Lidar(world, n_rays=self.n_lidar_rays_entities, max_range=self._lidar_range, entity_filter=lambda e: e.name.startswith("secondary_target"), render_color=Color.LIGHT_GREEN))
        return sensors

    def _initialize_agent_rewards(self, agent, batch_dim):
        """Initialize rewards for an agent."""
        agent.collision_rew = torch.zeros(batch_dim, device=self.device)
        agent.covering_reward = agent.collision_rew.clone()
        agent.oneshot_rew = agent.collision_rew.clone()
        agent.oneshot_signal = agent.collision_rew.clone()

        if self.use_entropy_rew:
            agent.entropy_based_rew = EntropyBasedReward(radius=self._lidar_range, max_buffer_size=30)
        
        if self.use_count_rew:
            agent.count_based_rew = CountBasedReward(k=5)
    
    def _create_agent_state_histories(self, agent):
        if self.observe_pos_history:
            agent.position_history = PositionHistory(self.num_envs,self.pos_history_length, self.pos_dim, self.device)
        if self.observe_vel_history:
            agent.velocity_history = VelocityHistory(self.num_envs,self.vel_history_length, self.vel_dim, self.device)

    def _create_targets(self, world):
        """Create target landmarks and add them to the world."""
        self._targets = [
            Landmark(f"target_{i}", collide=True, movable=False, shape=Sphere(radius=0.05), color=Color.GREEN)
            for i in range(self.n_targets)
        ]
        for target in self._targets:
            world.add_landmark(target)
        
        if self.target_attribute_objective:
            self._secondary_targets = [
            Landmark(f"secondary_target_{i}", collide=True, movable=False, shape=Sphere(radius=0.05), color=Color.LIGHT_GREEN)
            for i in range(self.n_targets)
            ]
            for target in self._secondary_targets:
                world.add_landmark(target)

    def _create_obstacles(self, world):
        """Create obstacle landmarks and add them to the world."""
        if not self.add_obstacles:
            return
        self._obstacles = [
            Landmark(f"obstacle_{i}", collide=True, movable=False, shape=Box(random.uniform(0.1, 0.25), random.uniform(0.1, 0.25)), color=Color.BLUE)
            for i in range(self.n_obstacles)
        ]
        for obstacle in self._obstacles:
            world.add_landmark(obstacle)
        
    def _create_global_heading_vis(self,world):

        if self.global_heading_objective:
            self.heading_landmark = Landmark(f"heading", collide=False, movable=False, shape=Sphere(radius=self.location_radius), color=Color.YELLOW)
            world.add_landmark(self.heading_landmark)

    def _initialize_rewards(self, batch_dim):
        """Initialize global rewards."""
        self.covered_targets = torch.zeros(batch_dim, self.n_targets, device=self.device)
        self.shared_covering_rew = torch.zeros(batch_dim, device=self.device)
        self.agent_stopped = torch.zeros(batch_dim, self.n_agents, dtype=torch.bool, device=self.device)
        self.jointentropy_rew = JointEntropyBasedReward(radius=self._lidar_range, n_agents=self.n_agents, max_buffer_size=30)
        self.num_covered_targets = torch.zeros(batch_dim, device=self.device)
        #===================
        # Language Driven Goals
        # 1) Count
        self.max_target_count = torch.ones(batch_dim, device=self.device) # Initialized to n_targets (ratio)
        # 2) Heading
        self.search_coordinates = torch.zeros((batch_dim,2),device=self.device) # Initialized to center
        self.search_encoding = torch.zeros((batch_dim,2),device=self.device) # Binary encoding for the observations
        self.possible_coordinates = torch.tensor([
            [-self.x_semidim / 2, -self.y_semidim / 2],
            [-self.x_semidim / 2,  self.y_semidim / 2],
            [ self.x_semidim / 2, -self.y_semidim / 2],
            [ self.x_semidim / 2,  self.y_semidim / 2]
        ], device=self.device)
        # 3) Attribute
        self.target_class = torch.zeros(batch_dim, device=self.device)
        #==================

        # Occupency Grid
        #occupency_grid = OccupancyGrid(x_dim=self.x_semidim*2,y_dim=self.y_semidim*2,num_cells=10,device=device)

    def _extras(self):

        if self.observe_jointpos_history:
            self.jointpos_history = JointPosHistory(self.n_agents,self.num_envs,self.pos_history_length,self.pos_dim,self.device)

    def observation(self, agent: Agent):

        lidar_measures = []
        for sensor in agent.sensors:
            lidar_measures.append(sensor.measure())
        lidar_measures = torch.cat(lidar_measures,dim=-1)

        # Compute obstacle distances if known map is enabled
        if self.add_obstacles:
            obs_dist_tensor = torch.stack(
                [self.world.get_distance(o, agent) for o in self._obstacles], dim=-1
            ) if self.known_map else None

        # Compute agent distances if known agent positions are enabled
        agent_dist_tensor = torch.stack(
            [self.world.get_distance(a, agent) for a in self.world.agents], dim=-1
        ) if self.known_agent_pos else None

        pos, vel = agent.state.pos, agent.state.vel

        # Get history observations if enabled
        pos_hist = agent.position_history.get_flattened() if self.observe_pos_history else None
        vel_hist = agent.velocity_history.get_flattened() if self.observe_vel_history else None

        # Collect all observation components
        obs_components = [pos, vel]
        obs_components.append(lidar_measures)
        # print(self.max_target_count)
        # print("==========================")
        # print(self.num_covered_targets/self.n_targets)
        # print(agent.oneshot_signal)

        if self.observe_pos_history:
            obs_components.append(pos_hist[: pos.shape[0], :])
        if self.observe_vel_history:
            obs_components.append(vel_hist[: vel.shape[0], :])
        if self.add_obstacles and self.known_map:
            obs_components.append(obs_dist_tensor)
        if self.known_agent_pos:
            obs_components.append(agent_dist_tensor)
        # if self.use_obstacle_lidar:
        #     obs_components.append(agent.sensors[1].measure())
        # if self.use_agent_lidar:
        #     obs_components.append(agent.sensors[2].measure())
        if self.target_attribute_objective:
            obs_components.append(self.target_class.unsqueeze(1))
        if self.max_target_objective:
            obs_components.append(self.max_target_count.unsqueeze(1))
            obs_components.append(self.num_covered_targets.unsqueeze(1)/self.n_targets)
        if self.global_heading_objective:
            obs_components.append(self.search_encoding)

        # Concatenate observations along last dimension
        obs = torch.cat([comp for comp in obs_components if comp is not None], dim=-1)

        # Update history buffers if enabled
        if self.observe_pos_history:
            agent.position_history.update(pos)
        if self.observe_vel_history:
            agent.velocity_history.update(vel)

        return obs

    def reset_world_at(self, env_index: int = None):
        """Reset the world for a given environment index."""
        if env_index is None: # Apply to all environements

            self.all_time_covered_targets = torch.full(
                (self.world.batch_dim, self.n_targets), False, device=self.world.device
            )
            self.agent_stopped = torch.full(
                (self.world.batch_dim, self.n_agents), False, device=self.world.device
            )

            # Randomize search coordinates
            if self.global_heading_objective:
                indices = torch.randint(0, 4, (self.world.batch_dim,), device=self.device)
                self.search_coordinates = self.possible_coordinates[indices]
                self.search_encoding = torch.stack([(indices >> 1) & 1, indices & 1], dim=-1)
                self.heading_landmark.set_pos(self.search_coordinates)

            # Randomize target class
            self.target_class = torch.randint(0, 2, (self.world.batch_dim,), device=self.device)

            # Do I need this?
            # # Reset novelty rewards
            # for agent in self.world.agents:
            #     if self.use_count_rew:
            #         agent.count_based_rew.reset()
            #     if self.use_entropy_rew:
            #         agent.entropy_based_rew.reset()
            #     if self.observe_pos_history:
            #         agent.position_history.reset()
            #     if self.observe_vel_history:
            #         agent.velocity_history.reset()
            #     agent.oneshot_signal.zero_()
            # if self.observe_jointpos_history:
            #     self.jointpos_history.reset()

        else:
            
            self.all_time_covered_targets[env_index] = False

            # Reset histories
            for agent in self.world.agents:
                if self.use_count_rew:
                    agent.count_based_rew.reset(env_index)
                if self.use_entropy_rew:
                    agent.entropy_based_rew.reset(env_index)
                if self.observe_pos_history:
                    agent.position_history.reset(env_index)
                if self.observe_vel_history:
                    agent.velocity_history.reset(env_index)
                agent.oneshot_signal[env_index] = 0.0
            if self.observe_jointpos_history:
                self.jointpos_history.reset(env_index)
            
            # Randomize max target count
            random_tensor = torch.randint(1, self.n_targets + 1, (self.world.batch_dim,))
            self.max_target_count[env_index] = random_tensor[env_index] / self.n_targets
            self.num_covered_targets[env_index].zero_()

            # Randomize search coordinates
            if self.global_heading_objective:
                indices = torch.randint(0, 4, (self.world.batch_dim,), device=self.device)
                self.search_coordinates[env_index] = self.possible_coordinates[indices[env_index]]
                self.search_encoding[env_index] = torch.tensor([(indices[env_index] >> 1) & 1,(indices[env_index] & 1)], device=self.device)
                self.heading_landmark.set_pos(self.search_coordinates[env_index],env_index)

            # Randomize target class
            if self.target_attribute_objective:
                rand = torch.randint(0, 2, (self.world.batch_dim,), device=self.device)
                self.target_class[env_index] = torch.randint(0, 2, (rand,), device=self.device)

        self._spawn_entities_randomly(env_index)

        
    def _spawn_entities_randomly(self, env_index: int):
        """Spawn agents, targets, and obstacles randomly while ensuring valid distances."""
        entities = self._targets[: self.n_targets] + self.world.agents
        if self.add_obstacles:
            entities += self._obstacles[: self.n_obstacles]
        if self.target_attribute_objective:
            entities += self._secondary_targets[: self.n_targets]

        ScenarioUtils.spawn_entities_randomly(
            entities=entities,
            world=self.world,
            env_index=env_index,
            min_dist_between_entities=self._min_dist_between_entities,
            x_bounds=(-self.world.x_semidim, self.world.x_semidim),
            y_bounds=(-self.world.y_semidim, self.world.y_semidim),
        )
        for target in self._targets[self.n_targets :]:
            target.set_pos(self._get_outside_pos(env_index), batch_index=env_index)
        for target in self._secondary_targets[self.n_targets :]:
            target.set_pos(self._get_outside_pos(env_index), batch_index=env_index)

    def reward(self, agent: Agent):
        """Compute the reward for a given agent."""
        is_first = agent == self.world.agents[0]
        is_last = agent == self.world.agents[-1]

        if is_first:
            if self.target_attribute_objective:
                self._compute_agent_distance_matrix(self._secondary_targets)
            else:
                self._compute_agent_distance_matrix(self._targets)
            self._compute_covering_rewards()

        # Initialize individual rewards
        agent.collision_rew[:] = 0
        agent.oneshot_rew[:] = 0
        covering_rew = agent.covering_reward if not self.shared_reward else self.shared_covering_rew
        covering_rew *= torch.abs(agent.oneshot_signal-1) # All postive rewards are deactivated once oneshot is on

        # Compute each reward component separately
        self._compute_collisions(agent)
        novelty_rew = self._compute_novelty_rewards(agent)
        ld_rew = self._compute_ld_rewards(agent)

        if is_last:
            self._handle_target_respawn(self._targets)
            self._handle_target_respawn(self._secondary_targets)

        return agent.collision_rew + covering_rew + self.time_penalty + agent.oneshot_rew + novelty_rew + ld_rew
    
    def agent_reward(self, agent):
        agent_index = self.world.agents.index(agent)

        agent.covering_reward[:] = 0
        targets_covered_by_agent = (
            self.agents_targets_dists[:, agent_index] < self._covering_range
        )
        num_covered_targets_covered_by_agent = (
            targets_covered_by_agent * self.covered_targets
        ).sum(dim=-1)
        agent.covering_reward += (
            num_covered_targets_covered_by_agent * self.covering_rew_coeff
        )
        return agent.covering_reward
    
    def _compute_agent_distance_matrix(self,targets):

        """Compute agent-target and agent-agent distances."""
        self.agents_pos = torch.stack([a.state.pos for a in self.world.agents], dim=1)
        self.targets_pos = torch.stack([t.state.pos for t in targets], dim=1)
        self.agents_targets_dists = torch.cdist(self.agents_pos, self.targets_pos)

        self.agents_covering_targets = self.agents_targets_dists < self._covering_range
        self.agents_per_target = torch.sum((self.agents_covering_targets).type(torch.int),dim=1)
        self.agent_is_covering = self.agents_covering_targets.any(dim=-1)
        self.covered_targets = self.agents_per_target >= self._agents_per_target

    def _compute_collisions(self, agent):
        """Compute penalties for collisions with agents and obstacles."""

        for a in self.world.agents:
            if a != agent:
                agent.collision_rew[
                    self.world.get_distance(a, agent) < self.min_collision_distance
                ] += self.agent_collision_penalty

        # Avoid collision with obstacles
        if self.add_obstacles:
            for o in self._obstacles:
                agent.collision_rew[
                    self.world.get_distance(o,agent) < self.min_collision_distance
                ] += self.obstacle_collision_penalty

    def _compute_novelty_rewards(self, agent):

        """Compute novelty-based rewards (count-based, entropy, joint entropy)."""
        pos = agent.state.pos
        reward = torch.zeros(self.world.batch_dim, device=self.world.device)

        if self.use_count_rew:
            reward += agent.count_based_rew.compute(pos)*torch.abs(agent.oneshot_signal-1) # If oneshot is activated, no more reward for moving
            agent.count_based_rew.update(pos)

        if self.use_entropy_rew:
            reward += agent.entropy_based_rew.compute(pos)*torch.abs(agent.oneshot_signal-1)
            agent.entropy_based_rew.update(pos)

        if self.use_jointentropy_rew:
            all_positions = torch.stack([a.state.pos for a in self.world.agents], dim=1)
            reward += self.jointentropy_rew.compute(all_positions)*torch.abs(agent.oneshot_signal-1)
            self.jointentropy_rew.update(all_positions)

        return reward
    
    def _compute_ld_rewards(self, agent):

        """Compute rewards for language-driven concepts"""
        reward = torch.zeros(self.world.batch_dim, device=self.world.device)

        # 1) Count
        if self.max_target_objective:
            #Activate oneshot penalty once the max target objective has been reached
            self.num_covered_targets = self.all_time_covered_targets.sum(dim=1)
            reached_mask = self.num_covered_targets >= self.max_target_count*self.n_targets
            if reached_mask.any():
                movement_penalty = torch.sum(agent.state.vel[reached_mask]**2, dim=-1) * -1.0
                agent.oneshot_rew[reached_mask] = movement_penalty
                agent.oneshot_signal[reached_mask] = 1.0

        # 2) Heading
        if self.global_heading_objective:
            pos = agent.state.pos
            squared_dist = torch.sum((self.search_coordinates - pos) ** 2, dim=-1)
            reached_mask = squared_dist < self.location_radius**2
            reward += -0.05 # Penalty for not being in the search region
            if reached_mask.any():
                reward[reached_mask] += 0.1 # Reward for being in the search region

        return reward

    def _compute_covering_rewards(self):
        """Compute covering rewards and update shared covering reward."""
        self.shared_covering_rew[:] = 0
        for agent in self.world.agents:
            self.shared_covering_rew += self.agent_reward(agent)
        self.shared_covering_rew[self.shared_covering_rew != 0] /= 2

    def _handle_target_respawn(self,targets):
        """Handle target respawn and removal for covered targets."""
        occupied_positions_agents = [self.agents_pos]

        for i, target in enumerate(targets):
            occupied_positions_targets = [o.state.pos.unsqueeze(1) for o in targets if o is not target]
            occupied_positions = torch.cat(occupied_positions_agents + occupied_positions_targets, dim=1)

            # Respawn targets that have been covered
            if self.targets_respawn:
                pos = ScenarioUtils.find_random_pos_for_entity(
                    occupied_positions,
                    env_index=None,
                    world=self.world,
                    min_dist_between_entities=self._min_dist_between_entities,
                    x_bounds=(-self.world.x_semidim, self.world.x_semidim),
                    y_bounds=(-self.world.y_semidim, self.world.y_semidim),
                )

                target.state.pos[self.covered_targets[:, i]] = pos[self.covered_targets[:, i]].squeeze(1)

            else:
                # Keep track of all-time covered targets
                self.all_time_covered_targets += self.covered_targets

                # # If all targets have been covered, apply final reward
                # if self.shared_final_reward and self.all_time_covered_targets.all():
                #     self.shared_covering_rew += 5  # Final reward

                # Move covered targets outside the environment
                target.state.pos[self.covered_targets[:, i]] = self._get_outside_pos(None)[
                    self.covered_targets[:, i]
                ]

    def _handle_agents_staying_at_target(self):
        # Handle agents staying at the target
        if self.agents_stay_at_target:
            for agent in self.world.agents:
                agent_index = self.world.agents.index(agent)
                reached_target = self.agent_is_covering[:, agent_index]

                newly_stopped = reached_target & ~self.agent_stopped[:, agent_index]
                self.agent_stopped[:, agent_index] |= newly_stopped

                # Apply movement penalty if the agent has stopped
                movement_penalty = torch.linalg.norm(agent.state.vel[self.agent_stopped[:, agent_index]], dim=-1) * -1.0
                agent.oneshot_rew[self.agent_stopped[:, agent_index]] = movement_penalty

    def _get_outside_pos(self, env_index):
        """Get a position far outside the environment to hide entities."""
        return torch.empty(
            (
                (1, self.world.dim_p)
                if env_index is not None
                else (self.world.batch_dim, self.world.dim_p)
            ),
            device=self.world.device,
        ).uniform_(-1000 * self.world.x_semidim, -10 * self.world.x_semidim)

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        """Return auxiliary information about the agent."""
        return {
            "covering_reward": agent.covering_reward if not self.shared_reward else self.shared_covering_rew,
            "collision_rew": agent.collision_rew,
            "onehsot_rew": agent.oneshot_rew,
            "targets_covered": self.covered_targets.sum(-1),
        }

    def done(self):
        """Check if all targets are covered and simulation should end."""
        return self.all_time_covered_targets.all(dim=-1)

    def extra_render(self, env_index: int = 0) -> "List[Geom]":
        """Render additional visual elements."""
        from vmas.simulator import rendering

        geoms = []
        for target in self._targets:
            range_circle = rendering.make_circle(self._covering_range, filled=False)
            xform = rendering.Transform()
            xform.set_translation(*target.state.pos[env_index])
            range_circle.add_attr(xform)
            range_circle.set_color(*self.target_color.value)
            geoms.append(range_circle)

        # Render communication lines between agents
        for i, agent1 in enumerate(self.world.agents):
            for j, agent2 in enumerate(self.world.agents):
                if j <= i:
                    continue
                if self.world.get_distance(agent1, agent2)[env_index] <= self._comms_range:
                    line = rendering.Line(
                        agent1.state.pos[env_index], agent2.state.pos[env_index], width=1
                    )
                    line.set_color(*Color.BLACK.value)
                    geoms.append(line)

        return geoms
    
