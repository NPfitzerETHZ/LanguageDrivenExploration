import torch
import random
import math
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.core import Agent,Landmark, Sphere, Box, World
from vmas.simulator.utils import Color, ScenarioUtils 

from histories import VelocityHistory, PositionHistory, JointPosHistory
from occupancy_grid import OccupancyGrid
from rewards import CountBasedReward, EntropyBasedReward, JointEntropyBasedReward

from vmas.simulator.sensors import Lidar

from typing import Dict

class MyScenario(BaseScenario):
    
    def _load_config(self,kwargs):

        #self.num_envs = kwargs.pop("num_envs", 96) disregard for BenchMarl
        self.n_agents = kwargs.pop("n_agents", 2)
        self.n_targets = kwargs.pop("n_targets", 4)
        self.n_obstacles = kwargs.pop("n_obstacle",5)
        self.x_semidim = kwargs.pop("x_semidim", 1.0)
        self.y_semidim = kwargs.pop("y_semidim", 1.0)
        self.agent_radius = kwargs.pop("agent_radius", 0.05)
        self._min_dist_between_entities = kwargs.pop("min_dist_between_entities", 0.2)
        self._lidar_range = kwargs.pop("lidar_range", 0.15)
        self._covering_range = kwargs.pop("covering_range", 0.15)

        self.use_lidar = kwargs.pop("use_lidar", True)
        self.use_target_lidar = kwargs.pop("use_target_lidar", False)
        self.use_agent_lidar = kwargs.pop("use_agent_lidar", False)
        self.use_obstacle_lidar = kwargs.pop("use_obstacle_lidar", False)
        self.n_lidar_rays_entities = kwargs.pop("n_lidar_rays_entities", 8)
        self.n_lidar_rays_agents = kwargs.pop("n_lidar_rays_agents", 12)

        self._agents_per_target = kwargs.pop("agents_per_target", 1)
        self.agents_stay_at_target = kwargs.pop("agents_stay_at_target", False)
        self.targets_respawn = kwargs.pop("targets_respawn", False)
        self.shared_reward = kwargs.pop("shared_reward", False)  # Reward for finding a target
        self.shared_final_reward = kwargs.pop("shared_final_reward", True) # Reward for finding all targets, if targets_respawn false

        self.add_obstacles = kwargs.pop("add_obstacles", False)  # This isn't implemented yet

        # Novelty rewards
        self.use_count_rew = kwargs.pop("use_count_rew", False)
        self.use_entropy_rew = kwargs.pop("use_entropy_rew", False)
        self.use_jointentropy_rew = kwargs.pop("use_jointentropy_rew", False)
        self.use_occupancy_grid_rew = kwargs.pop("use_occupencygrid_rew", True)

        self.known_map = kwargs.pop("known_map", False)
        self.known_agent_pos = kwargs.pop("known_agents",False) # Not ure about this one. Do I need a GNN?

        self.agent_collision_penalty = kwargs.pop("agent_collision_penalty", -0.0)
        self.obstacle_collision_penalty = kwargs.pop("obstacle_collision_penalty", -0.75)
        self.covering_rew_coeff = kwargs.pop("covering_rew_coeff", 5.0) # Large reward for finding a target
        self.false_covering_penalty_coeff = kwargs.pop("false_covering_penalty_coeff", -0.25) # Penalty for covering wrong target
        self.time_penalty = kwargs.pop("time_penalty", 0)
        ScenarioUtils.check_kwargs_consumed(kwargs)

        self.min_collision_distance = 0.005

        self.viewer_zoom = 1
        self.target_color = Color.GREEN
        self.obstacle_color = Color.BLUE

        # Histories
        self.pos_history_length = 10
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

        # Corridors
        self.num_corridors = 2

        # Grid
        self.observe_grid = kwargs.pop("observe_grid",False)
        self.num_grid_cells = (2*self.num_corridors+1)**2 # Must be n^2 with n = width 
        self.mini_grid_dim = 3

        self.plot_grid = True
        self.visualize_semidims = False
    
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
                shape=Sphere(radius=self.agent_radius),
                sensors=(self._create_agent_sensors(world) if self.use_lidar else []),
                color=Color.GREEN
            )
            self._initialize_agent_rewards(agent, batch_dim)
            self._create_agent_state_histories(agent, batch_dim)
            world.add_agent(agent)
    
    def _create_agent_sensors(self, world):
        """Create and return sensors for agents."""
        sensors = []
        
        if self.use_target_lidar:
            sensors.append(Lidar(world, n_rays=self.n_lidar_rays_entities, max_range=self._lidar_range, entity_filter=lambda e: e.name.startswith("target"), render_color=Color.GREEN))
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
    
    def _create_agent_state_histories(self, agent, batch_dim):
        if self.observe_pos_history:
            agent.position_history = PositionHistory(batch_dim,self.pos_history_length, self.pos_dim, self.device)
        if self.observe_vel_history:
            agent.velocity_history = VelocityHistory(batch_dim,self.vel_history_length, self.vel_dim, self.device)

    def _create_targets(self, world):
        """Create target landmarks and add them to the world."""

        self.target_groups = []
        self._targets = [
            Landmark(f"target_{i}", collide=True, movable=False, shape=Sphere(radius=0.05), color=Color.GREEN)
            for i in range(self.n_targets)
        ]
        for target in self._targets:
            world.add_landmark(target)
        self.target_groups.append(self._targets)
        
        if self.target_attribute_objective:
            self._secondary_targets = [
            Landmark(f"secondary_target_{i}", collide=True, movable=False, shape=Sphere(radius=0.05), color=Color.LIGHT_GREEN)
            for i in range(self.n_targets)
            ]
            for target in self._secondary_targets:
                world.add_landmark(target)
            self.target_groups.append(self._secondary_targets)
    
    def _create_obstacles(self, world):
        """Create obstacle landmarks and add them to the world."""
        
        self._obstacles = [
            Landmark(f"obstacle_{i}", collide=True, movable=False, shape=Box(random.uniform(0.1, 0.25), random.uniform(0.1, 0.25)), color=Color.BLUE)
            for i in range(self.n_obstacles)
        ]
        for obstacle in self._obstacles:
            world.add_landmark(obstacle)
    
    def _create_occupancy_grid(self, batch_dim):

        self.occupancy_grid = OccupancyGrid(
            batch_size=batch_dim,
            x_dim=self.x_semidim*2,
            y_dim=self.y_semidim*2,
            num_cells=self.num_grid_cells,
            num_headings=self.n_targets,
            mini_grid_dim=self.mini_grid_dim,
            device=self.device)

    def _initialize_heading(self, batch_dim):

        self.num_headings = 4
        self.search_indices = torch.zeros(batch_dim,device=self.device)
        self.search_encoding = torch.zeros((batch_dim,math.ceil(math.log2(self.num_headings))),device=self.device) # Binary encoding for the observations

    def _initialize_rewards(self, batch_dim):

        """Initialize global rewards."""
        self.covered_targets = torch.zeros(batch_dim, self.n_targets, device=self.device)
        self.shared_covering_rew = torch.zeros(batch_dim, device=self.device)
        self.agent_stopped = torch.zeros(batch_dim, self.n_agents, dtype=torch.bool, device=self.device)
        self.jointentropy_rew = JointEntropyBasedReward(radius=self._lidar_range, n_agents=self.n_agents, max_buffer_size=30)
        self.num_covered_targets = torch.zeros(batch_dim, device=self.device)
        self.covering_rew_val = torch.ones(batch_dim, device=self.device) * (self.covering_rew_coeff)
        #===================
        # Language Driven Goals
        # 1) Count
        self.max_target_count = torch.ones(batch_dim, device=self.device) # Initialized to n_targets (ratio)
        # 2) Heading
        self.search_coordinates = torch.zeros((batch_dim,2),device=self.device) # Initialized to center
        self.possible_coordinates = torch.tensor([
            [-self.x_semidim / 2, -self.y_semidim / 2],
            [-self.x_semidim / 2,  self.y_semidim / 2],
            [ self.x_semidim / 2, -self.y_semidim / 2],
            [ self.x_semidim / 2,  self.y_semidim / 2]
        ], device=self.device)
        # 3) Attribute
        self.target_class = torch.zeros(batch_dim, dtype=torch.int, device=self.device)
        self.targets_pos = torch.zeros((batch_dim,len(self.target_groups),self.n_targets,2), device=self.device)
        #==================
    
    #====================================================================================================================
    #====================================================================================================================

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
        if self.target_attribute_objective:
            obs_components.append(self.target_class.unsqueeze(1)/(len(self.target_groups)-1))
        if self.max_target_objective:
            obs_components.append(self.max_target_count.unsqueeze(1))
            obs_components.append(self.num_covered_targets.unsqueeze(1)/self.n_targets)
        if self.global_heading_objective:
            obs_components.append(self.search_encoding)
        if self.use_occupancy_grid_rew:
            obs_components.append(self.occupancy_grid.get_observation_normalized(pos,self.mini_grid_dim))
        if self.observe_grid:
            obs_components.append(self.occupancy_grid.observe_grid())

        # Concatenate observations along last dimension
        obs = torch.cat([comp for comp in obs_components if comp is not None], dim=-1)

        # Update history buffers if enabled
        if self.observe_pos_history:
            agent.position_history.update(pos)
        if self.observe_vel_history:
            agent.velocity_history.update(vel)

        return obs
    
    #====================================================================================================================
    #====================================================================================================================
    
    def agent_reward(self, agent):
        agent_index = self.world.agents.index(agent)

        agent.covering_reward[:] = 0
        targets_covered_by_agent = (
            self.agents_targets_dists[:, :, agent_index, :] < self._covering_range  # (batch_size, target_groups, n_targets)
        )
        num_covered_targets_covered_by_agent = (
            targets_covered_by_agent * self.covered_targets
        ).sum(dim=-1)  # (batch_size, target_groups)
        
        # Create a mask based on self.target_class
        reward_mask = torch.arange(self.covered_targets.shape[1], device=self.target_class.device).unsqueeze(0)  # (1, target_groups)
        reward_mask = (reward_mask == self.target_class.unsqueeze(1))  # (batch_size, target_groups)

        # Apply reward for the selected group and penalty for others
        group_rewards = (
            num_covered_targets_covered_by_agent * self.covering_rew_coeff * reward_mask
            + num_covered_targets_covered_by_agent * self.false_covering_penalty_coeff * (~reward_mask)
        )  # (batch_size, target_groups)

        # Aggregate over target_groups to get (batch_size,)
        agent.covering_reward += group_rewards.sum(dim=-1)
        return agent.covering_reward
    
    def _compute_agent_distance_matrix(self):

        """Compute agent-target and agent-agent distances."""
        self.agents_pos = torch.stack([a.state.pos for a in self.world.agents], dim=1)
        for i, targets in enumerate(self.target_groups):
            targets_pos = torch.stack([t.state.pos for t in targets], dim=1)  # (batch_size, n_targets, 2)
            self.targets_pos[:, i, :, :] = targets_pos

        # Compute agent-target distances
        self.agents_targets_dists = torch.cdist(
            self.agents_pos.unsqueeze(1),  # (batch_size, 1, n_agents, 2)
            self.targets_pos  # (batch_size, target_groups, n_targets, 2)
        )  # Output: (batch_size, target_groups, n_agents, n_targets)

        # Determine which agents are covering which targets
        self.agents_covering_targets = self.agents_targets_dists < self._covering_range  # (batch_size, target_groups, n_agents, n_targets)

        # Count number of agents covering each target
        self.agents_per_target = torch.sum(self.agents_covering_targets.int(), dim=2)  # (batch_size, target_groups, n_targets)

        # Identify which agents are covering at least one target
        self.agent_is_covering = self.agents_covering_targets.any(dim=2)  # (batch_size, target_groups, n_targets)

        # Define which targets are covered
        self.covered_targets = self.agents_per_target >= self._agents_per_target

    def _compute_collisions(self, agent):
        """Compute penalties for collisions with agents and obstacles."""

        for a in self.world.agents:
            if a != agent:
                agent.collision_rew[
                    self.world.get_distance(a, agent) < self.min_collision_distance
                ] += self.agent_collision_penalty

        # Avoid collision with obstacles
        pos = agent.state.pos
        if self.add_obstacles:
            for o in self._obstacles:
                agent.collision_rew[
                    self.world.get_distance(o,agent) < self.min_collision_distance
                ] += self.obstacle_collision_penalty

            # No collision with borders (make sure the grid is setup to see the border)
            mask_x = (pos[:, 0] > self.x_semidim - self.agent_radius) | (pos[:, 0] < -self.x_semidim + self.agent_radius)
            mask_y = (pos[:, 1] > self.y_semidim - self.agent_radius) | (pos[:, 1] < -self.y_semidim + self.agent_radius)
            agent.collision_rew[mask_x] += self.obstacle_collision_penalty
            agent.collision_rew[mask_y] += self.obstacle_collision_penalty
    
    def _compute_covering_rewards(self):
        """Compute covering rewards and update shared covering reward."""
        self.shared_covering_rew[:] = 0
        for agent in self.world.agents:
            self.shared_covering_rew += self.agent_reward(agent)
        self.shared_covering_rew[self.shared_covering_rew != 0] /= 2

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

        if self.use_jointentropy_rew: # This is wrongggg
            all_positions = torch.stack([a.state.pos for a in self.world.agents], dim=1)
            reward += self.jointentropy_rew.compute(all_positions)*torch.abs(agent.oneshot_signal-1)
            self.jointentropy_rew.update(all_positions)
        
        if self.use_occupancy_grid_rew:
            reward += self.occupancy_grid.compute_exploration_bonus(pos)
            self.occupancy_grid.update(pos)
        
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
    
    #====================================================================================================================
    #====================================================================================================================

    def _handle_target_respawn(self):
        """Handle target respawn and removal for covered targets."""

        for j, targets in enumerate(self.target_groups):
            indices = torch.where(self.target_class == j)[0]
            for i, target in enumerate(targets):
                # Keep track of all-time covered targets
                self.all_time_covered_targets[indices] += self.covered_targets[indices,self.target_class[indices]]

                # # If all targets have been covered, apply final reward
                # if self.shared_final_reward and self.all_time_covered_targets.all():
                #     self.shared_covering_rew += 5  # Final reward

                # Move covered targets outside the environment
                indices_selected = torch.where(self.covered_targets[indices,self.target_class[indices],i])[0]
                indices_selected = indices[indices_selected]
                target.state.pos[indices_selected,:] = self._get_outside_pos(None)[
                    indices_selected
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
    
    def info(self, agent: Agent) -> Dict[str, torch.Tensor]:
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
    