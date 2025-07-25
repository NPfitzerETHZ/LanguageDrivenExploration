import torch
import random
import math
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.core import Agent,Landmark, Sphere, Box, World
from vmas.simulator.utils import Color, ScenarioUtils 

from scenarios.centralized.scripts.histories import VelocityHistory, PositionHistory
from scenarios.grids.old.occupancy_grid import OccupancyGrid
from scenarios.old.rewards import CountBasedReward, EntropyBasedReward
from vmas.simulator.controllers.velocity_controller import VelocityController

from vmas.simulator.sensors import Lidar

from typing import Dict

class MyScenario(BaseScenario):
    
    def _load_config(self,kwargs):

        #self.num_envs = kwargs.pop("num_envs", 96) disregard for BenchMarl
        self.x_semidim = kwargs.pop("x_semidim", 1.0)
        self.y_semidim = kwargs.pop("y_semidim", 1.0)
        self._min_dist_between_entities = kwargs.pop("min_dist_between_entities", 0.2)
        self._lidar_range = kwargs.pop("lidar_range", 0.15)
        self._covering_range = kwargs.pop("covering_range", 0.15)

        self.use_target_lidar = kwargs.pop("use_target_lidar", False)
        self.use_agent_lidar = kwargs.pop("use_agent_lidar", False)
        self.n_lidar_rays_entities = kwargs.pop("n_lidar_rays_entities", 8)
        self.n_lidar_rays_agents = kwargs.pop("n_lidar_rays_agents", 12)

        self._agents_per_target = kwargs.pop("agents_per_target", 1)
        self.agents_stay_at_target = kwargs.pop("agents_stay_at_target", False)
        self.targets_respawn = kwargs.pop("targets_respawn", False)
        self.shared_target_reward = kwargs.pop("shared_target_reward", True)  # Reward for finding a target
        self.shared_final_reward = kwargs.pop("shared_final_reward", True) # Reward for finding all targets, if targets_respawn false

        self.add_obstacles = kwargs.pop("add_obstacles", False)  # This isn't implemented yet

        # Novelty rewards
        self.use_count_rew = kwargs.pop("use_count_rew", False)
        self.use_entropy_rew = kwargs.pop("use_entropy_rew", False)
        self.use_jointentropy_rew = kwargs.pop("use_jointentropy_rew", False)
        self.use_occupancy_grid_rew = kwargs.pop("use_occupencygrid_rew", True)

        self.known_map = kwargs.pop("known_map", False)
        self.known_agent_pos = kwargs.pop("known_agents",False) # Not ure about this one. Do I need a GNN?

        self.obstacle_collision_penalty = kwargs.pop("obstacle_collision_penalty", -0.75)
        self.covering_rew_coeff = kwargs.pop("covering_rew_coeff", 5.0) # Large reward for finding a target
        self.false_covering_penalty_coeff = kwargs.pop("false_covering_penalty_coeff", -0.25) # Penalty for covering wrong target
        self.time_penalty = kwargs.pop("time_penalty", 0)
        ScenarioUtils.check_kwargs_consumed(kwargs)

        self.min_collision_distance = kwargs.pop("min_collision_distance", 0.1)

        self.viewer_zoom = 1
        self.target_color = Color.GREEN
        self.obstacle_color = Color.BLUE

        # Histories
        self.pos_history_length = 10
        self.pos_dim = 2

        self.vel_history_length = 30
        self.vel_dim = 2
        self.observe_vel_history = kwargs.pop("observe_vel_history", False)

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
            dim_c=self.comm_dim,
            collision_force=500,
            substeps=2,
            drag=0.25,
        )
    
    def _create_agents(self, world, batch_dim, use_velocity_controler, silent):
        """Create agents and add them to the world."""
        for i in range(self.n_agents):
            agent = Agent(
                name=f"agent_{i}",
                collide=True,
                silent=silent,
                shape=Sphere(radius=self.agent_radius),
                mass=self.agent_weight,
                max_speed=self.agent_max_speed,
                u_multiplier=1,
                sensors=(self._create_agent_sensors(world) if self.use_lidar else []),
                color=Color.GREEN
            )
            if use_velocity_controler:
                pid_controller_params = [2, 6, 0.002]
                agent.controller = VelocityController(
                    agent, world, pid_controller_params, "standard"
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

    def _initialize_agent_rewards(self, agent: Agent, batch_dim):
        """Initialize rewards for an agent."""
        agent.collision_rew = torch.zeros(batch_dim, device=self.device)
        agent.covering_reward = agent.collision_rew.clone()

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
        self.shared_covering_rew = torch.zeros(batch_dim, device=self.device)
        self.agent_stopped = torch.zeros(batch_dim, self.n_agents, dtype=torch.bool, device=self.device)
        self.num_covered_targets = torch.zeros(batch_dim, device=self.device)
        self.covering_rew_val = torch.ones(batch_dim, device=self.device) * (self.covering_rew_coeff)
       
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

    def _compute_exploration_rewards(self, agent, pos):
        
        agent.exploration_rew += agent.occupancy_grid.compute_exploration_bonus(pos, exploration_rew_coeff=self.exploration_rew_coeff, new_cell_rew_coeff=self.new_cell_rew_coeff)
        if (self.global_heading_objective or self.llm_activate):
                agent.exploration_rew += self.occupancy_grid.compute_region_heading_bonus_normalized(pos, heading_exploration_rew_coeff=self.heading_exploration_rew_coeff)
                self.occupancy_grid.update_heading_coverage_ratio()
                if self.comm_dim > 0:
                    agent.coverage_rew = self.occupancy_grid.compute_coverage_ratio_bonus(self.coverage_action[agent.name]) 
        self.occupancy_grid.update(pos)
        agent.occupancy_grid.update(pos)
    
    def _compute_termination_rewards(self,agent):
        reached_mask = agent.num_covered_targets >= self.max_target_count 
        agent.termination_rew += reached_mask * (1 - agent.termination_signal) * self.terminal_rew_coeff
        if reached_mask.any():
            movement_penalty = torch.sum(agent.state.vel[reached_mask]**2, dim=-1) * self.termination_penalty_coeff
            agent.termination_rew[reached_mask] += movement_penalty
            agent.termination_signal[reached_mask] = 1.0
    
    #====================================================================================================================

    def _handle_target_respawn(self):
        """Handle target respawn and removal for covered targets."""

        for j, targets in enumerate(self.target_groups):
            indices = torch.where(self.target_class == j)[0]
            for i, target in enumerate(targets):
                # Keep track of all-time covered targets
                self.all_time_covered_targets[indices] += self.covered_targets[indices]

                # Move covered targets outside the environment
                indices_selected = torch.where(self.covered_targets[indices,self.target_class[indices],i])[0]
                indices_selected = indices[indices_selected]
                target.state.pos[indices_selected,:] = self._get_outside_pos(None)[
                    indices_selected
                ]
    
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
            "covering_reward": agent.covering_reward if not self.shared_target_reward else self.shared_covering_rew,
            "collision_rew": agent.collision_rew,
            "targets_covered": self.covered_targets.sum(-1),
        }

    def done(self):
        """Check if all targets are covered and simulation should end."""
        return self.all_time_covered_targets.all(dim=-1)
    