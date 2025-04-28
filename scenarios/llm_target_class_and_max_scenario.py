import typing
from typing import List

import torch
import numpy as np
import random

from vmas import render_interactively
from vmas.simulator.core import Agent,Landmark, Sphere, Box, World, Line
from vmas.simulator.utils import Color, ScenarioUtils

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom

from scenarios.myscenario import MyScenario
from scenarios.old.dead_end import DeadEndOccupancyGrid
from scenarios.scripts.heading import HeadingOccupancyGrid
from scenarios.scripts.general_purpose_occupancy_grid import GeneralPurposeOccupancyGrid, load_task_data, load_decoder

# color_dict = {
#     "red":      {"rgb": [1.0, 0.0, 0.0], "index": 0},
#     "green":    {"rgb": [0.0, 1.0, 0.0], "index": 1},
#     "blue":     {"rgb": [0.0, 0.0, 1.0], "index": 2},
#     "yellow":   {"rgb": [1.0, 1.0, 0.0], "index": 3},
#     "cyan":     {"rgb": [0.0, 1.0, 1.0], "index": 4},
#     "magenta":  {"rgb": [1.0, 0.0, 1.0], "index": 5},
#     "orange":   {"rgb": [1.0, 0.5, 0.0], "index": 6},
#     "purple":   {"rgb": [0.5, 0.0, 0.5], "index": 7},
#     "pink":     {"rgb": [1.0, 0.75, 0.8],"index": 8},
#     "brown":    {"rgb": [0.6, 0.4, 0.2], "index": 9},
#     "gray":     {"rgb": [0.5, 0.5, 0.5], "index": 10}
# }

color_dict = {
    "red":      {"rgb": [1.0, 0.0, 0.0], "index": 0},
    "green":    {"rgb": [0.0, 1.0, 0.0], "index": 1},
    "blue":     {"rgb": [0.0, 0.0, 1.0], "index": 2},
    "yellow":   {"rgb": [1.0, 1.0, 0.0], "index": 3},
    "orange":   {"rgb": [1.0, 0.5, 0.0], "index": 4},
    # "cyan":     {"rgb": [0.0, 1.0, 1.0], "index": 5},
    # "magenta":  {"rgb": [1.0, 0.0, 1.0], "index": 6},
    # "purple":   {"rgb": [0.5, 0.0, 0.5], "index": 7},
    # "pink":     {"rgb": [1.0, 0.75, 0.8], "index":8},
    # "brown":    {"rgb": [0.6, 0.4, 0.2], "index": 9},
    # "gray":     {"rgb": [0.5, 0.5, 0.5], "index": 10}
}

class MyLanguageScenario(MyScenario):

    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        """Initialize the world and entities."""
        self.device = device
        self._load_config(kwargs)
        self._load_scenario_config(kwargs)
        self._initialize_scenario_vars(batch_dim)
        world = self._create_world(batch_dim)
        self._create_occupancy_grid(batch_dim)
        self._create_agents(world, batch_dim)
        self._create_targets(world)
        self._create_obstacles(world)
        self._initialize_rewards(batch_dim)
        return world
    
    def _load_scenario_config(self,kwargs):

        self.data_json_path = kwargs.pop("data_json_path", "")
        self.decoder_model_path = kwargs.pop("decoder_model_path", "")
        self.use_decoder = kwargs.pop("use_decoder", False)
        self.use_grid_data = kwargs.pop("use_grid_data", True)
        self.use_class_data = kwargs.pop("use_class_data", True)
        self.use_max_targets_data = kwargs.pop("use_max_targets_data", True)
        
        self.n_targets_per_class = kwargs.pop("n_targets_per_class", 1)
        self.n_agents = kwargs.pop("n_agents", 6)

        self.use_gnn = kwargs.pop("use_gnn", False)
        self._comms_range = kwargs.pop("comms_radius", 0.35)
        self.agent_radius = kwargs.pop("agent_radius", 0.025)

        self.use_lidar = kwargs.pop("use_lidar", False)
        self.use_agent_lidar = kwargs.pop("use_agent_lidar", False)
        self.use_obstacle_lidar = kwargs.pop("use_obstacle_lidar", False)
        self.add_obstacles = kwargs.pop("add_obstacles", True)  # This isn't implemented yet

        # Novelty rewards
        self.use_count_rew = kwargs.pop("use_count_rew", False)
        self.use_entropy_rew = kwargs.pop("use_entropy_rew", False)
        self.use_jointentropy_rew = kwargs.pop("use_jointentropy_rew", False)
        self.use_occupancy_grid_rew = kwargs.pop("use_occupency_grid_rew", True)
        self.use_expo_search_rew = kwargs.pop("use_expo_search_rew", True)

        self.agent_collision_penalty = kwargs.pop("agent_collision_penalty", 0.00)
        self.obstacle_collision_penalty = kwargs.pop("obstacle_collision_penalty", -0.5)
        self.covering_rew_coeff = kwargs.pop("covering_rew_coeff", 3.0) # Large-ish reward for finding a target
        self.false_covering_penalty_coeff = kwargs.pop("false_covering_penalty_coeff", -0.25) # Penalty for covering wrong target if hinted
        self.time_penalty = kwargs.pop("time_penalty", -0.01)
        self.terminal_rew_coeff = kwargs.pop("terminal_rew_coeff", 15.0) # Large reward for finding max_targets
        self.exponential_search_rew = kwargs.pop("exponential_search_rew_coeff", 1.5)
        self.oneshot_coeff = kwargs.pop("oneshot_coeff", -10.0)
        self.exploration_rew_coeff = kwargs.pop("exploration_rew_coeff", -0.01)
        self.new_cell_rew_coeff = kwargs.pop("new_cell_rew_coeff", 0.125)
        self.heading_exploration_rew_coeff = kwargs.pop("heading_exploration_rew_coeff", 0.5)
        self.heading_sigma = kwargs.pop("heading_sigma", 3.0)

        #===================
        # Language Driven Goals
        
        # *** Simulators *** (Mimic language driven behavior)
        # 1) Count
        self.max_target_objective = kwargs.pop("max_target_objective", False) # Enter as fraction of total number of targets
        # 2) Heading
        self.global_heading_objective = kwargs.pop("global_heading_objective", False)
        # 3) Attribute
        self.target_attribute_objective = kwargs.pop("target_attribute_objective", False)
        self.n_target_classes = kwargs.pop("n_target_classes", 2)
        self.n_targets = self.n_target_classes * self.n_targets_per_class
        
        # *** Actually Using LLM ***
        self.llm_activate = kwargs.pop("llm_activate", True)
        if self.max_target_objective or self.global_heading_objective or self.target_attribute_objective:
            print("Make sure to deactivate all Language Driven Goal Simulators")
            self.llm_activate = False
        #===================

        # Grid
        self.n_obstacles = kwargs.pop("n_obstacles", 10)
        self.observe_grid = kwargs.pop("observe_grid",True)
        self.num_grid_cells = kwargs.pop("num_grid_cells", 400) # Must be n^2 with n = width 
        self.mini_grid_radius = kwargs.pop("mini_grid_radius", 1) # Radius of the mini grid

        self.plot_grid = True
        self.visualize_semidims = False

        # Histories
        self.pos_history_length = 4
        self.pos_dim = 2
        self.observe_pos_history = kwargs.pop("observe_pos_history", True)

    def _create_occupancy_grid(self, batch_dim):
        
        # Initialize Important Stuff
        if self.use_decoder: load_decoder(self.decoder_model_path, self.device)
        if self.llm_activate: load_task_data(
            json_path=self.data_json_path,
            use_decoder=self.use_decoder,
            use_grid_data=self.use_grid_data,
            use_class_data=self.use_class_data,
            use_max_targets_data=self.use_max_targets_data,
            device=self.device)
        self.occupancy_grid = GeneralPurposeOccupancyGrid(
            batch_size=batch_dim,
            x_dim=self.x_semidim*2,
            y_dim=self.y_semidim*2,
            num_cells=self.num_grid_cells,
            num_targets=self.n_targets,
            num_targets_per_class=self.n_targets_per_class,
            heading_mini_grid_radius=self.mini_grid_radius*2,
            device=self.device)
        self._covering_range = self.occupancy_grid.cell_radius

    
    def _create_obstacles(self, world):

        """Create obstacle landmarks and add them to the world."""
        self._obstacles = [
            Landmark(f"obstacle_{i}", collide=True, movable=False, shape=Box(self.occupancy_grid.cell_size_x,self.occupancy_grid.cell_size_y), color=Color.RED)
            for i in range(self.n_obstacles)
        ]
        for obstacle in self._obstacles:
            world.add_landmark(obstacle)
    
    def _create_targets(self, world):
        """Create target landmarks and add them to the world."""

        self.target_groups = []
        self._targets = []
        for i in range(self.n_target_classes):
            color = self.target_colors[i].tolist()
            targets = [
                Landmark(f"target_{i}_{j}", collide=False, movable=False, shape=Box(length=self.occupancy_grid.cell_size_y,width=self.occupancy_grid.cell_size_x), color=color)
                for j in range(self.n_targets_per_class)
            ]
            self._targets += targets
            self.target_groups.append(targets)
        for target in self._targets:
            world.add_landmark(target)
    
    def _initialize_scenario_vars(self, batch_dim):
        
        self.max_target_count = torch.ones(batch_dim, dtype=torch.int, device=self.device) * self.n_targets_per_class # Initialized to n_targets (ratio)
        self.target_class = torch.zeros(batch_dim, dtype=torch.int, device=self.device)
        self.targets_pos = torch.zeros((batch_dim,self.n_target_classes,self.n_targets_per_class,2), device=self.device)
        
        self.covered_targets = torch.zeros(batch_dim, self.n_target_classes, self.n_targets_per_class, device=self.device)
        
        self.target_colors = torch.zeros((self.n_target_classes, 3), device=self.device)
        for target_class_idx in range(self.n_target_classes):
            rgb = next(v["rgb"] for v in color_dict.values() if v["index"] == target_class_idx)
            self.target_colors[target_class_idx] = torch.tensor(rgb, device=self.device)
        
        self.step_count = 0
    
    def reset_world_at(self, env_index = None):
        """Reset the world for a given environment index."""

        if env_index is None: # Apply to all environements

            self.all_time_covered_targets = torch.full(
                (self.world.batch_dim, self.n_target_classes, self.n_targets_per_class), False, device=self.world.device
            )
            self.targets_pos.zero_()
            self.oneshot_signal.zero_()
            
            # Randomize max target count
            self.num_covered_targets.zero_()
            if self.max_target_objective:
                if self.target_attribute_objective:
                    self.max_target_count = torch.randint(1, self.n_targets_per_class + 1, (self.world.batch_dim,))
                else: 
                    self.max_target_count = torch.randint(1, self.n_targets + 1, (self.world.batch_dim,))

            # Randomize target class (the attribute agents are aiming for)
            if self.target_attribute_objective:
                self.target_class = torch.randint(0, len(self.target_groups), (self.world.batch_dim,), device=self.device)
            
            # Reset Occupancy grid
            if self.use_occupancy_grid_rew:
                self.occupancy_grid.reset_all()
            
            if self.use_expo_search_rew:
                self.covering_rew_val.fill_(1)
                self.covering_rew_val *= self.covering_rew_coeff
            # Reset agents
            for agent in self.world.agents:
                if self.observe_pos_history:
                    agent.position_history.reset_all()
                if self.observe_vel_history:
                    agent.velocity_history.reset_all()

        else:
            self.all_time_covered_targets[env_index] = False
            self.oneshot_signal[env_index] = 0.0
            self.targets_pos[env_index].zero_()

            # Reset Occupancy grid
            if self.use_occupancy_grid_rew:
                self.occupancy_grid.reset_env(env_index)
            
            if self.use_expo_search_rew:
                self.covering_rew_val[env_index] = self.covering_rew_coeff

            # Reset agents
            for agent in self.world.agents:
                if self.observe_pos_history:
                    agent.position_history.reset(env_index)
                if self.observe_vel_history:
                    agent.velocity_history.reset(env_index)

            # Randomize max target count
            self.num_covered_targets[env_index].zero_()
            if self.max_target_objective:
                if self.target_attribute_objective:
                    random_tensor = torch.randint(1, self.n_targets_per_class + 1, (self.world.batch_dim,))
                else: 
                    random_tensor = torch.randint(1, self.n_targets + 1, (self.world.batch_dim,))
                self.max_target_count[env_index] = random_tensor[env_index]

            # Randomize target class
            if self.target_attribute_objective:
                rand = torch.randint(0, len(self.target_groups), (self.world.batch_dim,), device=self.device)
                self.target_class[env_index] = rand[env_index]

        self._spawn_entities_randomly(env_index)
    
    def _spawn_entities_randomly(self, env_index):
        """Spawn agents, targets, and obstacles randomly while ensuring valid distances."""

        if env_index is None:
            env_index = torch.arange(self.world.batch_dim,dtype=torch.int, device=self.device)
        else:
            env_index = torch.tensor(env_index,device=self.device)
            env_index = torch.atleast_1d(env_index)

        #obs_poses, agent_poses, _ = self.occupancy_grid.spawn_map(env_index,self.n_obstacles,self.n_agents,n_targets,self.target_class)
        obs_poses, agent_poses, target_poses = self.occupancy_grid.spawn_llm_map(
            env_index, self.n_obstacles, self.n_agents, self.target_groups, self.target_class, self.max_target_count,heading_sigma=self.heading_sigma
        )

        for i, idx in enumerate(env_index):
            for j, obs in enumerate(self._obstacles):
                obs.set_pos(obs_poses[i,j],batch_index=idx)

            for j, agent in enumerate(self.world.agents):
                agent.set_pos(agent_poses[i,j],batch_index=idx)

            for j, targets in enumerate(self.target_groups):
                for t, target in enumerate(targets):
                    target.set_pos(target_poses[i,j,t],batch_index=idx)

        for target in self._targets[self.n_targets :]:
            target.set_pos(self._get_outside_pos(env_index), batch_index=env_index)
        
    def reward(self, agent: Agent):
        """Compute the reward for a given agent."""

        # Define First and Last flags
        is_first = agent == self.world.agents[0]
        is_last = agent == self.world.agents[-1]
        
        # Initialize individual rewards
        agent.collision_rew[:] = 0
        self.oneshot_rew[:] = 0
        covering_rew = agent.covering_reward if not self.shared_reward else self.shared_covering_rew

        # Compute each reward component separately
        self._compute_collisions(agent)
        pos = agent.state.pos
        exploration_rew = torch.zeros(self.world.batch_dim, device=self.world.device)
        
        # Reward for finding unexplored cells or a heading cell
        if self.use_occupancy_grid_rew:
            if (self.global_heading_objective or self.llm_activate):
                exploration_rew += self.occupancy_grid.compute_gaussian_heading_bonus(pos, heading_exploration_rew_coeff=self.heading_exploration_rew_coeff)
            exploration_rew += self.occupancy_grid.compute_exploration_bonus(pos, exploration_rew_coeff=self.exploration_rew_coeff, new_cell_rew_coeff=self.new_cell_rew_coeff)
            self.occupancy_grid.update(pos)
        
        self.num_covered_targets = self.all_time_covered_targets[torch.arange(0,self.world.batch_dim, device = self.device),self.target_class].sum(dim=-1)
            
        # Reward for finding an aditional target grows exponentially - Why? Because the exploration effort increases
        if self.use_expo_search_rew:
            self.covering_rew_val = torch.exp(self.exponential_search_rew*(self.num_covered_targets + 1) / self.max_target_count) + (self.covering_rew_coeff - 1)
        
        reward = agent.collision_rew + self.time_penalty + (covering_rew + exploration_rew) * (1 - 2 * self.oneshot_signal)  # All postive rewards are inverted once oneshot is on
        
        # Reward applied to the whole team
        if is_first:
            
            # Check if a heading is found, update accordingly
            if self.global_heading_objective or self.llm_activate:
                self.occupancy_grid.update_multi_target_gaussian_heading(self.all_time_covered_targets,self.target_class)
            
            # Collision Penalty
            self._compute_agent_distance_matrix()
            self._compute_covering_rewards()
            
            # Max Target Count Penalty
            if self.max_target_objective or self.llm_activate:
                reached_mask = self.num_covered_targets >= self.max_target_count
                
                all_found_rew = reached_mask * (1 - self.oneshot_signal) * self.terminal_rew_coeff
                reward += all_found_rew
                # Once the max target count is reached, the agent is penalized for moving
                # This is to prevent the agent from moving around once it has found the maximum number of targets
                if reached_mask.any():
                    movement_penalty = torch.sum(agent.state.vel[reached_mask]**2, dim=-1) * self.oneshot_coeff
                    self.oneshot_rew[reached_mask] = movement_penalty
                    self.oneshot_signal[reached_mask] = 1.0
            
        # Remove targets if found
        if is_last: self._handle_target_respawn()

        return reward + self.oneshot_rew

    def observation(self, agent: Agent):
        """Collect Observations from the environment"""
        
        # Collect lidar measurements if enabled
        if self.use_lidar:
            lidar_measures = []
            for sensor in agent.sensors:
                lidar_measures.append(sensor.measure())
            lidar_measures = torch.cat(lidar_measures,dim=-1)

        # Collect normalized position and velocity and get history if enabled
        pos, vel = agent.state.pos / torch.tensor([self.x_semidim, self.y_semidim], device=self.device), agent.state.vel
        pos_hist = agent.position_history.get_flattened() if self.observe_pos_history else None
        vel_hist = agent.velocity_history.get_flattened() if self.observe_vel_history else None

        # Collect all observation components
        obs_components = []
        
        # Sentence Embedding Observation
        if self.llm_activate:
            obs_components.append(self.occupancy_grid.observe_embeddings())
            
        # Targets
        obs_components.append(self.occupancy_grid.get_grid_target_observation(pos,self.mini_grid_radius))
        
        # Histories
        if self.observe_pos_history:
            obs_components.append(pos_hist[: pos.shape[0], :])
            agent.position_history.update(pos)
        if self.observe_vel_history:
            obs_components.append(vel_hist[: vel.shape[0], :])
            agent.velocity_history.update(vel)
            
        # Targeted attribute  
        if self.target_attribute_objective: # or not self.occupancy_grid.target_attribute_embedding_found:
            obs_components.append(self.target_class.unsqueeze(1))
            
        # Max number of targets and Current count (maybe there's a better way for this)
        if self.max_target_objective:
            obs_components.append(self.max_target_count.unsqueeze(1))
            obs_components.append(self.num_covered_targets.unsqueeze(1))
        
        # Heading Observation    
        if self.global_heading_objective:
            obs_components.append(self.occupancy_grid.get_heading_distance_observation(pos))
            
        # Grid Observation (Check out observation_grid.py/heading.py for different options)
        if self.use_occupancy_grid_rew:
            obs_components.append(self.occupancy_grid.get_grid_visits_obstacle_observation(pos,self.mini_grid_radius))

        # Current covered count for exponential search (increasing reward)
        if self.use_expo_search_rew or self.llm_activate:
            obs_components.append(self.num_covered_targets.unsqueeze(1))
        
        # Lidar measurements    
        if self.use_lidar:
            obs_components.append(lidar_measures)
        
        # Pose (GNN works different)    
        if not self.use_gnn:
            obs_components.append(pos)
            obs_components.append(vel)

        # Concatenate observations along last dimension
        obs = torch.cat([comp for comp in obs_components if comp is not None], dim=-1)

        if self.use_gnn:
            return {"obs": obs, "pos": pos, "vel": vel}
        else:
            return obs
    
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
        
    def agent_reward(self, agent):
        """Check for Target Covering and Compute Covering Reward"""
        
        agent.covering_reward[:] = 0
        agent_index = self.world.agents.index(agent)

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
            num_covered_targets_covered_by_agent * self.covering_rew_val.unsqueeze(1) * reward_mask)
        
        if self.target_attribute_objective or self.llm_activate:
            group_rewards += (num_covered_targets_covered_by_agent * self.false_covering_penalty_coeff * (~reward_mask) * self.occupancy_grid.searching_hinted_target.unsqueeze(1))
          # (batch_size, target_groups)

        # Aggregate over target_groups to get (batch_size,)
        agent.covering_reward += group_rewards.sum(dim=-1)
        return agent.covering_reward
    
    def pre_step(self):

        self.step_count += 1
        # Curriculum
        # 1) Once agents have learned that reaching a target can lead to reward, increase penalty for hitting wrong target.
        if (self.step_count % (25 * 200) == 0 and self.false_covering_penalty_coeff > -0.5): # Check this
            self.false_covering_penalty_coeff -= 0.25
            # Progressively decrease the size of the heading region
            # This is to promote faster convergence to the target.
            if self.heading_sigma > 1.0:
                self.heading_sigma -= 0.5
                self.heading_exploration_rew_coeff += 0.5
                
        # Slowly introduce the max target penalty. Only once Agents have learned to cover targets effectively.
        if (self.step_count % (10 * 200) == 0 and self.oneshot_coeff > -20.0):
            self.oneshot_coeff -= 2.0
    
    def extra_render(self, env_index: int = 0) -> "List[Geom]":
        """Render additional visual elements."""
        from vmas.simulator import rendering

        geoms = []
        # Render 
        for i, targets in enumerate(self.target_groups):
            for target in targets:
                range_circle = rendering.make_circle(self._covering_range, filled=False)
                xform = rendering.Transform()
                xform.set_translation(*target.state.pos[env_index])
                range_circle.add_attr(xform)
                color = self.target_colors[i].tolist()  # Convert tensor to list of floats
                range_circle.set_color(*color)
                geoms.append(range_circle)
        
        # Render Occupancy Grid lines
        if self.plot_grid:
            grid = self.occupancy_grid
            for i in range(grid.grid_width + 1):  # Vertical lines
                x = i * grid.cell_size_x - grid.x_dim / 2
                line = rendering.Line((x, -grid.y_dim / 2), (x, grid.y_dim / 2), width=1)
                line.set_color(*Color.BLUE.value)
                geoms.append(line)

            for j in range(grid.grid_height + 1):  # Horizontal lines
                y = j * grid.cell_size_y - grid.y_dim / 2
                line = rendering.Line((-grid.x_dim / 2, y), (grid.x_dim / 2, y), width=1)
                line.set_color(*Color.BLUE.value)
                geoms.append(line)

            # Render grid cells with color based on visit normalization
            heading_grid = grid.grid_heading[env_index,1:-1,1:-1]
            value_grid = grid.grid_visits_sigmoid[env_index,1:-1,1:-1]
            for i in range(heading_grid.shape[1]):
                for j in range(heading_grid.shape[0]):
                    x = i * grid.cell_size_x - grid.x_dim / 2
                    y = j * grid.cell_size_y - grid.y_dim / 2

                    # Heading
                    head = heading_grid[j, i]
                    if False and (self.global_heading_objective or self.llm_activate):
                        heading_lvl = head.item()
                        if heading_lvl >= 0.:
                            color = (self.target_colors[self.target_class[env_index]] * 0.8 * heading_lvl)
                            rect = rendering.FilledPolygon([(x, y), (x + grid.cell_size_x, y), 
                                                            (x + grid.cell_size_x, y + grid.cell_size_y), (x, y + grid.cell_size_y)])
                            rect.set_color(*color)
                            geoms.append(rect)

                    # Visits
                    visit_lvl = value_grid[j, i]
                    if visit_lvl > 0.05 :
                        intensity = visit_lvl.item() * 0.5
                        color = (1.0 - intensity, 1.0 - intensity, 1.0)  # Blueish gradient based on visits
                        rect = rendering.FilledPolygon([(x, y), (x + grid.cell_size_x, y), 
                                                        (x + grid.cell_size_x, y + grid.cell_size_y), (x, y + grid.cell_size_y)])
                        rect.set_color(*color)
                        geoms.append(rect)
                        
        # Render communication lines between agents
        if self.use_gnn:
            for i, agent1 in enumerate(self.world.agents):
                for j, agent2 in enumerate(self.world.agents):
                    if j <= i:
                        continue
                    if self.world.get_distance(agent1, agent2)[env_index] <= self._comms_range:
                        line = rendering.Line(
                            agent1.state.pos[env_index], agent2.state.pos[env_index], width=3
                        )
                        line.set_color(*Color.BLACK.value)
                        geoms.append(line)
        
        # Render Instruction Sentence
        if self.llm_activate:
            try:
                sentence = self.occupancy_grid.sentences[env_index]
                geom = rendering.TextLine(
                    text=sentence,
                    font_size=6
                )
                #geom.label.color = (255, 255, 255, 255)
                xform = rendering.Transform()
                geom.add_attr(xform)
                geoms.append(geom)
            except:
                print("No sentence found for this environment index, or syntax is wrong.")
                pass
            
        return geoms
    
    def done(self):
        """Check if all targets are covered and simulation should end."""
        return torch.tensor([False], device=self.world.device).expand(
            self.world.batch_dim
        )
    
    


    