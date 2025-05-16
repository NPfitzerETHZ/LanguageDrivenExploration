import typing
from typing import List

import torch
import numpy as np
import random

from vmas import render_interactively
from vmas.simulator.core import Agent,Landmark, Sphere, Box, World, Line
from vmas.simulator.utils import Color, ScenarioUtils, TorchUtils
from vmas.simulator.controllers.velocity_controller import VelocityController
from vmas.simulator.scenario import BaseScenario

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom

from scenarios.grids.world_occupancy_grid import WorldOccupancyGrid, CoreOccupancyGrid, load_task_data, load_decoder
from scenarios.decentralized.scripts.observation import observation
from scenarios.decentralized.scripts.rewards import compute_reward

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

class MyLanguageScenario(BaseScenario):

    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        """Initialize the world and entities."""
        self.device = device
        self._load_scenario_config(kwargs)
        self._initialize_scenario_vars(batch_dim)
        world = self._create_world(batch_dim)
        self._create_occupancy_grid(batch_dim)
        self._create_agents(world, batch_dim, self.use_velocity_controller, silent = self.comm_dim == 0)
        self._create_targets(world)
        self._create_obstacles(world)
        self._initialize_rewards(batch_dim)
        return world
    
    def _load_scenario_config(self, kwargs):
        
        # === Map & Scenario Layout ===
        self.x_semidim = kwargs.pop("x_semidim", 1.0)
        self.y_semidim = kwargs.pop("y_semidim", 1.0)
        self._min_dist_between_entities = kwargs.pop("min_dist_between_entities", 0.2)
        self._covering_range = kwargs.pop("covering_range", 0.15)
        self._lidar_range = kwargs.pop("lidar_range", 0.15)
        self.agent_radius = kwargs.pop("agent_radius", 0.01)
        self.n_obstacles = kwargs.pop("n_obstacles", 10)

        # === Agent/Target Counts & Behavior ===
        self.n_agents = kwargs.pop("n_agents", 6)
        self._agents_per_target = kwargs.pop("agents_per_target", 1)
        self.agents_stay_at_target = kwargs.pop("agents_stay_at_target", False)
        self.targets_respawn = kwargs.pop("targets_respawn", False)
        self.n_targets_per_class = kwargs.pop("n_targets_per_class", 1)
        self.n_target_classes = kwargs.pop("n_target_classes", 2)
        self.n_targets = self.n_target_classes * self.n_targets_per_class
        self.done_at_termination = kwargs.pop("done_at_termination", True)

        # === Rewards ===
        self.shared_target_reward = kwargs.pop("shared_target_reward", True)
        self.shared_final_reward = kwargs.pop("shared_final_reward", True)
        self.agent_collision_penalty = kwargs.pop("agent_collision_penalty", -0.5)
        self.obstacle_collision_penalty = kwargs.pop("obstacle_collision_penalty", -0.5)
        self.covering_rew_coeff = kwargs.pop("covering_rew_coeff", 5.0)
        self.false_covering_penalty_coeff = kwargs.pop("false_covering_penalty_coeff", -0.25)
        self.time_penalty = kwargs.pop("time_penalty", -0.05)
        self.terminal_rew_coeff = kwargs.pop("terminal_rew_coeff", 15.0)
        self.exponential_search_rew = kwargs.pop("exponential_search_rew_coeff", 1.5)
        self.termination_penalty_coeff = kwargs.pop("termination_penalty_coeff", -5.0)

        # === Exploration Rewards ===
        self.use_count_rew = kwargs.pop("use_count_rew", False)
        self.use_entropy_rew = kwargs.pop("use_entropy_rew", False)
        self.use_jointentropy_rew = kwargs.pop("use_jointentropy_rew", False)
        self.use_occupancy_grid_rew = kwargs.pop("use_occupency_grid_rew", True)
        self.use_expo_search_rew = kwargs.pop("use_expo_search_rew", True)
        self.grid_visit_threshold = kwargs.pop("grid_visit_threshold", 3)
        self.exploration_rew_coeff = kwargs.pop("exploration_rew_coeff", -0.05)
        self.new_cell_rew_coeff = kwargs.pop("new_cell_rew_coeff", 0.0)
        self.heading_exploration_rew_coeff = kwargs.pop("heading_exploration_rew_coeff", 20.0)
        self.heading_sigma_coef = kwargs.pop("heading_sigma_coef", 0.15)
        self.heading_curriculum = kwargs.pop("heading_curriculum", 0.00)

        # === Lidar & Sensing ===
        self.use_lidar = kwargs.pop("use_lidar", False)
        self.use_target_lidar = kwargs.pop("use_target_lidar", False)
        self.use_agent_lidar = kwargs.pop("use_agent_lidar", False)
        self.use_obstacle_lidar = kwargs.pop("use_obstacle_lidar", False)
        self.n_lidar_rays_entities = kwargs.pop("n_lidar_rays_entities", 8)
        self.n_lidar_rays_agents = kwargs.pop("n_lidar_rays_agents", 12)
        self.use_velocity_controller = kwargs.pop("use_velocity_controller", True)

        # === Agent Communication & GNNs ===
        self.use_gnn = kwargs.pop("use_gnn", False)
        self.comm_dim = kwargs.pop("comm_dim", 1)
        self._comms_range = kwargs.pop("comms_radius", 0.35)

        # === Observation Settings ===
        self.known_map = kwargs.pop("known_map", False)
        self.known_agent_pos = kwargs.pop("known_agents", False)
        self.observe_grid = kwargs.pop("observe_grid", True)
        self.observe_targets = kwargs.pop("observe_targets", True)
        self.observe_pos_history = kwargs.pop("observe_pos_history", True)
        self.observe_vel_history = kwargs.pop("observe_vel_history", False)
        self.use_occupancy_grid_obs = kwargs.pop("use_occupency_grid_obs", True)
        self.use_grid_data = kwargs.pop("use_grid_data", True)
        self.use_class_data = kwargs.pop("use_class_data", True)
        self.use_max_targets_data = kwargs.pop("use_max_targets_data", True)

        # === Grid Settings ===
        self.num_corridors = 2
        self.num_grid_cells = kwargs.pop("num_grid_cells", 400)
        self.mini_grid_dim = 3
        self.mini_grid_radius = kwargs.pop("mini_grid_radius", 1)
        self.plot_grid = True
        self.visualize_semidims = False

        # === Movement & Dynamics ===
        self.agent_weight = kwargs.pop("agent_weight", 1.0)
        self.agent_max_speed = kwargs.pop("agent_max_speed", 4.5)
        self.min_collision_distance = kwargs.pop("min_collision_distance", 0.1)

        # === Histories ===
        self.history_length = kwargs.pop("history_length", 2)
        self.pos_history_length = self.history_length
        self.pos_dim = 2
        self.vel_history_length = 30
        self.vel_dim = 2

        # === Language & LLM Goals ===
        self.max_target_objective = kwargs.pop("max_target_objective", False)
        self.global_heading_objective = kwargs.pop("global_heading_objective", False)
        self.target_attribute_objective = kwargs.pop("target_attribute_objective", False)
        self.embedding_size = kwargs.pop("embedding_size", 1024)
        self.llm_activate = kwargs.pop("llm_activate", True)

        if self.max_target_objective or self.global_heading_objective or self.target_attribute_objective:
            print("Make sure to deactivate all Language Driven Goal Simulators")
            self.llm_activate = False

        # === External Inputs ===
        self.data_json_path = kwargs.pop("data_json_path", "")
        self.decoder_model_path = kwargs.pop("decoder_model_path", "")
        self.use_decoder = kwargs.pop("use_decoder", False)

        # === Visuals ===
        self.viewer_zoom = 1
        self.target_color = Color.GREEN
        self.obstacle_color = Color.BLUE

        # Final check
        ScenarioUtils.check_kwargs_consumed(kwargs)

    
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
                
            if self.use_occupancy_grid_obs:
                agent.occupancy_grid = CoreOccupancyGrid(
                    x_dim=self.x_semidim*2,
                    y_dim=self.x_semidim*2, 
                    num_cells=self.num_grid_cells,
                    visit_threshold=self.grid_visit_threshold,
                    batch_size=batch_dim,
                    embedding_size=self.embedding_size,
                    num_targets=self.n_targets,
                    device=self.device)
                
            # Initialize Agent Variables
            agent.collision_rew = torch.zeros(batch_dim, device=self.device)
            agent.covering_reward = agent.collision_rew.clone()
            agent.exploration_rew = agent.collision_rew.clone()
            agent.coverage_rew = agent.collision_rew.clone()
            self._create_agent_state_histories(agent, batch_dim)
            agent.num_covered_targets = torch.zeros(batch_dim, dtype=torch.int, device=self.device)
            agent.termination_rew = torch.zeros(batch_dim, device=self.device)
            agent.termination_signal = torch.zeros(batch_dim, device=self.device)
            world.add_agent(agent)

    def _create_occupancy_grid(self, batch_dim):
        
        # Initialize Important Stuff
        if self.use_decoder: load_decoder(self.decoder_model_path, self.embedding_size, self.device)
        if self.llm_activate: load_task_data(
            json_path=self.data_json_path,
            use_decoder=self.use_decoder,
            use_grid_data=self.use_grid_data,
            use_class_data=self.use_class_data,
            use_max_targets_data=self.use_max_targets_data,
            device=self.device)
        self.occupancy_grid = WorldOccupancyGrid(
            batch_size=batch_dim,
            x_dim=self.x_semidim*2,
            y_dim=self.x_semidim*2, 
            num_cells=self.num_grid_cells,
            num_targets=self.n_targets,
            num_targets_per_class=self.n_targets_per_class,
            visit_threshold=self.grid_visit_threshold,
            embedding_size=self.embedding_size,
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
        
        # Coverage action
        self.coverage_action = {}
    
    def _initialize_rewards(self, batch_dim):

        """Initialize global rewards."""
        self.shared_covering_rew = torch.zeros(batch_dim, device=self.device)
        self.covering_rew_val = torch.ones(batch_dim, device=self.device) * (self.covering_rew_coeff)
    
    def reset_world_at(self, env_index = None):
        """Reset the world for a given environment index."""

        if env_index is None: # Apply to all environements

            self.all_time_covered_targets = torch.full(
                (self.world.batch_dim, self.n_target_classes, self.n_targets_per_class), False, device=self.world.device
            )
            self.targets_pos.zero_()
            
            # Reset Occupancy grid
            self.occupancy_grid.reset_all()
            
            if self.use_expo_search_rew:
                self.covering_rew_val.fill_(1)
                self.covering_rew_val *= self.covering_rew_coeff
                
            # Reset agents
            for agent in self.world.agents:
                agent.occupancy_grid.reset_all()
                agent.num_covered_targets.zero_()
                agent.termination_signal.zero_()
                if self.observe_pos_history:
                    agent.position_history.reset_all()
                if self.observe_vel_history:
                    agent.velocity_history.reset_all()

        else:
            self.all_time_covered_targets[env_index] = False
            self.targets_pos[env_index].zero_()

            # Reset Occupancy grid
            self.occupancy_grid.reset_env(env_index)
            
            if self.use_expo_search_rew:
                self.covering_rew_val[env_index] = self.covering_rew_coeff

            # Reset agents
            for agent in self.world.agents:
                agent.occupancy_grid.reset_env(env_index)
                agent.num_covered_targets[env_index] = 0
                agent.termination_signal[env_index] = 0.0
                if self.observe_pos_history:
                    agent.position_history.reset(env_index)
                if self.observe_vel_history:
                    agent.velocity_history.reset(env_index)

        self._spawn_entities_randomly(env_index)
    
    def _spawn_entities_randomly(self, env_index):
        """Spawn agents, targets, and obstacles randomly while ensuring valid distances."""

        if env_index is None:
            env_index = torch.arange(self.world.batch_dim,dtype=torch.int, device=self.device)
        else:
            env_index = torch.tensor(env_index,device=self.device)
            env_index = torch.atleast_1d(env_index)
            
        obs_poses, agent_poses, target_poses = self.occupancy_grid.spawn_llm_map(
            env_index, self.n_obstacles, self.n_agents, self.target_groups, self.target_class, self.max_target_count,heading_sigma_coef=self.heading_sigma_coef
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
        
        # Update agent occupancy grids with copies of the global occupancy grid
        for agent in self.world.agents:
            if self.use_occupancy_grid_obs:
                
                ### MAPS ###
                agent.occupancy_grid.grid_visits = self.occupancy_grid.grid_visits.clone()
                agent.occupancy_grid.grid_visits_sigmoid = self.occupancy_grid.grid_visits_sigmoid.clone()
                agent.occupancy_grid.grid_targets = self.occupancy_grid.grid_targets.clone()
                agent.occupancy_grid.grid_obstacles = self.occupancy_grid.grid_obstacles.clone()
                
                ### EMBEDDINGS ###
                agent.occupancy_grid.embeddings = self.occupancy_grid.embeddings.clone()
                agent.occupancy_grid.sentences = self.occupancy_grid.sentences.copy()
        
    def reward(self, agent: Agent):
        """Compute the reward for a given agent."""
        return compute_reward(agent,self)

    def observation(self, agent: Agent):
        """Collect Observations from the environment"""
        return observation(agent, self)   
    
    def pre_step(self):

        self.step_count += 1
        # Curriculum
        # 1) Once agents have learned that reaching a target can lead to reward, increase penalty for hitting wrong target.
        if (self.step_count % (25 * 200) == 0 and self.false_covering_penalty_coeff > -0.5): # Check this
            self.false_covering_penalty_coeff -= 0.25
            # Progressively decrease the size of the heading region
            # This is to promote faster convergence to the target.
            if self.heading_sigma_coef > 0.05:
                self.heading_sigma_coef -= self.heading_curriculum
 
                
    def process_action(self, agent: Agent):
        
        if self.comm_dim > 0:
            self.coverage_action[agent.name] = agent.action._c.clone()
            
        # Clamp square to circle
        agent.action.u = TorchUtils.clamp_with_norm(agent.action.u, agent.u_range)

        # Zero small input
        action_norm = torch.linalg.vector_norm(agent.action.u, dim=1)
        agent.action.u[action_norm < 0.08] = 0

        agent.controller.process_force()
        
        
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
                    if self.global_heading_objective or self.llm_activate:
                        heading_lvl = head.item()
                        if heading_lvl >= 0.:
                            if self.n_targets > 0:
                                color = (self.target_colors[self.target_class[env_index]] * 0.8 * heading_lvl * self.num_grid_cells * 0.1)
                            else:
                                # redish gradient based on heading
                                color = (1.0, 1.0 - heading_lvl * self.num_grid_cells * 0.1, 1.0 - heading_lvl * self.num_grid_cells * 0.1)  # Redish gradient based on heading
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
                            agent1.state.pos[env_index], agent2.state.pos[env_index], width=5
                        )
                        line.set_color(*Color.GREEN.value)
                        geoms.append(line)
        
        # Render Instruction Sentence
        if self.llm_activate:
            try:
                sentence = self.occupancy_grid.sentences[env_index]
                geom = rendering.TextLine(
                    text=sentence,
                    font_size=6
                )
                geom.label.color = (255, 255, 255, 255)
                xform = rendering.Transform()
                geom.add_attr(xform)
                geoms.append(geom)
            except:
                print("No sentence found for this environment index, or syntax is wrong.")
                pass
            
        return geoms
    
    def done(self):
        """Check if all targets are covered and simulation should end."""
        
        if self.done_at_termination:
            return self.all_time_covered_targets[torch.arange(0,self.world.batch_dim, device = self.device),self.target_class].all(dim=-1)
        else:
            return torch.tensor([False], device=self.world.device).expand(
                self.world.batch_dim
            )
        
    
    


    