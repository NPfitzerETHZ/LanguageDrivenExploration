import typing
from typing import List

import torch
import random

from vmas import render_interactively
from vmas.simulator.core import Agent,Landmark, Sphere, Box, World, Line
from vmas.simulator.utils import Color, ScenarioUtils

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom

from myscenario import MyScenario
from spatial_diffusion import SpatialDiffusionOccupancyGrid

class MyGridMapScenario(MyScenario):

    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        """Initialize the world and entities."""
        self.device = device
        self._load_config(kwargs)
        self._load_scenario_config(kwargs)
        world = self._create_world(batch_dim)
        self._create_occupancy_grid(batch_dim)
        self._create_agents(world, batch_dim)
        self._create_targets(world)
        self._create_obstacles(world)
        self._initialize_rewards(batch_dim)
        return world
    
    def _load_scenario_config(self,kwargs):

        self.use_agent_lidar = kwargs.pop("use_agent_lidar", False)
        self.use_obstacle_lidar = kwargs.pop("use_obstacle_lidar", False)
        self.add_obstacles = kwargs.pop("add_obstacles", True)  # This isn't implemented yet

        # Novelty rewards
        self.use_count_rew = kwargs.pop("use_count_rew", False)
        self.use_entropy_rew = kwargs.pop("use_entropy_rew", False)
        self.use_jointentropy_rew = kwargs.pop("use_jointentropy_rew", False)
        self.use_occupancy_grid_rew = kwargs.pop("use_occupencygrid_rew", True)

        self.agent_collision_penalty = kwargs.pop("agent_collision_penalty", -0.0)
        self.obstacle_collision_penalty = kwargs.pop("obstacle_collision_penalty", -0.75)
        self.covering_rew_coeff = kwargs.pop("covering_rew_coeff", 8.0) # Large reward for finding a target
        self.false_covering_penalty_coeff = kwargs.pop("false_covering_penalty_coeff", -0.25) # Penalty for covering wrong target
        self.time_penalty = kwargs.pop("time_penalty", 0)

        #===================
        # Language Driven Goals
        # 1) Count
        self.max_target_objective = kwargs.pop("max_target_objective", False) # Enter as fraction of total number of targets
        # 2) Heading
        self.global_heading_objective = kwargs.pop("global_heading_objective", False)
        # 3) Attribute
        self.target_attribute_objective = kwargs.pop("target_attribute_objective", False)
        #===================

        # Grid
        self.n_obstacles = kwargs.pop("n_obstacle",8)
        self.observe_grid = kwargs.pop("observe_grid",True)
        self.num_grid_cells = 64 # Must be n^2 with n = width 
        self.mini_grid_dim = 3

        self.plot_grid = True
        self.visualize_semidims = False

        # Histories
        self.pos_history_length = 4
        self.pos_dim = 2
        self.observe_pos_history = kwargs.pop("observe_pos_history", True)

    def _create_occupancy_grid(self, batch_dim):

        self.occupancy_grid = SpatialDiffusionOccupancyGrid(
            batch_size=batch_dim,
            x_dim=self.x_semidim*2,
            y_dim=self.y_semidim*2,
            num_cells=self.num_grid_cells,
            num_headings=self.n_targets,
            mini_grid_dim=self.mini_grid_dim,
            device=self.device)

    def _create_obstacles(self, world):

        """Create obstacle landmarks and add them to the world."""
        self._obstacles = 10

        self._obstacles = [
            Landmark(f"obstacle_{i}", collide=True, movable=False, shape=Box(self.occupancy_grid.cell_size_x,self.occupancy_grid.cell_size_y), color=Color.RED)
            for i in range(self.n_obstacles)
        ]
        for obstacle in self._obstacles:
            world.add_landmark(obstacle)
    
    def _create_targets(self, world):
        """Create target landmarks and add them to the world."""

        self.target_groups = []
        self._targets = [
            Landmark(f"target_{i}", collide=False, movable=False, shape=Box(length=self.occupancy_grid.cell_size_y,width=self.occupancy_grid.cell_size_x), color=Color.GREEN)
            for i in range(self.n_targets)
        ]
        for target in self._targets:
            world.add_landmark(target)
        self.target_groups.append(self._targets)
        
        if self.target_attribute_objective:
            self._secondary_targets = [
            Landmark(f"secondary_target_{i}", collide=True, movable=False, shape=Box(length=self.occupancy_grid.cell_size_y,width=self.occupancy_grid.cell_size_x), color=Color.RED)
            for i in range(self.n_targets)
            ]
            for target in self._secondary_targets:
                world.add_landmark(target)
            self.target_groups.append(self._secondary_targets)
    
    def reset_world_at(self, env_index = None):
        """Reset the world for a given environment index."""

        if env_index is None: # Apply to all environements

            self.all_time_covered_targets = torch.full(
                (self.world.batch_dim, self.n_targets), False, device=self.world.device
            )
            self.agent_stopped = torch.full(
                (self.world.batch_dim, self.n_agents), False, device=self.world.device
            )

            # Randomize target class
            if self.target_attribute_objective:
                self.target_class = torch.randint(0, len(self.target_groups), (self.world.batch_dim,), device=self.device)
            
            # Reset Occupancy grid
            if self.use_occupancy_grid_rew:
                self.occupancy_grid.reset_all()

        else:

            self.all_time_covered_targets[env_index] = False
            self.targets_pos[env_index].zero_()

            # Reset Occupancy grid
            if self.use_occupancy_grid_rew:
                self.occupancy_grid.reset_env(env_index)

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
            if self.max_target_objective:
                random_tensor = torch.randint(1, self.n_targets + 1, (self.world.batch_dim,))
                self.max_target_count[env_index] = random_tensor[env_index]
                self.num_covered_targets[env_index].zero_()

            # Randomize target class
            if self.target_attribute_objective:
                rand = torch.randint(0, len(self.target_groups), (self.world.batch_dim,), device=self.device)
                self.target_class[env_index] = rand[env_index]

        self._spawn_entities_randomly(env_index)
    
    def _spawn_entities_randomly(self, env_index):

        """Spawn agents, targets, and obstacles randomly while ensuring valid distances."""

        # First spawn obstacles
        n_targets = self.n_targets + (len(self._secondary_targets) if self.target_attribute_objective else 0)
        obs_poses, agent_poses, target_poses = self.occupancy_grid.spawn_map(env_index,self.n_obstacles,self.n_agents,n_targets,self.target_class)
        self.occupancy_grid.diffusion_update(env_ids=env_index)

        for i in range(obs_poses.shape[0]):
            for j, obs in enumerate(self._obstacles):
                obs.set_pos(obs_poses[i,j],batch_index=env_index)

        for i in range(agent_poses.shape[0]):
            for j, agent in enumerate(self.world.agents):
                agent.set_pos(agent_poses[i,j],batch_index=env_index)
        
        for i in range(target_poses.shape[0]):
            for j, target in enumerate(self._targets):
                target.set_pos(target_poses[i,j],batch_index=env_index)
            if self.target_attribute_objective:
                for j,target in enumerate(self._secondary_targets):
                    target.set_pos(target_poses[i,self.n_targets+j],batch_index=env_index)

        target_poses = []
        for targets in self.target_groups:
            for target in targets[:self.n_targets]:
                target_poses.append(target.state.pos[env_index])
            for target in targets[self.n_targets :]:
                target.set_pos(self._get_outside_pos(env_index), batch_index=env_index)

        # Initialize Headings
        if self.global_heading_objective:
            self.occupancy_grid._initalize_headings(target_poses,self.mini_grid_dim,env_index)
        
    def reward(self, agent: Agent):
        """Compute the reward for a given agent."""
        is_first = agent == self.world.agents[0]
        is_last = agent == self.world.agents[-1]

        if is_first:
            self._compute_agent_distance_matrix()
            self._compute_covering_rewards()

            # Update diffusion grid if needed
            self.occupancy_grid.diffusion_update_count += 1
            mask = (self.occupancy_grid.diffusion_update_count == self.occupancy_grid.diffusion_update_thresh)
            env_ids = mask.nonzero(as_tuple=True)[0]
            if env_ids.numel() > 0:
                self.occupancy_grid.diffusion_update(env_ids)
                self.occupancy_grid.diffusion_update_count[env_ids] = 0


        # Initialize individual rewards
        agent.collision_rew[:] = 0
        agent.oneshot_rew[:] = 0
        covering_rew = agent.covering_reward if not self.shared_reward else self.shared_covering_rew
        covering_rew *= torch.abs(agent.oneshot_signal-1) # All postive rewards are deactivated once oneshot is on

        # Compute each reward component separately
        self._compute_collisions(agent)
        pos = agent.state.pos
        exploration_rew = torch.zeros(self.world.batch_dim, device=self.world.device)
        if self.use_occupancy_grid_rew:
            exploration_rew += self.occupancy_grid.compute_exploration_bonus(pos)
            self.occupancy_grid.update(pos)
        if self.global_heading_objective:
            exploration_rew += self.occupancy_grid.compute_heading_bonus(pos)

        if is_last:
            self._handle_target_respawn()

        return agent.collision_rew + covering_rew + self.time_penalty + agent.oneshot_rew + exploration_rew 

    def observation(self, agent: Agent):

        if self.use_lidar:
            lidar_measures = []
            for sensor in agent.sensors:
                lidar_measures.append(sensor.measure())
            lidar_measures = torch.cat(lidar_measures,dim=-1)

        pos, vel = agent.state.pos, agent.state.vel

        # Get history observations if enabled
        pos_hist = agent.position_history.get_flattened() if self.observe_pos_history else None
        vel_hist = agent.velocity_history.get_flattened() if self.observe_vel_history else None

        # Collect all observation components
        obs_components = [pos, vel]
        obs_components.append(self.occupancy_grid.get_grid_target_observation(pos,self.mini_grid_dim))
        if self.observe_pos_history:
            obs_components.append(pos_hist[: pos.shape[0], :])
        if self.observe_vel_history:
            obs_components.append(vel_hist[: vel.shape[0], :])
        if self.target_attribute_objective:
            obs_components.append(self.target_class.unsqueeze(1))
        if self.max_target_objective:
            obs_components.append(self.max_target_count.unsqueeze(1))
            obs_components.append(self.num_covered_targets.unsqueeze(1)/self.n_targets)
        if self.global_heading_objective:
            #obs_components.append(self.occupancy_grid.get_heading_distance_observation(pos))
            obs_components.append(self.occupancy_grid.get_grid_heading_observation(pos,self.mini_grid_dim))
        if self.use_occupancy_grid_rew:
            #obs_components.append(self.occupancy_grid.get_grid_map_observation(pos,self.mini_grid_dim))
            #obs_components.append(self.occupancy_grid.get_grid_visits_obstacle_observation(pos,self.mini_grid_dim))
            obs_components.append(self.occupancy_grid.get_value_grid_observation(pos,self.mini_grid_dim))
            #obs_components.append(self.occupancy_grid.get_grid_visits_observation(pos,self.mini_grid_dim))
            #obs_components.append(self.occupancy_grid.get_grid_obstacle_observation(pos,self.mini_grid_dim))
            #obs_components.append(self.occupancy_grid.get_flat_grid_pos_from_pos(pos).unsqueeze(1))
        if self.use_lidar:
            obs_components.append(lidar_measures)

        # Concatenate observations along last dimension
        obs = torch.cat([comp for comp in obs_components if comp is not None], dim=-1)

        # Update history buffers if enabled
        if self.observe_pos_history:
            agent.position_history.update(pos)
        if self.observe_vel_history:
            agent.velocity_history.update(vel)

        return obs
    
    def extra_render(self, env_index: int = 0) -> "List[Geom]":
        """Render additional visual elements."""
        from vmas.simulator import rendering

        geoms = []
        for targets in self.target_groups:
            for target in targets:
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
            # for i in range(grid.grid_width):
            #     for j in range(grid.grid_height):
            #         x = i * grid.cell_size_x - grid.x_dim / 2
            #         y = j * grid.cell_size_y - grid.y_dim / 2

            #         # Heading
            #         head = grid.grid_heading[env_index, j, i]
            #         if head == 1:
            #             color = (1.0,1.0,0) #yellow
            #             rect = rendering.FilledPolygon([(x, y), (x + grid.cell_size_x, y), 
            #                                             (x + grid.cell_size_x, y + grid.cell_size_y), (x, y + grid.cell_size_y)])
            #             rect.set_color(*color)
            #             geoms.append(rect)

            #         # Visits
            #         visit_lvl = grid.grid_visits_sigmoid[env_index, j, i]
            #         if visit_lvl > 0.05 :
            #             intensity = visit_lvl.item() * 0.5
            #             color = (1.0 - intensity, 1.0 - intensity, 1.0)  # Blueish gradient based on visits
            #             rect = rendering.FilledPolygon([(x, y), (x + grid.cell_size_x, y), 
            #                                             (x + grid.cell_size_x, y + grid.cell_size_y), (x, y + grid.cell_size_y)])
            #             rect.set_color(*color)
            #             geoms.append(rect)
            value_grid = grid.value_grid[env_index,1:-1,1:-1]
            for i in range(value_grid.shape[1]):
                for j in range(value_grid.shape[0]):
                    x = i * grid.cell_size_x - grid.x_dim / 2
                    y = j * grid.cell_size_y - grid.y_dim / 2

                    # Value
                    visit_lvl = grid.value_grid[env_index, j+1, i+1]
                    if visit_lvl > 0.05 :
                        intensity = visit_lvl.item() * 0.75
                        color = (1.0 - intensity, 1.0 - intensity, 1.0)  # Blueish gradient based on visits
                        rect = rendering.FilledPolygon([(x, y), (x + grid.cell_size_x, y), 
                                                        (x + grid.cell_size_x, y + grid.cell_size_y), (x, y + grid.cell_size_y)])
                        rect.set_color(*color)
                        geoms.append(rect)

        return geoms
