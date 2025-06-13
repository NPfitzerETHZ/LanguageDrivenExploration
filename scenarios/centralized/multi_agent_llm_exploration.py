import typing
from typing import List
import torch

from vmas import render_interactively
from vmas.simulator.core import Agent,Landmark, Sphere, Box, World, Line
from vmas.simulator.utils import Color, ScenarioUtils, TorchUtils
from vmas.simulator.controllers.velocity_controller import VelocityController
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.dynamics.holonomic import Holonomic
from vmas.simulator.sensors import Lidar

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom

from scenarios.kinematic_dynamic_models.kinematic_unicycle import KinematicUnicycle
from scenarios.grids.world_occupancy_grid import WorldOccupancyGrid, load_task_data, load_decoder
from scenarios.centralized.scripts.histories import VelocityHistory, PositionHistory
from scenarios.centralized.scripts.observation import observation
from scenarios.centralized.scripts.rewards import compute_reward
from scenarios.centralized.scripts.load_config import load_scenario_config

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
        load_scenario_config(kwargs,self)
        self._initialize_scenario_vars(batch_dim)
        world = self._create_world(batch_dim)
        self._create_occupancy_grid(batch_dim)
        self._create_agents(world, batch_dim, self.use_velocity_controller, silent = self.comm_dim == 0)
        self._create_targets(world)
        self._create_obstacles(world)
        self._initialize_rewards(batch_dim)
        return world
    
    def _create_world(self, batch_dim: int):
        """Create and return the simulation world."""
        return World(
            batch_dim,
            self.device,
            dt=0.1,
            x_semidim=self.x_semidim,
            y_semidim=self.y_semidim,
            dim_c=self.comm_dim,
            collision_force=500,
            substeps=5,
            linear_friction=self.linear_friction,
            drag=0
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
                u_range=self.agent_u_range,
                f_range=self.agent_f_range,
                v_range=self.agent_v_range,
                sensors=(self._create_agent_sensors(world) if self.use_lidar else []),
                dynamics= (KinematicUnicycle(world,use_velocity_controler) if self.use_kinematic_model else Holonomic()),
                render_action=True,
                color=Color.GREEN
            )
            
            if use_velocity_controler:
                pid_controller_params = [2, 6, 0.002]
                agent.controller = VelocityController(
                    agent, world, pid_controller_params, "standard"
                )
                
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
    
    def _create_agent_sensors(self, world):
        """Create and return sensors for agents."""
        sensors = []
        
        if self.use_target_lidar:
            sensors.append(Lidar(world, n_rays=self.n_lidar_rays_entities, max_range=self.lidar_range, entity_filter=lambda e: e.name.startswith("target"), render_color=Color.GREEN))
        if self.use_obstacle_lidar:
            sensors.append(Lidar(world, n_rays=self.n_lidar_rays_entities, max_range=self.lidar_range, entity_filter=lambda e: e.name.startswith("obstacle"), render_color=Color.BLUE))
        if self.use_agent_lidar:
            sensors.append(Lidar(world, n_rays=self.n_lidar_rays_agents, max_range=self.lidar_range, entity_filter=lambda e: e.name.startswith("agent"), render_color=Color.RED))
        return sensors
    
    def _create_agent_state_histories(self, agent, batch_dim):
        if self.observe_pos_history:
            agent.position_history = PositionHistory(batch_dim,self.pos_history_length, self.pos_dim, self.device)
        if self.observe_vel_history:
            agent.velocity_history = VelocityHistory(batch_dim,self.vel_history_length, self.vel_dim, self.device)

    def _create_occupancy_grid(self, batch_dim):
        
        # Initialize Important Stuff
        if self.use_decoder: load_decoder(self.decoder_model_path, self.embedding_size, self.device)
        if self.llm_activate and (self.use_grid_data or self.use_class_data or self.use_max_targets_data): load_task_data(
            json_path=self.data_json_path,
            use_decoder=self.use_decoder,
            use_grid_data=self.use_grid_data,
            use_class_data=self.use_class_data,
            use_max_targets_data=self.use_max_targets_data,
            use_confidence_data=self.use_confidence_data,
            device=self.device)
        self.occupancy_grid = WorldOccupancyGrid(
            batch_size=batch_dim,
            x_dim=2, # [-1,1]
            y_dim=2, # [-1,1]
            x_scale=self.x_semidim,
            y_scale=self.y_semidim,
            num_cells=self.num_grid_cells,
            num_targets=self.n_targets,
            num_targets_per_class=self.n_targets_per_class,
            visit_threshold=self.grid_visit_threshold,
            embedding_size=self.embedding_size,
            use_embedding_ratio= self.use_embedding_ratio,
            device=self.device)
        self._covering_range = self.occupancy_grid.cell_radius + self.agent_radius

    
    def _create_obstacles(self, world):

        """Create obstacle landmarks and add them to the world."""
        self._obstacles = [
            Landmark(f"obstacle_{i}", collide=True, movable=False, shape=Box(self.occupancy_grid.cell_size_x * self.x_semidim ,self.occupancy_grid.cell_size_y * self.y_semidim), color=Color.RED)
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
                Landmark(f"target_{i}_{j}", collide=False, movable=False, shape=Box(length=self.occupancy_grid.cell_size_y * self.y_semidim ,width=self.occupancy_grid.cell_size_x * self.x_semidim), color=color)
                for j in range(self.n_targets_per_class)
            ]
            self._targets += targets
            self.target_groups.append(targets)
        for target in self._targets:
            world.add_landmark(target)
    
    def _initialize_scenario_vars(self, batch_dim):
        
        self.max_target_count = torch.ones(batch_dim, dtype=torch.int, device=self.device) * self.n_targets_per_class # Initialized to n_targets (ratio)
        self.target_class = torch.zeros(batch_dim, dtype=torch.int, device=self.device)
        self.confidence_level = torch.zeros(batch_dim, dtype=torch.int, device=self.device)
        self.targets_pos = torch.zeros((batch_dim,self.n_target_classes,self.n_targets_per_class,2), device=self.device)
        
        self.covered_targets = torch.zeros(batch_dim, self.n_target_classes, self.n_targets_per_class, device=self.device)
        
        self.target_colors = torch.zeros((self.n_target_classes, 3), device=self.device)
        for target_class_idx in range(self.n_target_classes):
            rgb = next(v["rgb"] for v in color_dict.values() if v["index"] == target_class_idx)
            self.target_colors[target_class_idx] = torch.tensor(rgb, device=self.device)
        
        self.step_count = 0
        self.team_spread = torch.zeros((batch_dim,self.max_steps), device=self.device)
        
        # Coverage action
        self.coverage_action = {}
    
    def _initialize_rewards(self, batch_dim):

        """Initialize global rewards."""
        self.shared_covering_rew = torch.zeros(batch_dim, device=self.device)
        self.covering_rew_val = torch.ones(batch_dim, device=self.device) * (self.covering_rew_coeff)

    def reset_world_at(self, env_index = None):
        """Reset the world for a given environment index."""

        if env_index is None: # Apply to all environements
            
            self.team_spread.zero_()

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
                agent.num_covered_targets.zero_()
                agent.termination_signal.zero_()
                if self.observe_pos_history:
                    agent.position_history.reset_all()
                if self.observe_vel_history:
                    agent.velocity_history.reset_all()

        else:
            self.team_spread[env_index].zero_()
            
            self.all_time_covered_targets[env_index] = False
            self.targets_pos[env_index].zero_()

            # Reset Occupancy grid
            self.occupancy_grid.reset_env(env_index)
            
            if self.use_expo_search_rew:
                self.covering_rew_val[env_index] = self.covering_rew_coeff

            # Reset agents
            for agent in self.world.agents:
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
            env_index, self.n_obstacles, self.n_agents, self.target_groups, self.target_class, self.max_target_count, self.confidence_level
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
        
    def reward(self, agent: Agent):
        """Compute the reward for a given agent."""
        return self.reward_scale_factor * compute_reward(agent,self)

    def observation(self, agent: Agent):
        """Collect Observations from the environment"""
        return observation(agent, self)  
    
    def pre_step(self):
        
        self.step_count += 1
        # Curriculum
        # 1) Once agents have learned that reaching a target can lead to reward, increase penalty for hitting wrong target.
        if (self.step_count % (20 * 250) == 0 and self.false_covering_penalty_coeff > -0.5): # Check this
            self.false_covering_penalty_coeff -= 0.25
            # Progressively decrease the size of the heading region
            # This is to promote faster convergence to the target.
        
        #if (self.step_count % (20 * 250) == 0 and self.agent_collision_penalty > -1.5): # Check this
            #self.agent_collision_penalty -= 0.25
 
                
    def process_action(self, agent: Agent):
        
        if self.comm_dim > 0:
            self.coverage_action[agent.name] = agent.action._c.clone()
            
        # Clamp square to circle
        agent.action.u = TorchUtils.clamp_with_norm(agent.action.u, agent.u_range)

        # Zero small input
        action_norm = torch.linalg.vector_norm(agent.action.u, dim=1)
        agent.action.u[action_norm < 0.005] = 0

        if self.use_velocity_controller and not self.use_kinematic_model:
            agent.controller.process_force()
    
    def post_step(self):

        team_pos = torch.stack(
            [agent.state.pos       
            for agent in self.world.agents],
            dim=1                       
        )                       

        centroid = team_pos.mean(dim=1)     

        disp   = team_pos - centroid.unsqueeze(1) 
        dist2  = (disp * disp).sum(dim=-1)        

        var    = dist2.mean(dim=1)             
        rms    = torch.sqrt(var)  
        
        self.team_spread[:,(self.step_count-1) % self.max_steps] = rms                     

        
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
                x = i * grid.cell_size_x * self.x_semidim - grid.x_dim * self.x_semidim / 2
                line = rendering.Line((x, -grid.y_dim * self.y_semidim / 2), (x, grid.y_dim * self.y_semidim / 2), width=1)
                line.set_color(*Color.BLUE.value)
                geoms.append(line)

            for j in range(grid.grid_height + 1):  # Horizontal lines
                y = j * grid.cell_size_y * self.y_semidim - grid.y_dim * self.y_semidim / 2
                line = rendering.Line((-grid.x_dim * self.x_semidim / 2, y), (grid.x_dim * self.x_semidim / 2, y), width=1)
                line.set_color(*Color.BLUE.value)
                geoms.append(line)

            # Render grid cells with color based on visit normalization
            #heading_grid = grid.grid_heading[env_index,1:-1,1:-1]
            heading_grid = grid.grid_gaussian_heading.max(dim=1).values[env_index,1:-1,1:-1]
            value_grid = grid.grid_visits_sigmoid[env_index,1:-1,1:-1]
            for i in range(heading_grid.shape[1]):
                for j in range(heading_grid.shape[0]):
                    x = i * grid.cell_size_x * self.x_semidim - grid.x_dim * self.x_semidim / 2
                    y = j * grid.cell_size_y * self.y_semidim - grid.y_dim * self.y_semidim / 2

                    # Heading
                    head = heading_grid[j, i]
                    if self.llm_activate:
                        heading_lvl = head.item()
                        if heading_lvl >= 0.:
                            if self.n_targets > 0:
                                #color = (self.target_colors[self.target_class[env_index]] * 0.8 * heading_lvl * self.num_grid_cells * 0.1)
                                color = (self.target_colors[self.target_class[env_index]] * 0.6 * heading_lvl)
                            else:
                                # redish gradient based on heading
                                #color = (1.0, 1.0 - heading_lvl, 1.0 - heading_lvl)
                                color = (1.0, 1.0 - heading_lvl * self.num_grid_cells * 0.1, 1.0 - heading_lvl * self.num_grid_cells * 0.1)  # Redish gradient based on heading
                            rect = rendering.FilledPolygon([(x, y), (x + grid.cell_size_x * self.x_semidim, y), 
                                                            (x + grid.cell_size_x * self.x_semidim, y + grid.cell_size_y * self.y_semidim), (x, y + grid.cell_size_y * self.y_semidim)])
                            rect.set_color(*color)
                            geoms.append(rect)

                    # Visits
                    visit_lvl = value_grid[j, i]
                    if visit_lvl > 0.05 :
                        intensity = visit_lvl.item() * 0.5
                        color = (1.0 - intensity, 1.0 - intensity, 1.0)  # Blueish gradient based on visits
                        rect = rendering.FilledPolygon([(x, y), (x + grid.cell_size_x * self.x_semidim, y), 
                                                        (x + grid.cell_size_x * self.x_semidim, y + grid.cell_size_y * self.y_semidim), (x, y + grid.cell_size_y * self.y_semidim)])
                        rect.set_color(*color)
                        geoms.append(rect)
                        
        # Render communication lines between agents
        if self.use_gnn:
            for i, agent1 in enumerate(self.world.agents):
                for j, agent2 in enumerate(self.world.agents):
                    if j <= i:
                        continue
                    if self.world.get_distance(agent1, agent2)[env_index] <= self._comms_range * (self.x_semidim + self.y_semidim)/2:
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
        
    
    


    