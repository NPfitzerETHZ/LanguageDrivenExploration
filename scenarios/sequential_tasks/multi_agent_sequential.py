import torch
import typing
from typing import List

from vmas import render_interactively
from vmas.simulator.core import Agent, Landmark, Sphere, Box, World
from vmas.simulator.utils import Color, TorchUtils
from vmas.simulator.controllers.velocity_controller import VelocityController
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.dynamics.holonomic import Holonomic
from vmas.simulator.sensors import Lidar

# Project-specific imports
from scenarios.kinematic_dynamic_models.kinematic_unicycle import KinematicUnicycle
from scenarios.sequential_tasks.scripts.language_grid import (
    LanguageGrid, load_task_data, load_decoder,
    EXPLORE, NAVIGATE, DEFEND_TIGHT, DEFEND_WIDE, IDLE
)
from scenarios.centralized.scripts.histories import VelocityHistory, PositionHistory
from scenarios.sequential_tasks.scripts.observation import observation
from scenarios.sequential_tasks.scripts.reward_exploration import compute_reward as exploration_reward
from scenarios.sequential_tasks.scripts.reward_defend import compute_reward as defend_reward
from scenarios.sequential_tasks.scripts.reward_navigation import compute_reward as navigation_reward
from scenarios.sequential_tasks.scripts.load_config import load_scenario_config

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom

color_dict = {
    "red":      {"rgb": [1.0, 0.0, 0.0], "index": 0},
    "green":    {"rgb": [0.0, 1.0, 0.0], "index": 1},
    "blue":     {"rgb": [0.0, 0.0, 1.0], "index": 2},
    "yellow":   {"rgb": [1.0, 1.0, 0.0], "index": 3},
    "orange":   {"rgb": [1.0, 0.5, 0.0], "index": 4}
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
        self._create_base(world)
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
            init_tensors = lambda: torch.zeros(batch_dim, device=self.device)
            
            # Reward Terms
            agent.collision_rew = init_tensors()
            agent.covering_reward = init_tensors()
            agent.exploration_rew = init_tensors()
            agent.coverage_rew = init_tensors()
            agent.termination_rew = init_tensors()
            agent.nav_pos_rew = init_tensors()
            
            # Useful Variables
            agent.num_covered_targets = init_tensors()
            agent.termination_signal = init_tensors()
            agent.nav_pos_shaping = init_tensors()
            agent.def_dist_shaping = {DEFEND_TIGHT: init_tensors(), DEFEND_WIDE: init_tensors()}
            self._create_agent_state_histories(agent, batch_dim)
            
            # Events
            agent.holding_flag = torch.zeros(batch_dim, dtype=torch.bool, device=self.device)
            agent.spotted_enemy = torch.zeros(batch_dim, dtype=torch.bool, device=self.device)
            agent.on_base = torch.zeros(batch_dim, dtype=torch.bool, device=self.device)
            
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
        if self.llm_activate and self.use_grid_data : load_task_data(
            json_path=self.data_json_path,
            use_decoder=self.use_decoder,
            use_grid_data=self.use_grid_data,
            device=self.device)
        
        self.occupancy_grid = LanguageGrid(
            x_dim=2, # [-1,1]
            y_dim=2, # [-1,1]
            x_scale=self.x_semidim,
            y_scale=self.y_semidim,
            num_cells=self.num_grid_cells,
            batch_size=batch_dim,
            num_targets=self.n_targets,
            num_targets_per_class=self.n_targets_per_class,
            visit_threshold=self.grid_visit_threshold,
            embedding_size=self.embedding_size,
            use_embedding_ratio= self.use_embedding_ratio,
            device=self.device)

    def _create_obstacles(self, world):

        """Create obstacle landmarks and add them to the world."""
        self._obstacles = [
            Landmark(f"obstacle_{i}", collide=True, movable=False, shape=Box(self.occupancy_grid.cell_size_y * self.y_semidim ,self.occupancy_grid.cell_size_x * self.x_semidim), color=Color.RED)
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

    def _create_base(self, world):
        """Create a base landmark and add it to the world."""
        self.base = Landmark(
            name="base",
            collide=False,
            movable=False,
            shape=Box(length=self.occupancy_grid.cell_size_y * self.y_semidim * 3 ,width=self.occupancy_grid.cell_size_x * self.x_semidim * 3),
            color=Color.YELLOW
        )
        world.add_landmark(self.base)
    
    def _initialize_scenario_vars(self, batch_dim):
        
        
        self.target_class = torch.zeros(batch_dim, dtype=torch.int, device=self.device)
        self.targets_pos = torch.zeros((batch_dim,self.n_target_classes,self.n_targets_per_class,2), device=self.device)
        self.flock_target = torch.zeros((batch_dim,2), device=self.device) 
        
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
            self.all_time_agent_covered_targets = torch.full(
                (self.world.batch_dim, self.n_target_classes, self.n_targets_per_class), False, device=self.world.device
            )
            
            self.all_base_reached = torch.full(
                (self.world.batch_dim,), False, device=self.world.device
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
            self.all_time_agent_covered_targets[env_index] = False
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
            env_index = torch.arange(self.world.batch_dim, device=self.device)
        else:
            env_index = torch.atleast_1d(torch.tensor(env_index, device=self.device))

        obs_poses, agent_poses, target_poses = self.occupancy_grid.spawn_llm_map(
            env_index, self.n_obstacles, self.n_agents, self.target_groups, self.target_class
        )

        self._place_obstacles(obs_poses, env_index)
        self._place_agents(agent_poses, env_index)
        self._place_targets(target_poses, env_index)
        self._place_base(env_index)
        self._update_agent_rewards(env_index)

    def _place_obstacles(self, obs_poses, env_index):
        for i, idx in enumerate(env_index):
            for j, obs in enumerate(self._obstacles):
                obs.set_pos(obs_poses[i, j], batch_index=idx)

    def _place_agents(self, agent_poses, env_index):
        for i, idx in enumerate(env_index):
            for j, agent in enumerate(self.world.agents):
                agent.set_pos(agent_poses[i, j], batch_index=idx)

    def _place_targets(self, target_poses, env_index):
        for i, idx in enumerate(env_index):
            for j, targets in enumerate(self.target_groups):
                for t, target in enumerate(targets):
                    target.set_pos(target_poses[i, j, t], batch_index=idx)
            self.targets_pos[idx] = target_poses[i]

        for target in self._targets[self.n_targets:]:
            target.set_pos(self._get_outside_pos(env_index), batch_index=env_index)

    def _place_base(self, env_index):
        for idx in env_index:
            self.base.set_pos(torch.tensor([0, 0], device=self.world.device), batch_index=idx)

    def _update_agent_rewards(self, env_index):
        self.flock_target[env_index] = self.targets_pos[env_index, 0, 0, :]

        dist = torch.full((self.world.batch_dim,), float('inf'), device=self.world.device)
        agent_index = torch.zeros(self.world.batch_dim, dtype=torch.int, device=self.world.device)
        for i, agent in enumerate(self.world.agents):
            
            # Find closest agent to flock target
            dist_new = torch.linalg.vector_norm(
                agent.state.pos[env_index] - self.flock_target[env_index],
                dim=-1
            )
            dist_mask = dist_new < dist[env_index]
            dist[env_index][dist_mask] = dist_new[dist_mask]
            agent_index[env_index][dist_mask] = i
            
            # Spotted enemy event matches defending state, 0: DEFEND WIDE, 1: DEFEND TIGHT

            agent.spotted_enemy[env_index] = (
                self.occupancy_grid.states[env_index] == DEFEND_TIGHT
            )
            
            agent.nav_pos_shaping[env_index] = (
                torch.linalg.vector_norm(agent.state.pos[env_index] - self.base.state.pos[env_index])
                * self.nav_pos_shaping_factor
            )
            for key in agent.def_dist_shaping:
                agent.def_dist_shaping[key][env_index] = (
                    torch.stack([
                        torch.linalg.vector_norm(agent.state.pos[env_index] - a.state.pos[env_index])
                        for a in self.world.agents if a != agent
                    ], dim=0)
                    - self.desired_distance[key]
                ).pow(2).mean(-1) * self.defend_dist_shaping_factor
        
        # Set found flag for agents closest to the flock target
        for i, agent in enumerate(self.world.agents):
            
            agent.holding_flag[env_index] = (
                self.occupancy_grid.states[env_index] == NAVIGATE
            ) | (
                self.occupancy_grid.states[env_index] == DEFEND_TIGHT
            ) | (
                self.occupancy_grid.states[env_index] == DEFEND_WIDE
            )
            agent.holding_flag[env_index] &= (agent_index[env_index] == i)
        
            
    def _handle_target_respawn(self):
        """Handle target respawn and removal for covered targets."""

        for j, _ in enumerate(self.target_groups):
            indices = torch.where(self.target_class == j)[0]
            self.all_time_covered_targets[indices] += self.covered_targets[indices]
            self.all_time_agent_covered_targets[indices] += self.agent_is_covering[indices]
                
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
        # Reward Mask
        rew = torch.zeros(self.world.batch_dim, device=self.world.device)
        mask_explore = self.occupancy_grid.states == EXPLORE
        mask_navigate = self.occupancy_grid.states == NAVIGATE
        mask_defend_tight = self.occupancy_grid.states == DEFEND_TIGHT
        mask_defend_wide = self.occupancy_grid.states == DEFEND_WIDE
        mask_idle = self.occupancy_grid.states == IDLE
        
        # Compute the reward
        rew[mask_idle] = (agent.state.vel[mask_idle] ** 2).sum(dim=-1) * self.termination_penalty_coeff * 2
        rew[mask_explore] = exploration_reward(agent,self)[mask_explore] 
        rew[mask_navigate] = navigation_reward(agent,self)[mask_navigate] * 2.0
        rew[mask_defend_tight] = defend_reward(agent,self, DEFEND_TIGHT)[mask_defend_tight] * 0.3
        rew[mask_defend_wide] = defend_reward(agent,self, DEFEND_WIDE)[mask_defend_wide] * 0.3
        
        # Print mean rewards per task if at step 100
        if self.step_count % 200 == 0:
            print(
                f"Step: {self.step_count}, "
                f"Explore: {rew[mask_explore].mean().item():.2f}, "
                f"Navigate: {rew[mask_navigate].mean().item():.2f}, "
                f"Defend Tight: {rew[mask_defend_tight].mean().item():.2f}, "
                f"Defend Wide: {rew[mask_defend_wide].mean().item():.2f}, "
                f"Idle: {rew[mask_idle].mean().item():.2f}, "
                f"Total: {rew.mean().item():.2f}"
            )

        return self.reward_scale_factor * rew

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
        
    def consturct_event(self):
        """
        This function is a placeholder for the team-event GNN model.
        We collect agent-level events and construct a team-level event
        """
        
        # Holding Flag
        holding_flag = torch.any(
            torch.stack([a.holding_flag for a in self.world.agents], dim=-1),
            dim=-1,
        )
        # Spotted Enemy
        spotted_enemy = torch.any(
            torch.stack([a.spotted_enemy for a in self.world.agents], dim=-1),
            dim=-1,
        )
        # On Base
        on_base = torch.all(
            torch.stack([a.on_base for a in self.world.agents], dim=-1),
            dim=-1,
        )
        
        return torch.stack(
            [holding_flag, spotted_enemy, on_base],
            dim=-1
        ).to(self.world.device)
        
    
    def post_step(self):
        
        
        # If we are using sequence model, compute the next subtask embedding
        if self.llm_activate and self.use_sequence_model:
            # Get the subtask embedding from the RNN model
            env_index = torch.arange(self.world.batch_dim, device=self.world.device)
            event = self.consturct_event()
            self.occupancy_grid.compute_subtask_embedding_from_rnn(env_index, event)

        # Compute team spread
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
        
        task = self.occupancy_grid.states[env_index].item()

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
        if self.plot_grid and task in [EXPLORE]:
            grid = self.occupancy_grid
            for i in range(grid.grid_width + 1):  # Vertical lines
                x = i * grid.cell_size_x * self.x_semidim - grid.x_dim * self.x_semidim / 2
                line = rendering.Line((x, -grid.y_dim* self.y_semidim / 2), (x, grid.y_dim * self.y_semidim / 2), width=1)
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
            value_grid = grid.internal_grid.grid_visits_sigmoid[env_index,1:-1,1:-1]
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
                        if task == EXPLORE:
                            line.set_color(*Color.RED.value)
                        elif task == NAVIGATE:
                            line.set_color(*Color.YELLOW.value)
                        elif task == DEFEND_TIGHT:
                            line.set_color(*Color.BLUE.value) 
                        elif task == DEFEND_WIDE:
                            line.set_color(*Color.PURPLE.value)
                        else: # IDLE
                            line.set_color(*Color.GREEN.value)
                        geoms.append(line)
        
        # Render Instruction Sentence
        if self.llm_activate:
            try:
                sentence = self.occupancy_grid.response[env_index]
                geom = rendering.TextLine(
                    text=sentence,
                    font_size=6
                )
                geom.label.color = (255, 255, 255, 255) if task == EXPLORE else (0, 0, 0, 255)
                xform = rendering.Transform()
                geom.add_attr(xform)
                geoms.append(geom)
            except:
                print("No sentence found for this environment index, or syntax is wrong.")
                pass
            
        return geoms
    
    def done(self):
        """Check if all targets are covered and simulation should end."""
        
        states = self.occupancy_grid.states
        B = self.world.batch_dim
        dones = torch.full((B,), False, device=self.world.device)

        mask_explore = states == EXPLORE
        if mask_explore.any():
            dones[mask_explore] = self.all_time_covered_targets[
                mask_explore, self.target_class[mask_explore]
            ].all(dim=-1)

        # mask_navigate = states == NAVIGATE
        # if mask_navigate.any():
        #     dones[mask_navigate] = self.all_base_reached[mask_navigate]

        # No need to explicitly handle DEFEND_TIGHT and DEFEND_WIDE;
        # `dones` is initialized to False

        return dones

        
    
    


    