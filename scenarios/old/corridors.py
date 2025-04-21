import typing
from typing import List

import torch
import random

from vmas import render_interactively
from vmas.simulator.core import Agent,Landmark, Sphere, Box, World, Line
from vmas.simulator.utils import Color

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom

from scenarios.myscenario import MyScenario

class MyCorridorScenario(MyScenario):

    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        """Initialize the world and entities."""
        self.device = device
        self._load_config(kwargs)
        world = self._create_world(batch_dim)
        self._create_occupancy_grid(batch_dim)
        self.spawn_map(world)
        self._create_agents(world, batch_dim)
        self._create_targets(world)
        self._create_obstacles(world)
        self._initialize_rewards(batch_dim)
        return world
    
    def spawn_map(self, world: World):
        """Spawns the map with horizontal and vertical walls, and defines hotspots."""
        
        # Compute wall lengths
        self.h_wall_length = (2 * self.x_semidim) / (self.num_corridors * 2 + 1) # one cell width
        v_wall_length = (2 * self.y_semidim) / (self.num_corridors * 2 + 1) # one cell height

        self.corridor_length = ((self.num_corridors * 2 + 1) // 2)*v_wall_length
        self.main_corridor_length = ((self.num_corridors * 2 + 1) % 2)*v_wall_length

        # Create horizontal walls
        self.h_walls = [
            Landmark(
                name=f"wall_h {i}",
                collide=True,
                shape=Line(length=self.h_wall_length),
                color=Color.BLACK,
            )
            for i in range((self.num_corridors * 2 + 1) * 2)
        ]
        
        for wall in self.h_walls:
            world.add_landmark(wall)

        # Create vertical walls
        self.v_walls = [
            Landmark(
                name=f"wall_v {i}",
                collide=True,
                shape=Line(length=self.main_corridor_length if i < 2 else self.corridor_length),
                color=Color.BLACK,
            )
            for i in range(2 + 4 * self.num_corridors)
        ]

        for wall in self.v_walls:
            world.add_landmark(wall)
        
        # Define Hotspots
        self.hotspots = [
            (-self.x_semidim + self.agent_radius, 0),  # East extremity
            (self.x_semidim - self.agent_radius, 0)    # West extremity
        ]
        
        # North/South hotspots
        self.hotspots += [
            (
                -self.x_semidim + 2 * self.h_wall_length * (i // 2 + 3/4),
                (1 if i % 2 == 0 else -1) * (self.y_semidim - self.agent_radius)
            )
            for i in range(self.num_corridors * 2)
        ]

        # Rectangles used to initalize the occupancy grid and create heading regions (although that could change later)
        self.rectangles = []
        self.rectangles += [
            (
                -self.x_semidim,-self.main_corridor_length/2.1,2*self.x_semidim,self.main_corridor_length
            )
        ]
        for i in range(self.num_corridors):
            self.rectangles.append((-self.x_semidim + self.h_wall_length * (2*i  + 1) + 1e-6,-self.y_semidim,self.h_wall_length,2*self.y_semidim))

        #self.reset_map(env_index=None)
        #self.occupancy_grid._initalize_headings(self.rectangles)
        self.occupancy_grid._initialize_rectangle_mask(self.rectangles)

    def reset_map(self, env_index):

        for i, landmark in enumerate(self.h_walls):

            landmark.set_pos(
                torch.tensor(
                    [
                        (
                            -self.x_semidim + (i // 2) * self.h_wall_length + self.h_wall_length / 2
                        ),
                        (1 if i % 2 == 0 else -1) * (self.main_corridor_length * 1/2 if (i // 2) % 2 == 0 else self.y_semidim) 
                    ],
                    dtype=torch.float32,
                    device=self.device,
                ),
                batch_index=env_index,
            )
            landmark.set_rot(
                torch.tensor(
                    [0],
                    dtype=torch.float32,
                    device=self.device,
                ),
                batch_index=env_index,
            )
        
        for i, landmark in enumerate(self.v_walls):

            landmark.set_rot(
                    torch.tensor(
                        [torch.pi/2],
                        dtype=torch.float32,
                        device=self.device,
                    ),
                    batch_index=env_index,
                )

            if i < 2: # Extremities

                landmark.set_pos(
                    torch.tensor(
                        [
                            (
                                -self.x_semidim + i * self.x_semidim * 2
                            ),
                            0
                        ],
                        dtype=torch.float32,
                        device=self.device,
                    ),
                    batch_index=env_index,
                )

            else:
                landmark.set_pos(
                    torch.tensor(
                        [
                            (
                                -self.x_semidim + (i // 2) * self.h_wall_length
                            ),
                            (1 if i % 2 == 0 else -1) * (self.y_semidim-self.corridor_length/2)
                        ],
                        dtype=torch.float32,
                        device=self.device,
                    ),
                    batch_index=env_index,
                )

    def reset_world_at(self, env_index = None):
        """Reset the world for a given environment index."""

        if env_index is None: # Apply to all environements

            self.all_time_covered_targets = torch.full(
                (self.world.batch_dim, self.n_targets), False, device=self.world.device
            )
            self.agent_stopped = torch.full(
                (self.world.batch_dim, self.n_agents), False, device=self.world.device
            )

            # Randomize search coordinates
            # if self.global_heading_objective:
            #     indices = torch.randint(0, self.num_headings, (self.world.batch_dim,), device=self.device)
            #     self.search_coordinates = self.possible_coordinates[indices]
            #     self.search_encoding = torch.stack([(indices >> 1) & 1, indices & 1], dim=-1)
            #     self.heading_landmark.set_pos(self.search_coordinates)

            # Randomize target class
            if self.target_attribute_objective:
                self.target_class = torch.randint(0, len(self.target_groups), (self.world.batch_dim,), device=self.device)

            # Initialize occupency gird
            if self.use_occupancy_grid_rew:
                self.occupancy_grid.apply_rectangle_mask(env_index=None)
        else:

            self.reset_map(env_index)
            
            self.all_time_covered_targets[env_index] = False
            self.targets_pos[env_index].zero_()

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
                rand = torch.randint(0, len(self.target_groups), (self.world.batch_dim,), device=self.device)
                self.target_class[env_index] = rand[env_index]

        self._spawn_entities_randomly(env_index)
    
    def _spawn_entities_randomly(self, env_index):

        """Spawn agents, targets, and obstacles randomly while ensuring valid distances."""
        entities = self._targets[: self.n_targets] + self.world.agents
        if self.add_obstacles:
            entities += self._obstacles[: self.n_obstacles]
        if self.target_attribute_objective:
            entities += self._secondary_targets[: self.n_targets]

        assert len(entities) <= len(self.hotspots), "Not enough hotspots for all entities"
        
        # Shuffle hotspots to ensure randomness
        random.shuffle(self.hotspots)
        
        # Assign each entity a unique hotspot
        for entity, hotspot in zip(entities, self.hotspots):
            entity.set_pos(torch.tensor(hotspot),batch_index=env_index)
        
        # Initialize occupancy grid with new obstacle positions
        if self.use_occupancy_grid_rew:
            self.occupancy_grid.apply_rectangle_mask(env_index)
    
    def reward(self, agent: Agent):
        """Compute the reward for a given agent."""
        is_first = agent == self.world.agents[0]
        is_last = agent == self.world.agents[-1]

        if is_first:
            self._compute_agent_distance_matrix()
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
            self._handle_target_respawn()

        return agent.collision_rew + covering_rew + self.time_penalty + agent.oneshot_rew + novelty_rew + ld_rew

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
            for i in range(grid.grid_width):
                for j in range(grid.grid_height):
                    x = i * grid.cell_size_x - grid.x_dim / 2
                    y = j * grid.cell_size_y - grid.y_dim / 2
                    count = grid.grid_visits[env_index, j, i]
                    intensity = 1/(1+torch.exp(5 - count)).item() * 0.5
                    color = (1.0 - intensity, 1.0 - intensity, 1.0)  # Blueish gradient based on visits
                    rect = rendering.FilledPolygon([(x, y), (x + grid.cell_size_x, y), 
                                                    (x + grid.cell_size_x, y + grid.cell_size_y), (x, y + grid.cell_size_y)])
                    rect.set_color(*color)
                    geoms.append(rect)

        return geoms