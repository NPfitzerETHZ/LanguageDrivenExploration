import torch
import random
from typing import List
from vmas.simulator.core import Landmark

X = 0
Y = 1

AGENT=0
VISITED=1
OBSTACLE=2
TARGET=1
VISITED_TARGET= -1


# Proposed Change: Instead of a normalized grid, jsut do visited or not visited.

class OccupancyGrid:

    def __init__(self, x_dim, y_dim, num_cells, batch_size, num_targets, heading_mini_grid_radius=1, device='cpu'):

        self.x_dim = x_dim  # World width
        self.y_dim = y_dim  # World height
        self.num_cells = num_cells  # Total number of grid cells
        self.device = device
        self.batch_size = batch_size

        self.grid_width = int(num_cells ** 0.5)  # Assuming a square grid
        self.grid_height = self.grid_width  # Square grid assumption
        self.cell_size_x = self.x_dim / self.grid_width
        self.cell_size_y = self.y_dim / self.grid_height
        self.cell_radius = ((self.cell_size_x/2)**2+(self.cell_size_y/2)**2)**0.5
        self.num_targets = num_targets

        self.visit_threshold = 3 

        self.padded_grid_width = self.grid_width + 2 # Added padding to set obstacles around the env.
        self.padded_grid_height = self.grid_height + 2

        self.border_mask = torch.zeros((self.padded_grid_height, self.padded_grid_width), dtype=torch.bool, device=self.device)
        self.border_mask[0, :] = True
        self.border_mask[-1, :] = True
        self.border_mask[:, 0] = True
        self.border_mask[:, -1] = True

        self.heading_mini_grid_radius = heading_mini_grid_radius
        self.headings = torch.zeros((batch_size, num_targets, 2), dtype=torch.int, device=self.device)
        self.heading_to_target_ratio = 0.75

        ###  MAPS ###
        # grid obstacles
        self.grid_obstacles = torch.zeros((batch_size,self.padded_grid_height, self.padded_grid_width), device=self.device)
        self.grid_obstacles[:, self.border_mask] = OBSTACLE
        # grid targets
        self.grid_targets = torch.zeros((batch_size,self.padded_grid_height, self.padded_grid_width), dtype=torch.int32, device=self.device)
        #grid headings
        self.grid_heading = torch.zeros((batch_size,self.padded_grid_height, self.padded_grid_width), device=self.device)

        # Initialize the visit count grid (keeps track of visits per cell)
        self.grid_visits = torch.zeros((batch_size,self.padded_grid_height, self.padded_grid_width), device=self.device)
        self.grid_visits_sigmoid = torch.zeros((batch_size,self.padded_grid_height, self.padded_grid_width), device=self.device)
        self.grid_visited = torch.zeros((batch_size,self.padded_grid_height, self.padded_grid_width), dtype=torch.int32, device=self.device)
    
    def _initalize_headings(self,target_poses,mini_grid_radius,env_index):

        """
        Create heandings: collections of cells to be targeted
        Rectangles are the set of possible headings in map coordinates
        """

        if env_index is None:
            env_index = torch.arange(self.batch_size,dtype=torch.int, device=self.device)
        else:
            env_index = torch.tensor(env_index,device=self.device).view(-1,1)

        batch_size = env_index.shape[0]

        for i, pos in enumerate(target_poses):
            # Convert world coordinates to grid indices
            rando = random.random()
            if rando < self.heading_to_target_ratio:
                grid_x, grid_y = self.world_to_grid(pos, padding=True)
                randos = torch.randint(-mini_grid_radius,mini_grid_radius+1,(batch_size,2)).float() # Create a bit of chaos
                self.headings[env_index, i, 0] = torch.clamp(grid_x + randos[:, 0], min=0, max=self.grid_width - 1)
                self.headings[env_index, i, 1] = torch.clamp(grid_y + randos[:, 1], min=0, max=self.grid_height - 1)


                x_min = (self.headings[env_index,i,0] - mini_grid_radius).int()
                y_min = (self.headings[env_index,i,1] - mini_grid_radius).int()
                
                x_range = torch.arange(mini_grid_radius*2+1, device=self.device).view(1, -1) + x_min.view(-1, 1)
                y_range = torch.arange(mini_grid_radius*2+1, device=self.device).view(1, -1) + y_min.view(-1, 1)

                x_range = torch.clamp(x_range, min=0, max=self.grid_width - 1)
                y_range = torch.clamp(y_range, min=0, max=self.grid_height - 1)

                self.grid_heading[env_index.unsqueeze(1).unsqueeze(2), y_range.unsqueeze(2), x_range.unsqueeze(1)] = 1
            else:
                self.headings[env_index, i] = VISITED_TARGET
    
    def world_to_grid(self, pos, padding):
        """
        Convert continuous world coordinates to discrete grid coordinates.
        Ensures that the world origin (0,0) maps exactly to the center of the occupancy grid.
        """
        if padding: 
            grid_x = torch.round((pos[..., 0] / self.cell_size_x) + (self.grid_width - 1) / 2).int().clamp(0, self.grid_width - 1) + 1
            grid_y = torch.round((pos[..., 1] / self.cell_size_y) + (self.grid_height - 1) / 2).int().clamp(0, self.grid_height - 1) + 1
        else:
            grid_x = torch.round((pos[..., 0] / self.cell_size_x) + (self.grid_width - 1) / 2).int().clamp(0, self.grid_width - 1)
            grid_y = torch.round((pos[..., 1] / self.cell_size_y) + (self.grid_height - 1) / 2).int().clamp(0, self.grid_height - 1)

        return grid_x, grid_y
    
    def grid_to_world(self, grid_x, grid_y):

        """
        Convert discrete grid coordinates to continuous world coordinates.
        Ensures that the center of each grid cell corresponds to a world coordinate.

        Args:
            grid_x (torch.Tensor): Grid x-coordinates.
            grid_y (torch.Tensor): Grid y-coordinates.

        Returns:
            torch.Tensor: World coordinates (x, y).
        """
        world_x = (grid_x - (self.grid_width - 1) / 2) * self.cell_size_x
        world_y = (grid_y - (self.grid_height - 1) / 2) * self.cell_size_y

        return torch.stack((world_x, world_y), dim=-1)
    
    def spawn_map(self, env_index: torch.Tensor, n_obstacles, n_agents, n_targets, target_class, padding = True):

        if env_index is None:
            env_index = torch.arange(self.batch_size,dtype=torch.int, device=self.device)
        else:
            env_index = torch.tensor(env_index,device=self.device).view(-1,1)
        
        batch_size = env_index.shape[0]
        grid_size = self.grid_width * self.grid_height

        assert grid_size >= n_obstacles + n_agents + n_targets, "Not enough room for all entities"

        # Generate random values and take the indices of the top `n_obstacles` smallest values
        rand_values = torch.rand(batch_size, grid_size, device=self.device)
        sample_start = torch.randint(0, grid_size - n_agents, (batch_size,), device=self.device)
        sample_range = torch.arange(0, n_agents, device=self.device).view(1, -1) + sample_start.view(-1, 1)  # Shape: (batch_size, n_agents)
        batch_indices = torch.arange(batch_size, device=self.device).view(-1, 1).expand_as(sample_range)
        rand_values[batch_indices, sample_range] = float('inf')
        sort_indices = torch.argsort(rand_values, dim=1)

        # Extract obstacle and agent indices
        obstacle_indices = sort_indices[:, :n_obstacles]  # First n_obstacles indices
        target_indices = sort_indices[:,n_obstacles:n_obstacles+n_targets]
        agent_indices = sort_indices[:, -n_agents:]

        # Convert flat indices to (x, y) grid coordinates
        obstacle_grid_x = (obstacle_indices % self.grid_width).view(batch_size, n_obstacles, 1)
        obstacle_grid_y = (obstacle_indices // self.grid_width).view(batch_size, n_obstacles, 1)

        target_grid_x = (target_indices % self.grid_width).view(batch_size, n_targets, 1)
        target_grid_y = (target_indices // self.grid_width).view(batch_size, n_targets, 1)

        agent_grid_x = (agent_indices % self.grid_width).view(batch_size, n_agents, 1)
        agent_grid_y = (agent_indices // self.grid_width).view(batch_size, n_agents, 1)

        # Update grid_obstacles and grid_targets for the given environments and adjust for padding
        if padding:
            self.grid_obstacles[env_index.unsqueeze(1).unsqueeze(2), obstacle_grid_y+1, obstacle_grid_x+1] = OBSTACLE  # Mark obstacles
            self.grid_targets[env_index.unsqueeze(1).unsqueeze(2), target_grid_y+1, target_grid_x+1] = (TARGET + target_class[env_index.unsqueeze(1).unsqueeze(2)]) # Mark targets 
        else:
            self.grid_obstacles[env_index.unsqueeze(1).unsqueeze(2), obstacle_grid_y, obstacle_grid_x] = OBSTACLE  # Mark obstacles
            self.grid_targets[env_index.unsqueeze(1).unsqueeze(2), target_grid_y, target_grid_x] = (TARGET + target_class[env_index.unsqueeze(1).unsqueeze(2)]) # Mark targets 

        # Convert to world coordinates
        obstacle_centers = self.grid_to_world(obstacle_grid_x, obstacle_grid_y)  # Ensure shape (batch_size, n_obstacles, 2)
        target_centers = self.grid_to_world(target_grid_x, target_grid_y)  # Ensure shape (batch_size, n_targets, 2)
        agent_centers = self.grid_to_world(agent_grid_x,agent_grid_y)

        return obstacle_centers.squeeze(-2), agent_centers.squeeze(-2), target_centers.squeeze(-2)
        
    def update(self, agent_positions: torch.Tensor):
        """
        Update the grid and visit count based on the agents' current positions.
        """
        grid_x, grid_y = self.world_to_grid(agent_positions, padding=True)
        batch_indices = torch.arange(agent_positions.shape[0], device=agent_positions.device)  # Shape: (batch,)
        self.grid_visits[batch_indices, grid_y, grid_x] += 1
        self.grid_visits_sigmoid[batch_indices, grid_y, grid_x] = 1/(1+torch.exp(self.visit_threshold - self.grid_visits[batch_indices, grid_y, grid_x]))
        self.grid_visited[batch_indices, grid_y, grid_x] = VISITED

    def update_heading(self, all_time_covered_targets: torch.Tensor):

        # All found heading coordinates are set to a out of bounds value (not sure this will work)

        mask = all_time_covered_targets  # (batch, n_targets)
        env_ids = mask.nonzero(as_tuple=True)[0] # Get batch indices where targets are found
        grid_x, grid_y = self.headings[mask][:, 0], self.headings[mask][:, 1]

        # Mark covered targets with out-of-bounds value (-1)
        self.headings[mask] = VISITED_TARGET

        # Only update for newly discovered headings 
        update_indices = torch.where((grid_x >= 0) & (grid_y >= 0))

        x_min = (grid_x[update_indices] - self.heading_mini_grid_radius)
        y_min = (grid_y[update_indices] - self.heading_mini_grid_radius)

        x_range = torch.arange(self.heading_mini_grid_radius*2+1, device=self.device).view(1, -1) + x_min.view(-1, 1)
        y_range = torch.arange(self.heading_mini_grid_radius*2+1, device=self.device).view(1, -1) + y_min.view(-1, 1)

        # Clamp to avoid out-of-bounds indexing
        x_range = torch.clamp(x_range, min=0, max=self.grid_width)
        y_range = torch.clamp(y_range, min=0, max=self.grid_height)

        self.grid_heading[env_ids[update_indices].unsqueeze(1).unsqueeze(2), y_range.unsqueeze(2), x_range.unsqueeze(1)] = 0
    
    
    # For a multi-target scenario....
    def update_multi_target_gaussian_heading(self, all_time_covered_targets: torch.Tensor):

        # All found heading coordinates are set to a out of bounds value (not sure this will work)

        mask = all_time_covered_targets  # (batch, n_targets)
        self.headings[mask] = VISITED_TARGET
        self.grid_gaussian_heading[mask] = 0.0
        self.grid_heading = self.grid_gaussian_heading.max(dim=1).values
        visit_mask = (self.grid_visited == VISITED)
        self.grid_heading[visit_mask] = 0.0

    def get_grid_visits_observation(self, pos, mini_grid_radius):

        x_range , y_range = self.sample_mini_grid(pos, mini_grid_radius)

        #mini_grid = self.grid_visits_normalized[torch.arange(pos.shape[0]).unsqueeze(1).unsqueeze(2), y_range.unsqueeze(2), x_range.unsqueeze(1)]
        mini_grid = self.grid_visits_sigmoid[torch.arange(pos.shape[0]).unsqueeze(1).unsqueeze(2), y_range.unsqueeze(2), x_range.unsqueeze(1)]

        return mini_grid.flatten(start_dim=1, end_dim=-1)
    
    def get_grid_obstacle_observation(self, pos, mini_grid_radius):

        x_range , y_range = self.sample_mini_grid(pos, mini_grid_radius)

        mini_grid = self.grid_obstacles[torch.arange(pos.shape[0]).unsqueeze(1).unsqueeze(2), y_range.unsqueeze(2), x_range.unsqueeze(1)]

        return mini_grid.flatten(start_dim=1, end_dim=-1)
    
    def get_grid_visits_obstacle_observation(self, pos, mini_grid_radius):

        x_range , y_range = self.sample_mini_grid(pos, mini_grid_radius)

        mini_grid = self.grid_obstacles[torch.arange(pos.shape[0]).unsqueeze(1).unsqueeze(2), y_range.unsqueeze(2), x_range.unsqueeze(1)]
        mini_grid_visited = self.grid_visits_sigmoid[torch.arange(pos.shape[0]).unsqueeze(1).unsqueeze(2), y_range.unsqueeze(2), x_range.unsqueeze(1)]

        mask = torch.where(mini_grid == 0)
        mini_grid[mask] = mini_grid_visited[mask]

        return mini_grid.flatten(start_dim=1, end_dim=-1)
    
    def get_grid_target_observation(self, pos, mini_grid_radius):

        x_range , y_range = self.sample_mini_grid(pos, mini_grid_radius)
        mini_grid = self.grid_targets[torch.arange(pos.shape[0]).unsqueeze(1).unsqueeze(2), y_range.unsqueeze(2), x_range.unsqueeze(1)]

        return mini_grid.flatten(start_dim=1, end_dim=-1)
    
    def get_grid_heading_observation(self, pos, mini_grid_radius):

        x_range , y_range = self.sample_mini_grid(pos, mini_grid_radius)

        mini_grid = self.grid_heading[torch.arange(pos.shape[0]).unsqueeze(1).unsqueeze(2), y_range.unsqueeze(2), x_range.unsqueeze(1)]

        return mini_grid.flatten(start_dim=1, end_dim=-1)

    def get_heading_distance_observation(self,pos):

        grid_x, grid_y = self.world_to_grid(pos, padding=True)

        grid_coords = torch.stack((grid_x, grid_y), dim=1).unsqueeze(1)
        dist = self.headings - grid_coords
        dist[self.headings == VISITED_TARGET] = 0
        #("observation: ",dist.flatten(start_dim=1,end_dim=-1)[0])

        return dist.flatten(start_dim=1,end_dim=-1)
    
    def get_absolute_heading_observation(self):

        return self.headings.flatten(start_dim=1,end_dim=-1)



    def compute_exploration_bonus(self, agent_positions, exploration_rew_coeff = -0.02, new_cell_rew_coeff = 0.25):
        """
        Compute exploration reward: Reward for visiting new cells.
        """
        grid_x, grid_y = self.world_to_grid(agent_positions, padding=True)
        #visits = self.grid_visits_normalized[torch.arange(agent_positions.shape[0]), grid_y, grid_x]
        visit_lvl = self.grid_visits_sigmoid[torch.arange(agent_positions.shape[0]), grid_y, grid_x]
        new_cell_bonus = (visit_lvl < 0.2).float() * new_cell_rew_coeff
        #visits = self.grid_visited[torch.arange(agent_positions.shape[0]), grid_y, grid_x]

        # Works good, negative reward with short postive.
        reward = exploration_rew_coeff * visit_lvl + new_cell_bonus #  Sigmoid Penalty for staying in a visited cell + bonus for discovering a new cell

        # Here I try postive reward only
        #reward = 0.05*(1/(1+torch.exp(visits - 3)))
        #return -0.05 * visit_lvl
        return reward
    
    def compute_heading_bonus(self,pos, heading_exploration_rew_coeff = 2.0):

        grid_x, grid_y = self.world_to_grid(pos, padding=True)

        in_heading_cell = self.grid_heading[torch.arange(pos.shape[0]),grid_y,grid_x]
        visit_lvl = self.grid_visits_sigmoid[torch.arange(pos.shape[0]), grid_y, grid_x]
        heading_val = self.grid_heading[torch.arange(pos.shape[0]),grid_y,grid_x]

        return heading_val
        #return in_heading_cell * heading_exploration_rew_coeff  # Huge reward for discovering a heading cell, but staying there is not beneficial
    
    
    def sample_mini_grid(self,pos,mini_grid_radius):

        grid_x, grid_y = self.world_to_grid(pos, padding=True)
        x_min = (grid_x - mini_grid_radius).int()
        y_min = (grid_y - mini_grid_radius).int()

        x_range = torch.arange(mini_grid_radius*2+1, device=self.device).view(1, -1) + x_min.view(-1, 1)
        y_range = torch.arange(mini_grid_radius*2+1, device=self.device).view(1, -1) + y_min.view(-1, 1)

        # Clamp to avoid out-of-bounds indexing
        x_range = torch.clamp(x_range, min=0, max=self.grid_width)
        y_range = torch.clamp(y_range, min=0, max=self.grid_height)

        return x_range, y_range

    def reset_all(self):
        """
        Reset all the grid and visit counts
        """
        self.grid_visits.zero_()
        self.grid_visits_sigmoid.zero_()
        self.grid_visited.zero_()

        self.grid_targets.zero_()
        self.grid_heading.zero_()
        self.headings.zero_()
    
        self.grid_obstacles.zero_()
        self.grid_obstacles[:,self.border_mask] = OBSTACLE

    def reset_env(self,env_index):
        """
        Reset grid and count for specific envs
        """
        self.grid_visits[env_index].zero_()
        self.grid_visits_sigmoid[env_index].zero_()
        self.grid_visited[env_index].zero_()

        self.grid_targets[env_index].zero_()
        self.grid_heading[env_index].zero_()
        self.headings[env_index].zero_()

        self.grid_obstacles[env_index].zero_()
        self.grid_obstacles[env_index,self.border_mask] = OBSTACLE
