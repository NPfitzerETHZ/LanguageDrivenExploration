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

class CoreOccupancyGrid:

    def __init__(self, x_dim, y_dim, x_scale, y_scale, num_cells, visit_threshold, batch_size, num_targets, embedding_size, device='cpu'):

        self.x_dim = x_dim  # World width normalized
        self.y_dim = y_dim  # World height normalized
        self.x_scale = x_scale # World width scale
        self.y_scale = y_scale # World height scale
        
        self.num_cells = num_cells  # Total number of grid cells
        self.device = device
        self.batch_size = batch_size

        self.grid_width = int(num_cells ** 0.5)  # Assuming a square grid
        self.grid_height = self.grid_width  # Square grid assumption
        self.cell_size_x = self.x_dim / self.grid_width
        self.cell_size_y = self.y_dim / self.grid_height
        self.cell_radius = ((self.cell_size_x/2)**2+(self.cell_size_y/2)**2)**0.5
        self.num_targets = num_targets
        
        self.visit_threshold = visit_threshold

        self.padded_grid_width = self.grid_width + 2 # Added padding to set obstacles around the env.
        self.padded_grid_height = self.grid_height + 2

        self.border_mask = torch.zeros((self.padded_grid_height, self.padded_grid_width), dtype=torch.bool, device=self.device)
        self.border_mask[0, :] = True
        self.border_mask[-1, :] = True
        self.border_mask[:, 0] = True
        self.border_mask[:, -1] = True

        ###  MAPS ###
        # grid obstacles
        self.grid_obstacles = torch.zeros((batch_size,self.padded_grid_height, self.padded_grid_width), device=self.device)
        self.grid_obstacles[:, self.border_mask] = OBSTACLE
        # grid targets
        self.grid_targets = torch.zeros((batch_size,self.padded_grid_height, self.padded_grid_width), dtype=torch.int32, device=self.device)
        # visits
        self.grid_visits = torch.zeros((batch_size,self.padded_grid_height, self.padded_grid_width), device=self.device)
        self.grid_visits_sigmoid = torch.zeros((batch_size,self.padded_grid_height, self.padded_grid_width), device=self.device)
        
        ### EMBEDDING ###
        self.embedding_size = embedding_size
        self.embeddings = torch.zeros((self.batch_size,embedding_size),device=self.device)
        self.sentences = [ "" for _ in range(self.batch_size)]
    
    def world_to_grid(self, pos, padding):
        """
        Convert continuous world coordinates to discrete grid coordinates.
        Ensures that the world origin (0,0) maps exactly to the center of the occupancy grid.
        """
        if padding: 
            grid_x = torch.round((pos[..., 0] / (self.cell_size_x * self.x_scale)) + (self.grid_width - 1) / 2).int().clamp(0, self.grid_width - 1) + 1
            grid_y = torch.round((pos[..., 1] / (self.cell_size_y * self.y_scale)) + (self.grid_height - 1) / 2).int().clamp(0, self.grid_height - 1) + 1
        else:
            grid_x = torch.round((pos[..., 0] / (self.cell_size_x * self.x_scale)) + (self.grid_width - 1) / 2).int().clamp(0, self.grid_width - 1)
            grid_y = torch.round((pos[..., 1] / (self.cell_size_y * self.y_scale)) + (self.grid_height - 1) / 2).int().clamp(0, self.grid_height - 1)

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
        world_x = (grid_x - (self.grid_width - 1) / 2) * self.cell_size_x * self.x_scale
        world_y = (grid_y - (self.grid_height - 1) / 2) * self.cell_size_y * self.y_scale

        return torch.stack((world_x, world_y), dim=-1)
        
    def update(self, agent_positions: torch.Tensor):
        """
        Update the grid and visit count based on the agents' current positions.
        """
        grid_x, grid_y = self.world_to_grid(agent_positions, padding=True)
        batch_indices = torch.arange(agent_positions.shape[0], device=agent_positions.device)  # Shape: (batch,)
        self.grid_visits[batch_indices, grid_y, grid_x] += 1
        self.grid_visits_sigmoid[batch_indices, grid_y, grid_x] = 1/(1+torch.exp(self.visit_threshold - self.grid_visits[batch_indices, grid_y, grid_x]))

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

    def compute_exploration_bonus(self, agent_positions, exploration_rew_coeff = -0.02, new_cell_rew_coeff = 0.25):
        """
        Compute exploration reward: Reward for visiting new cells.
        """
        grid_x, grid_y = self.world_to_grid(agent_positions, padding=True)
        #visits = self.grid_visits_normalized[torch.arange(agent_positions.shape[0]), grid_y, grid_x]
        visit_lvl = self.grid_visits_sigmoid[torch.arange(agent_positions.shape[0]), grid_y, grid_x]
        new_cell_bonus = (visit_lvl < 0.2).float() * new_cell_rew_coeff

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
    
    def observe_embeddings(self):

        return self.embeddings.flatten(start_dim=1,end_dim=-1)
    
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

        self.grid_targets.zero_()
    
        self.grid_obstacles.zero_()
        self.grid_obstacles[:,self.border_mask] = OBSTACLE
        
        self.sentences = [ ""  for _ in range(self.batch_size)]
        self.embeddings.zero_()

    def reset_env(self,env_index):
        """
        Reset grid and count for specific envs
        """
        self.grid_visits[env_index].zero_()
        self.grid_visits_sigmoid[env_index].zero_()

        self.grid_targets[env_index].zero_()

        self.grid_obstacles[env_index].zero_()
        self.grid_obstacles[env_index,self.border_mask] = OBSTACLE
        
        self.sentences[env_index] = ""
        self.embeddings[env_index].zero_()
