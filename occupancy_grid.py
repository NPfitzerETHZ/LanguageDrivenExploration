import torch
from typing import List
from vmas.simulator.core import Landmark

X = 0
Y = 1

class OccupancyGrid:

    def __init__(self, x_dim, y_dim, num_cells, batch_size, device):
        self.x_dim = x_dim  # World width
        self.y_dim = y_dim  # World height
        self.num_cells = num_cells  # Total number of grid cells
        self.device = device
        self.batch_size = batch_size

        self.grid_width = int(num_cells ** 0.5)  # Assuming a square grid
        self.grid_height = self.grid_width  # Square grid assumption
        self.cell_size_x = self.x_dim / self.grid_width
        self.cell_size_y = self.y_dim / self.grid_height

        # Initialize the visit count grid (keeps track of visits per cell)
        self.grid_visits = torch.zeros(batch_size,self.grid_height, self.grid_width, dtype=torch.int32, device=self.device)
        self.grid_visits_normalized = torch.zeros(batch_size,self.grid_height, self.grid_width, dtype=torch.int32, device=self.device)
        self.visit_count = torch.zeros(batch_size,device=self.device)

    def initialize_obstacles(self, obstacles: List[Landmark]): # This is dumb
        """
        Mark grid cells corresponding to obstacles as explored.
        """
        for landmark in obstacles:
            pos = landmark.state.pos  
            width = landmark.shape._width
            height = landmark.shape._height

            grid_x, grid_y = self.world_to_grid(pos)
            x_dim = int(width / self.cell_size_x)
            y_dim = int(height / self.cell_size_y)

            x_min = torch.clamp(grid_x - x_dim // 2, min=0).int()
            y_min = torch.clamp(grid_y - y_dim // 2, min=0).int()

            x_range = torch.arange(x_dim, device=self.grid_visits.device).view(1, -1) + x_min.view(-1, 1)
            y_range = torch.arange(y_dim, device=self.grid_visits.device).view(1, -1) + y_min.view(-1, 1)

            x_range = torch.clamp(x_range, min=0, max=self.grid_width - 1)
            y_range = torch.clamp(y_range, min=0, max=self.grid_height - 1)

            self.grid_visits_normalized[torch.arange(pos.shape[0]).unsqueeze(1).unsqueeze(2), y_range.unsqueeze(2), x_range.unsqueeze(1)] = 1
    
    def initialize_obstacles_single_env(self, env_index, obstacles: List[Landmark]): # This is dumb
        """
        Mark grid cells corresponding to obstacles as explored.
        """
        for landmark in obstacles:
            pos = landmark.state.pos[env_index]  
            width = landmark.shape._width
            height = landmark.shape._height

            grid_x, grid_y = self.world_to_grid(pos)
            x_dim = int(width / self.cell_size_x)
            y_dim = int(height / self.cell_size_y)

            x_min = torch.clamp(grid_x - x_dim // 2, min=0).int()
            y_min = torch.clamp(grid_y - y_dim // 2, min=0).int()

            x_range = torch.arange(x_dim, device=self.grid_visits.device).view(1, -1) + x_min.view(-1, 1)
            y_range = torch.arange(y_dim, device=self.grid_visits.device).view(1, -1) + y_min.view(-1, 1)
            
            x_range = torch.clamp(x_range, min=0, max=self.grid_width - 1)
            y_range = torch.clamp(y_range, min=0, max=self.grid_height - 1)

            self.grid_visits_normalized[env_index, y_range, x_range] = 1

    def world_to_grid(self, pos):
        """
        Convert continuous world coordinates to discrete grid coordinates.
        Ensures that the world origin (0,0) maps exactly to the center of the occupancy grid.
        """
        grid_x = torch.round((pos[..., 0] / self.cell_size_x) + (self.grid_width - 1) / 2).int().clamp(0, self.grid_width - 1)
        grid_y = torch.round((pos[..., 1] / self.cell_size_y) + (self.grid_height - 1) / 2).int().clamp(0, self.grid_height - 1)
        return grid_x, grid_y

    def update(self, agent_positions: torch.Tensor):
        """
        Update the grid and visit count based on the agents' current positions.
        """
        grid_x, grid_y = self.world_to_grid(agent_positions)
        self.visit_count[:] += 1
        batch_indices = torch.arange(agent_positions.shape[0], device=agent_positions.device)  # Shape: (batch,)
        self.grid_visits[batch_indices, grid_y, grid_x] += 1
        self.grid_visits_normalized = self.grid_visits / self.visit_count.unsqueeze(-1).unsqueeze(-1)
    
    def get_observation_normalized(self, pos, mini_grid_dim):

        grid_x, grid_y = self.world_to_grid(pos)
        x_min = torch.clamp(grid_x - mini_grid_dim // 2, min=0).int()
        y_min = torch.clamp(grid_y - mini_grid_dim // 2, min=0).int()

        x_range = torch.arange(mini_grid_dim, device=self.grid_visits.device).view(1, -1) + x_min.view(-1, 1)
        y_range = torch.arange(mini_grid_dim, device=self.grid_visits.device).view(1, -1) + y_min.view(-1, 1)

        # Clamp to avoid out-of-bounds indexing
        x_range = torch.clamp(x_range, min=0, max=self.grid_width - 1)
        y_range = torch.clamp(y_range, min=0, max=self.grid_height - 1)

        mini_grid = self.grid_visits[torch.arange(pos.shape[0]).unsqueeze(1).unsqueeze(2), y_range.unsqueeze(2), x_range.unsqueeze(1)]

        visit_count_expanded = self.visit_count.unsqueeze(-1).unsqueeze(-1)  # Ensure it matches (N, 1, 1)
        mini_grid_normalized = mini_grid.float() / (visit_count_expanded + 1e-6)  # Avoid division by zero
        return mini_grid_normalized.flatten(start_dim=1, end_dim=-1)

    def compute_exploration_bonus(self, agent_positions):
        """
        Compute exploration reward: Reward for visiting new cells.
        """
        grid_x, grid_y = self.world_to_grid(agent_positions)
        visits = self.grid_visits[torch.arange(agent_positions.shape[0]), grid_y, grid_x]
        visit_count_safe = self.visit_count + 1e-6  # Avoid division errors
        visits_norm = visits / visit_count_safe
        reward = 0.05*(1 - visits_norm) # Reward for visiting a new cell
        return reward

    def reset_all(self):
        """
        Reset all the grid and visit counts
        """
        self.grid_visits.zero_()
        self.visit_count.zero_()
    
    def reset_env(self,env_index):
        """
        Reset grid and count for specific envs
        """
        self.grid_visits[env_index].zero_()
        self.visit_count[env_index] = 0

