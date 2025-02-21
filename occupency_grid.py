import torch
from typing import List
from vmas.simulator.core import Landmark

class OccupancyGrid:
    def __init__(self, x_dim, y_dim, num_cells, device):
        self.x_dim = x_dim  # World width
        self.y_dim = y_dim  # World height
        self.num_cells = num_cells  # Total number of grid cells
        self.device = device

        self.grid_width = int(num_cells ** 0.5)  # Assuming a square grid
        self.grid_height = self.grid_width  # Square grid assumption
        self.cell_size_x = self.x_dim / self.grid_width
        self.cell_size_y = self.y_dim / self.grid_height

        # Initialize the occupancy grid (0: unexplored, 1: explored or obstacle)
        self.grid = torch.zeros(self.grid_height, self.grid_width, device=self.device)

        # Initialize the visit count grid (keeps track of visits per cell)
        self.visit_grid = torch.zeros(self.grid_height, self.grid_width, dtype=torch.int32, device=self.device)

    def initialize_obstacles(self, obstacles: List[Landmark]):
        """
        Mark grid cells corresponding to obstacles as explored.
        """
        for landmark in obstacles:
            pos = landmark.state.pos  
            width = landmark.shape._width
            height = landmark.shape._height

            grid_x, grid_y = self.world_to_grid(pos)
            grid_width = int(width / self.cell_size_x)
            grid_height = int(height / self.cell_size_y)

            x_min = max(0, grid_x - grid_width // 2)
            x_max = min(self.grid_width, grid_x + grid_width // 2)
            y_min = max(0, grid_y - grid_height // 2)
            y_max = min(self.grid_height, grid_y + grid_height // 2)

            self.grid[y_min:y_max, x_min:x_max] = 1  

    def world_to_grid(self, pos):
        """
        Convert continuous world coordinates to discrete grid coordinates.
        Ensures that the world origin (0,0) maps exactly to the center of the occupancy grid.
        """
        grid_x = ((pos[..., 0] / self.cell_size_x) + (self.grid_width - 1) / 2).long().clamp(0, self.grid_width - 1)
        grid_y = ((pos[..., 1] / self.cell_size_y) + (self.grid_height - 1) / 2).long().clamp(0, self.grid_height - 1)
        return grid_x, grid_y

    def update(self, agent_positions):
        """
        Update the grid and visit count based on the agents' current positions.
        """
        grid_x, grid_y = self.world_to_grid(agent_positions)
        self.grid[grid_y, grid_x] = 1  # Mark cells as explored
        self.visit_grid[grid_y, grid_x] += 1  # Increment visit count

    def compute_exploration_bonus(self, agent_positions):
        """
        Compute exploration reward: Reward for visiting new cells.
        """
        grid_x, grid_y = self.world_to_grid(agent_positions)
        unexplored = self.grid[grid_y, grid_x] == 0  # Check if cell is unexplored
        reward = unexplored.float() * 0.1  # Reward for visiting a new cell
        return reward

    def get_cnn_observation(self):
        """
        Returns the occupancy grid as a tensor suitable for CNN input.
        The output has a shape of (1, grid_height, grid_width) to match CNN input expectations.
        """
        return self.grid.unsqueeze(0)  # Add a channel dimension for CNN input

    def get_mlp_observation(self):
        """
        Returns the occupancy grid as a flattened tensor suitable for MLP input.
        The output has a shape of (grid_width * grid_height,).
        """
        return self.grid.flatten()

    def reset(self):
        """
        Reset the grid and visit count for a new episode.
        """
        self.grid.fill_(0)
        self.visit_grid.fill_(0)
