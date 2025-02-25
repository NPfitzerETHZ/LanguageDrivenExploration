import torch
from typing import List
from vmas.simulator.core import Landmark

X = 0
Y = 1


# Proposed Change: Instead of a normalized grid, jsut do visited or not visited.

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
        #self.grid_visits_normalized = torch.zeros(batch_size,self.grid_height, self.grid_width, dtype=torch.int32, device=self.device)
        self.grid_visited = torch.zeros(batch_size,self.grid_height, self.grid_width, dtype=torch.int32, device=self.device)
        #self.visit_count = torch.zeros(batch_size,device=self.device)

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

            x_range = torch.arange(x_dim, device=self.device).view(1, -1) + x_min.view(-1, 1)
            y_range = torch.arange(y_dim, device=self.device).view(1, -1) + y_min.view(-1, 1)

            x_range = torch.clamp(x_range, min=0, max=self.grid_width - 1)
            y_range = torch.clamp(y_range, min=0, max=self.grid_height - 1)

            self.grid_visited[torch.arange(pos.shape[0]).unsqueeze(1).unsqueeze(2), y_range.unsqueeze(2), x_range.unsqueeze(1)] = 1
    
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

            x_range = torch.arange(x_dim, device=self.device).view(1, -1) + x_min.view(-1, 1)
            y_range = torch.arange(y_dim, device=self.device).view(1, -1) + y_min.view(-1, 1)
            
            x_range = torch.clamp(x_range, min=0, max=self.grid_width - 1)
            y_range = torch.clamp(y_range, min=0, max=self.grid_height - 1)

            self.grid_visited[env_index, y_range, x_range] = 1

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
        #self.visit_count[:] += 1
        batch_indices = torch.arange(agent_positions.shape[0], device=agent_positions.device)  # Shape: (batch,)
        self.grid_visits[batch_indices, grid_y, grid_x] += 1
        #self.grid_visits_normalized = self.grid_visits / (self.visit_count.unsqueeze(-1).unsqueeze(-1) + 1e-6)
        self.grid_visited[batch_indices, grid_y, grid_x] = 1
    
    def get_observation_normalized(self, pos, mini_grid_dim):

        grid_x, grid_y = self.world_to_grid(pos)
        x_min = torch.clamp(grid_x - mini_grid_dim // 2, min=0).int()
        y_min = torch.clamp(grid_y - mini_grid_dim // 2, min=0).int()

        x_range = torch.arange(mini_grid_dim, device=self.device).view(1, -1) + x_min.view(-1, 1)
        y_range = torch.arange(mini_grid_dim, device=self.device).view(1, -1) + y_min.view(-1, 1)

        # Clamp to avoid out-of-bounds indexing
        x_range = torch.clamp(x_range, min=0, max=self.grid_width - 1)
        y_range = torch.clamp(y_range, min=0, max=self.grid_height - 1)

        #mini_grid = self.grid_visits_normalized[torch.arange(pos.shape[0]).unsqueeze(1).unsqueeze(2), y_range.unsqueeze(2), x_range.unsqueeze(1)]
        mini_grid = self.grid_visited[torch.arange(pos.shape[0]).unsqueeze(1).unsqueeze(2), y_range.unsqueeze(2), x_range.unsqueeze(1)]

        return mini_grid.flatten(start_dim=1, end_dim=-1)

    def compute_exploration_bonus(self, agent_positions):
        """
        Compute exploration reward: Reward for visiting new cells.
        """
        grid_x, grid_y = self.world_to_grid(agent_positions)
        #visits = self.grid_visits_normalized[torch.arange(agent_positions.shape[0]), grid_y, grid_x]
        visits = self.grid_visits[torch.arange(agent_positions.shape[0]), grid_y, grid_x]
        #visits = self.grid_visited[torch.arange(agent_positions.shape[0]), grid_y, grid_x]
        reward = 0.02*(-1* 1/(1+torch.exp(5 - visits))) #  Sigmoid Penalty for staying in a visited cell
        return reward
    
    def apply_rectangle_mask(self, env_index, rectangles):
        """
        Modifies self.grid_visits for a given batch index (env_index).
        - Sets grid cells inside the given rectangles to 0.
        - Sets grid cells outside these rectangles to 10.

        Parameters:
        - env_index (int): The index in the batch.
        - rectangles (list of tuples): Each tuple is (min_x, min_y, width, height).
        """

        if env_index is None:
            env_index = torch.arange(self.batch_size, device=self.device)

        # Set the entire grid to 10 initially
        self.grid_visits[env_index].fill_(10)
        self.grid_visited[env_index].fill_(1)

        for min_x, min_y, width, height in rectangles:
            # Convert world coordinates to grid indices
            grid_x_min, grid_y_min = self.world_to_grid((min_x, min_y))
            grid_x_max, grid_y_max = self.world_to_grid((min_x + width, min_y + height))

            # Use torch.clamp to ensure indices stay within bounds
            grid_x_min = torch.clamp(grid_x_min, 0, self.grid_width - 1)
            grid_x_max = torch.clamp(grid_x_max, 0, self.grid_width - 1)
            grid_y_min = torch.clamp(grid_y_min, 0, self.grid_height - 1)
            grid_y_max = torch.clamp(grid_y_max, 0, self.grid_height - 1)

            # Set the corresponding grid region to 0
            self.grid_visits[env_index, grid_y_min:grid_y_max+1, grid_x_min:grid_x_max+1] = 0
            self.grid_visited[env_index, grid_y_min:grid_y_max+1, grid_x_min:grid_x_max+1] = 0

    def reset_all(self):
        """
        Reset all the grid and visit counts
        """
        self.grid_visits.zero_()
        #self.grid_visits_normalized.zero_()
        self.grid_visited.zero_()
        #self.visit_count.zero_()
    
    def reset_env(self,env_index):
        """
        Reset grid and count for specific envs
        """
        self.grid_visits[env_index].zero_()
        #self.grid_visits_normalized[env_index].zero_()
        self.grid_visited[env_index].zero_()
        #self.visit_count[env_index] = 0

