import torch
import torch.nn.functional as F
from occupancy_grid import OccupancyGrid
from occupancy_grid import VISITED

class DeadEndOccupancyGrid(OccupancyGrid):
    def __init__(self, x_dim, y_dim, num_cells, batch_size, num_targets, mini_grid_dim=3, device='cpu'):
        super().__init__(x_dim, y_dim, num_cells, batch_size, num_targets, mini_grid_dim, device)

        self.dead_end_grid = torch.zeros((batch_size,self.padded_grid_height, self.padded_grid_width), device=self.device)
        self.value_grid = torch.zeros((batch_size,self.padded_grid_height, self.padded_grid_width), device=self.device)

    def compute_dead_end_grid(self, env_index, radius = 1):
        """
        Identifies dead-end paths and propagates a dead_end_val outward using PyTorch.
        Avoids explicit loops via convolution-based neighbor counting and wavefront expansion.
        """
        if env_index is None:
            env_index = torch.arange(self.batch_size, dtype=torch.int, device=self.device).view(-1,1)
        else:
            if isinstance(env_index, torch.Tensor): 
                env_index = env_index.view(-1, 1)
            else:
                env_index = torch.tensor(env_index, device=self.device).view(-1, 1)  # Shape: (batch_size,1,12,12)

        keep_mask = self.grid_obstacles[env_index] == 0
        grid = (keep_mask).float()  # Shape: (batch_size,1,12,12)
        padded_grid = F.pad(grid, (1, 1, 1, 1), mode='constant', value=1)

        batch_size, _, h, w = grid.shape
        device = grid.device

        kernel_size = 2 * radius + 1

        # Define convolutional kernel
        kernel = torch.ones((1, 1, kernel_size, kernel_size), device=self.device)  # (out_channels, in_channels, H, W)

        # Apply convolution (counts free cells in the neighborhood)
        free_space = torch.nn.functional.conv2d(padded_grid, kernel) # (batch_size,1,12,12)

        # Mask obstacles (set their openness score to 0)
        free_space[grid == 0] = 0  

        # Normalize to [0,1] considering only free space cells
        max_free_space = torch.max(free_space)
        if max_free_space > 0:
            free_space /= max_free_space  
        free_space[~keep_mask] = 0.
        #======================================

        # Step 1: Identify initial free cells in large open areas
        kernel = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=torch.float32, device=device).view(1, 1, 3, 3)

        open_neighbors = torch.nn.functional.conv2d(padded_grid, kernel)  # Shape: (batch_size,1,12,12)

        dead_end_val = torch.full((batch_size, 1, h, w), 0.0, device=device)
        # Identify dead-end cells (cells with exactly 1 open neighbor)
        dead_ends = (grid == 1) & (open_neighbors <= 1)
        
        if dead_ends.any():
            dead_end_val[dead_ends] = 1  # Assign initial dead_end_val to dead-ends

            # Wavefront propagation without explicit loops
            front = dead_ends.clone().float()  # Active cells (1 if in front, 0 otherwise)
            distance = torch.zeros_like(grid)  # Track distances

            while front.any():
                # Create a padded version of front for convolution
                padded_front = F.pad(front, (1, 1, 1, 1), mode='constant', value=0)
                # Convolve to expand the wavefront
                new_front = (F.conv2d(padded_front, kernel) > 0) & (grid == 1) & (dead_end_val == 0)
                
                # Update distances (only for newly reached cells)
                distance[new_front] = distance[front.bool()].min() + 100
                dead_end_val[new_front] = distance[new_front] ** 0.08

                # Move the front forward
                front = new_front.float()

            # Normalize the dead_end_val
            max_dead_end_val = dead_end_val.max()
            if max_dead_end_val > 0:
                dead_end_val /= max_dead_end_val  # Normalize to [0,1]
            dead_end_val[~keep_mask] = 0.
            dead_end_val = 1 - dead_end_val

        self.dead_end_grid[env_index] = dead_end_val * 0.5 + (1 - free_space) * 0.5 
        self.value_grid[env_index] = self.dead_end_grid[env_index]

    def update(self, agent_positions):
        """
        Update the grid and visit count based on the agents' current positions.
        """
        grid_x, grid_y = self.world_to_grid(agent_positions, padding=True)
        batch_indices = torch.arange(agent_positions.shape[0], device=agent_positions.device)  # Shape: (batch,)
        self.grid_visits[batch_indices, grid_y, grid_x] += 1
        self.grid_visits_sigmoid[batch_indices, grid_y, grid_x] = 1/(1+torch.exp(self.visit_threshold - self.grid_visits[batch_indices, grid_y, grid_x]))
        self.grid_visited[batch_indices, grid_y, grid_x] = VISITED
        self.value_grid[batch_indices, grid_y, grid_x] = self.dead_end_grid[batch_indices, grid_y, grid_x] + self.grid_visits_sigmoid[batch_indices, grid_y, grid_x]
        
    def get_deadend_grid_observation(self, pos, mini_grid_dim):

        x_range , y_range = self.sample_mini_grid(pos, mini_grid_dim)

        mini_grid_dead_end = self.dead_end_grid[torch.arange(pos.shape[0]).unsqueeze(1).unsqueeze(2), y_range.unsqueeze(2), x_range.unsqueeze(1)]

        return mini_grid_dead_end.flatten(start_dim=1, end_dim=-1)
    
    def get_value_grid_observation(self, pos, mini_grid_dim):

        x_range , y_range = self.sample_mini_grid(pos, mini_grid_dim)

        mini_grid = self.grid_obstacles[torch.arange(pos.shape[0]).unsqueeze(1).unsqueeze(2), y_range.unsqueeze(2), x_range.unsqueeze(1)]
        mini_grid_value = self.value_grid[torch.arange(pos.shape[0]).unsqueeze(1).unsqueeze(2), y_range.unsqueeze(2), x_range.unsqueeze(1)]

        mask = torch.where(mini_grid == 0)

        mini_grid[mask] = mini_grid_value[mask]

        return mini_grid.flatten(start_dim=1, end_dim=-1)
    
    def compute_exploration_bonus(self, agent_positions):
        """
        Compute exploration reward: Reward for visiting new cells.
        """
        grid_x, grid_y = self.world_to_grid(agent_positions, padding=True)
        #visits = self.grid_visits_normalized[torch.arange(agent_positions.shape[0]), grid_y, grid_x]
        visit_lvl = self.value_grid[torch.arange(agent_positions.shape[0]), grid_y, grid_x]
        new_cell_bonus = (visit_lvl < 0.2).float() * 0.5
        #visits = self.grid_visited[torch.arange(agent_positions.shape[0]), grid_y, grid_x]

        # Works good, negative reward with short postive.
        #reward = -0.05 * visit_lvl + new_cell_bonus #  Sigmoid Penalty for staying in a visited cell + bonus for discovering a new cell

        # Here I try postive reward only
        #reward = 0.05*(1/(1+torch.exp(visits - 3)))
        return -0.05 * visit_lvl
    
    def reset_all(self):
        self.dead_end_grid.zero_()
        self.value_grid.zero_()
        return super().reset_all()
    
    def reset_env(self, env_index):
        self.dead_end_grid[env_index].zero_()
        self.value_grid[env_index].zero_()
        return super().reset_env(env_index)
