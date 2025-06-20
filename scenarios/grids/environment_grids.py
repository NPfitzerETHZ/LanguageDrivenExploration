import torch
from scenarios.grids.core_grid import CoreGrid

TARGET=1
OBSTACLE=2

class EnvironmentGrid(CoreGrid):

    def __init__(self, x_dim, y_dim, x_scale, y_scale, num_cells, batch_size, device='cpu'):

        super().__init__(x_dim, y_dim, x_scale, y_scale, num_cells, batch_size, device)
        
        self.grid_obstacles = torch.zeros((batch_size,self.padded_grid_height, self.padded_grid_width), device=self.device)
        self.grid_obstacles[:, self.border_mask] = OBSTACLE
        self.grid_targets = torch.zeros((batch_size,self.padded_grid_height, self.padded_grid_width), dtype=torch.float, device=self.device)
    
    def get_grid_obstacle_observation(self, pos, mini_grid_radius):

        x_range , y_range = self.sample_mini_grid(pos, mini_grid_radius)

        mini_grid = self.grid_obstacles[torch.arange(pos.shape[0]).unsqueeze(1).unsqueeze(2), y_range.unsqueeze(2), x_range.unsqueeze(1)]

        return mini_grid.flatten(start_dim=1, end_dim=-1) / OBSTACLE
    
    def get_grid_obstacle_observation_2d(self, pos, mini_grid_radius):

        x_range , y_range = self.sample_mini_grid(pos, mini_grid_radius)

        mini_grid = self.grid_obstacles[torch.arange(pos.shape[0]).unsqueeze(1).unsqueeze(2), y_range.unsqueeze(2), x_range.unsqueeze(1)]

        return mini_grid / OBSTACLE
    
    def get_grid_target_observation(self, pos, mini_grid_radius):

        x_range , y_range = self.sample_mini_grid(pos, mini_grid_radius)
        mini_grid = self.grid_targets[torch.arange(pos.shape[0]).unsqueeze(1).unsqueeze(2), y_range.unsqueeze(2), x_range.unsqueeze(1)]

        return mini_grid.flatten(start_dim=1, end_dim=-1) / TARGET
    
    def get_grid_target_observation_2d(self, pos, mini_grid_radius):

        x_range , y_range = self.sample_mini_grid(pos, mini_grid_radius)
        mini_grid = self.grid_targets[torch.arange(pos.shape[0]).unsqueeze(1).unsqueeze(2), y_range.unsqueeze(2), x_range.unsqueeze(1)]

        return mini_grid / TARGET
    
    def reset_all(self):
        """
        Reset all the grid and visit counts
        """
        self.grid_targets.zero_()
        self.grid_obstacles.zero_()
        self.grid_obstacles[:,self.border_mask] = OBSTACLE
        
    def reset_env(self,env_index):
        """
        Reset grid and count for specific envs
        """
        self.grid_targets[env_index].zero_()
        self.grid_obstacles[env_index].zero_()
        self.grid_obstacles[env_index,self.border_mask] = OBSTACLE
        
    