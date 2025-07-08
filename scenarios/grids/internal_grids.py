import torch
from scenarios.grids.core_grid import CoreGrid
from scenarios.grids.environment_grids import TARGET, OBSTACLE


VISIT = 1
VISITED_TARGET= -1
UNKNOWN_TARGET = 0.5
EMPTY = 0

class InternalOccupancyGrid(CoreGrid):

    def __init__(self, x_dim, y_dim, x_scale, y_scale, num_cells, visit_threshold, batch_size, device='cpu'):

        super().__init__(x_dim, y_dim, x_scale, y_scale, num_cells, batch_size, device)
        
        self.visit_threshold = visit_threshold

        ### INTERNAL MAPS - Memory ###
        self.grid_observed_targets = torch.ones((batch_size,self.padded_grid_height, self.padded_grid_width), dtype=torch.float, device=self.device) * UNKNOWN_TARGET
        self.grid_found_targets = torch.zeros((batch_size,self.padded_grid_height, self.padded_grid_width), device=self.device)
        self.grid_visits = torch.zeros((batch_size,self.padded_grid_height, self.padded_grid_width), device=self.device)
        self.grid_visits_sigmoid = torch.zeros((batch_size,self.padded_grid_height, self.padded_grid_width), device=self.device)
    
    def update(
        self,
        agent_positions: torch.Tensor,   # (B,2)
        mini_grid_radius: int,
        grid_targets: torch.Tensor,      # (ALL, H, W)
        env_index: torch.Tensor = None,  # (B,)
        despawn_targets: bool = True,  # if True, despawn targets that are visited
        ):
        
        if env_index is None:
            B, dev = agent_positions.size(0), agent_positions.device
            env_index    = torch.arange(B, device=dev)
        
        gx, gy = self.world_to_grid(agent_positions, padding=True)      # (B,)

        # 1) mark targets the agents are standing on
        visited = grid_targets[env_index, gy, gx] == TARGET
        self.grid_found_targets[env_index[visited], gy[visited], gx[visited]] = VISITED_TARGET
        if despawn_targets:
            grid_targets[env_index[visited], gy[visited], gx[visited]] = EMPTY                      # despawn

        # 2) copy current field of view into observation map
        b = env_index[:, None, None]                                          # (B,1,1)
        x_rng, y_rng = self.sample_mini_grid(agent_positions, mini_grid_radius)
        mini = grid_targets[b, y_rng[..., None], x_rng[:, None, :]]
        self.grid_observed_targets[b, y_rng[..., None], x_rng[:, None, :]] = mini

        # 3) make sure visited targets also show up in the observation map
        mask = self.grid_found_targets == VISITED_TARGET             # (B,H,W) bool
        self.grid_observed_targets[mask] = VISITED_TARGET

        # 4) update visit counts
        self.grid_visits[env_index, gy, gx] += 1
        v = self.grid_visits[env_index, gy, gx] - self.visit_threshold
        self.grid_visits_sigmoid[env_index, gy, gx] = VISIT * torch.sigmoid(v)
    
    def update_visits(self, agent_positions):
        
        B, dev = agent_positions.size(0), agent_positions.device
        gx, gy = self.world_to_grid(agent_positions, padding=True)      # (B,)
        idx    = torch.arange(B, device=dev)
        
        self.grid_visits[idx, gy, gx] += 1
        v = self.grid_visits[idx, gy, gx] - self.visit_threshold
        self.grid_visits_sigmoid[idx, gy, gx] = VISIT * torch.sigmoid(v)
    
    def get_grid_observation_2d(self, pos, mini_grid_radius):
        
        x_range , y_range = self.sample_mini_grid(pos, mini_grid_radius)
        
        mini_grid_observed_targets = self.grid_observed_targets[torch.arange(pos.shape[0]).unsqueeze(1).unsqueeze(2), y_range.unsqueeze(2), x_range.unsqueeze(1)]
        mini_grid_observed_targets /= TARGET
        
        mini_grid_visited = self.grid_visits_sigmoid[torch.arange(pos.shape[0]).unsqueeze(1).unsqueeze(2), y_range.unsqueeze(2), x_range.unsqueeze(1)]
        mini_grid_visited /= VISIT
        
        mini_grid_stack = torch.stack([mini_grid_visited, mini_grid_observed_targets], dim=1)
        
        return  mini_grid_stack
    
    def get_grid_visits_obstacle_observation(self, pos, mini_grid_radius, mini_grid_obstacles):

        x_range , y_range = self.sample_mini_grid(pos, mini_grid_radius)

        mini_grid_visited = self.grid_visits_sigmoid[torch.arange(pos.shape[0]).unsqueeze(1).unsqueeze(2), y_range.unsqueeze(2), x_range.unsqueeze(1)]
        mini_grid_visited /= VISIT
        
        mini_grid_obstacles /= OBSTACLE
        mask = torch.where(mini_grid_obstacles == EMPTY)
        mini_grid_obstacles[mask] = mini_grid_visited[mask]

        return mini_grid_obstacles.flatten(start_dim=1, end_dim=-1)

    def compute_exploration_bonus(self, agent_positions, exploration_rew_coeff = -0.02, new_cell_rew_coeff = 0.25):
        """
        Compute exploration reward: Reward for visiting new cells.
        """
        grid_x, grid_y = self.world_to_grid(agent_positions, padding=True)
        visit_lvl = self.grid_visits_sigmoid[torch.arange(agent_positions.shape[0]), grid_y, grid_x]
        new_cell_bonus = (visit_lvl < 0.2).float() * new_cell_rew_coeff

        # Works good, negative reward with short postive.
        reward = exploration_rew_coeff * visit_lvl + new_cell_bonus #  Sigmoid Penalty for staying in a visited cell + bonus for discovering a new cell
        return reward
    
    def reset_all(self):
        """
        Reset all the grid and visit counts
        """
        self.grid_visits.zero_()
        self.grid_visits_sigmoid.zero_()
        self.grid_found_targets.zero_()
        self.grid_observed_targets.fill_(UNKNOWN_TARGET)

    def reset_env(self,env_index):
        """
        Reset grid and count for specific envs
        """
        self.grid_visits[env_index].zero_()
        self.grid_visits_sigmoid[env_index].zero_()
        self.grid_found_targets[env_index].zero_()
        self.grid_observed_targets[env_index].fill_(UNKNOWN_TARGET)