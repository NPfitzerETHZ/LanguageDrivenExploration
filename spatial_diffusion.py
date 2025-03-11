import torch
from occupancy_grid import OccupancyGrid, OBSTACLE, VISITED, TARGET

class SpatialDiffusionOccupancyGrid(OccupancyGrid):  # Updated class name
    def __init__(self, x_dim, y_dim, num_cells, batch_size, num_headings=0, mini_grid_dim=3, device='cpu'):
        super().__init__(x_dim, y_dim, num_cells, batch_size, num_headings, mini_grid_dim, device)

        self.diffusion_update_count = torch.zeros(batch_size, dtype=torch.int, device=self.device)
        self.diffusion_update_thresh = 1

        # Initialize value function grid
        self.value_grid = torch.zeros((batch_size,self.padded_grid_height, self.padded_grid_width), device=self.device)
        self.value_grid[self.grid_obstacles == OBSTACLE] = OBSTACLE

    def diffusion_update(self, env_ids, gamma=0.5, threshold=1e-3, max_iters=100):
        """
        Performs diffusion-based value propagation with visited cell penalty, batch-compatible.
        :param gamma: Discount factor for future propagation.
        :param threshold: Convergence threshold.
        :param max_iters: Maximum number of iterations.
        """
        if env_ids is None:
            env_index = torch.arange(self.batch_size, dtype=torch.int, device=self.device)
        else:
            if isinstance(env_ids, torch.Tensor): 
                env_index = env_ids.view(-1, 1)
            else:
                env_index = torch.tensor(env_ids, device=self.device).view(-1, 1)

        deltas = []
        for _ in range(max_iters):
            old_value_grid = self.value_grid.clone()

            # Compute all neighbors using batch operations
            up    = self.value_grid[env_index, :-2, 1:-1]  
            down  = self.value_grid[env_index, 2:, 1:-1]  
            left  = self.value_grid[env_index, 1:-1, :-2]  
            right = self.value_grid[env_index, 1:-1, 2:]  
            up_left    = self.value_grid[env_index, :-2, :-2]
            up_right   = self.value_grid[env_index, :-2, 2:]
            down_left  = self.value_grid[env_index, 2:, :-2]
            down_right = self.value_grid[env_index, 2:, 2:]

            # Stack and compute mean neighbors
            all_neighbors = torch.stack([up, down, left, right, up_left, up_right, down_left, down_right], dim=0)
            mean_neighbors = all_neighbors.mean(dim=0)

            update_grid = gamma * (mean_neighbors + 0.5 * OBSTACLE * self.grid_visits_sigmoid[env_index, 1:-1, 1:-1])

            # Ensure obstacles are still considered
            mask = (self.grid_obstacles[env_index, 1:-1, 1:-1] != OBSTACLE)
            self.value_grid[env_index, 1:-1, 1:-1] = torch.where(mask, update_grid, self.value_grid[env_index, 1:-1, 1:-1])

            # Compute difference for convergence check
            delta = torch.abs(self.value_grid - old_value_grid).max()
            deltas.append(delta.item())

            if delta < threshold:
                #print(f"Converged in {_+1} iterations.")
                break

        return deltas  # Return convergence data for debugging
    
    # def update(self, agent_positions: torch.Tensor, mini_grid_dim: int = 3):
    #     """
    #     Update the grid and visit count based on the agents' current positions.

    #     Args:
    #         agent_positions (torch.Tensor): The current positions of agents.
    #         mini_grid_dim (int, optional): The dimension of the mini-grid. Defaults to 3.
    #     """

    #     # Convert agent positions to grid coordinates
    #     grid_x, grid_y = self.world_to_grid(agent_positions, padding=True)
        
    #     # Batch indices for multi-agent updates
    #     batch_indices = torch.arange(agent_positions.shape[0], device=agent_positions.device)  

    #     # Get mini-grid ranges for updating
    #     x_range, y_range = self.sample_mini_grid(agent_positions, mini_grid_dim)

    #     # Prepare index tensors
    #     batch_idx_expanded = batch_indices.unsqueeze(-1).unsqueeze(-1)
    #     y_expanded = y_range.unsqueeze(-1)
    #     x_expanded = x_range.unsqueeze(-2)

    #     # Update visit count
    #     self.grid_visits[batch_idx_expanded, y_expanded, x_expanded] += self.visit_mask.unsqueeze(0)

    #     # Apply sigmoid transformation
    #     visits_sub = self.grid_visits[batch_idx_expanded, y_expanded, x_expanded]
    #     self.grid_visits_sigmoid[batch_idx_expanded, y_expanded, x_expanded] = 1 / (1 + torch.exp(self.visit_threshold - visits_sub))

    #     # Mark visited positions
        self.grid_visited[batch_indices, grid_y, grid_x] = VISITED

    def get_value_grid_observation(self, pos, mini_grid_dim):

        x_range , y_range = self.sample_mini_grid(pos, mini_grid_dim)

        mini_grid = self.value_grid[torch.arange(pos.shape[0]).unsqueeze(1).unsqueeze(2), y_range.unsqueeze(2), x_range.unsqueeze(1)]

        return mini_grid.flatten(start_dim=1, end_dim=-1)
    
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
            self.grid_obstacles[env_index, obstacle_grid_y+1, obstacle_grid_x+1] = OBSTACLE  # Mark obstacles
            self.grid_targets[env_index, target_grid_y+1, target_grid_x+1] = (TARGET + target_class[env_index]) # Mark targets 

            self.grid_map[env_index, obstacle_grid_y+1, obstacle_grid_x+1] = OBSTACLE
            self.grid_map[env_index, target_grid_y+1, target_grid_x+1] = TARGET + target_class[env_index]

            self.value_grid[env_index, obstacle_grid_y+1, obstacle_grid_x+1] = OBSTACLE
        else:
            self.grid_obstacles[env_index, obstacle_grid_y, obstacle_grid_x] = OBSTACLE  # Mark obstacles
            self.grid_targets[env_index, target_grid_y, target_grid_x] = (TARGET + target_class[env_index]) # Mark targets 

            self.grid_map[env_index, obstacle_grid_y, obstacle_grid_x] = OBSTACLE
            self.grid_map[env_index, target_grid_y, target_grid_x] = TARGET + target_class[env_index]

            self.value_grid[env_index, obstacle_grid_y, obstacle_grid_x] = OBSTACLE

        # Convert to world coordinates
        obstacle_centers = self.grid_to_world(obstacle_grid_x, obstacle_grid_y)  # Ensure shape (batch_size, n_obstacles, 2)
        target_centers = self.grid_to_world(target_grid_x, target_grid_y)  # Ensure shape (batch_size, n_obstacles, 2)
        agent_centers = self.grid_to_world(agent_grid_x,agent_grid_y)

        return obstacle_centers.squeeze(-2), agent_centers.squeeze(-2), target_centers.squeeze(-2)
    
    def reset_all(self):
        self.value_grid.zero_()
        self.value_grid[:,self.border_mask] = OBSTACLE
        return super().reset_all()
    
    def reset_env(self, env_index):
        self.value_grid[env_index].zero_()
        self.value_grid[env_index,self.border_mask] = OBSTACLE
        return super().reset_env(env_index)
    
