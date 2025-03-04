import torch
from occupancy_grid import OccupancyGrid, OBSTACLE, VISITED

class SpatialDiffusionOccupancyGrid(OccupancyGrid):  # Updated class name
    def __init__(self, x_dim, y_dim, num_cells, batch_size, num_headings=0, mini_grid_dim=3, device='cpu'):
        super().__init__(x_dim, y_dim, num_cells, batch_size, num_headings, mini_grid_dim, device)

        # Initialize value function grid
        self.value_grid = torch.zeros((batch_size,self.padded_grid_height, self.padded_grid_width), device=self.device)
        self.value_grid[self.grid_obstacles == OBSTACLE] = OBSTACLE

        # Visit penalty factor
        self.visit_penalty = 0.  # Negative value to discourage re-visits
        self.occupied_neighbours_penalty = - 0.1

    def update_visited(self, x, y):
        """
        Marks a cell as visited and applies a visit penalty.
        :param x: X coordinate in grid
        :param y: Y coordinate in grid
        """
        self.grid_visits[:, y, x] += 1
        self.grid_visits_sigmoid[:, y, x] = 1/(1+torch.exp(self.visit_threshold - self.grid_visits[:, y, x]))
    
    def update_obstacles(self, x, y):
        """Marks a cell as an obstacle with a high initial value."""
        self.value_grid[:, y, x] = OBSTACLE
        self.grid_obstacles[:, y, x] = OBSTACLE

    def value_iteration(self, gamma=0.5, threshold=1e-3, max_iters=100):
        """
        Performs diffusion-based value propagation with visited cell penalty, batch-compatible.
        :param gamma: Discount factor for future propagation.
        :param threshold: Convergence threshold.
        :param max_iters: Maximum number of iterations.
        """
        deltas = []
        for _ in range(max_iters):
            old_value_grid = self.value_grid.clone()

            # Compute all neighbors using batch operations
            up    = self.value_grid[:, :-2, 1:-1]  
            down  = self.value_grid[:, 2:, 1:-1]  
            left  = self.value_grid[:, 1:-1, :-2]  
            right = self.value_grid[:, 1:-1, 2:]  
            up_left    = self.value_grid[:, :-2, :-2]
            up_right   = self.value_grid[:, :-2, 2:]
            down_left  = self.value_grid[:, 2:, :-2]
            down_right = self.value_grid[:, 2:, 2:]

            # Stack and compute mean neighbors
            all_neighbors = torch.stack([up, down, left, right, up_left, up_right, down_left, down_right], dim=0)
            mean_neighbors = all_neighbors.mean(dim=0)

            update_grid = gamma * (mean_neighbors + 0.5 * OBSTACLE * self.grid_visits_sigmoid[:, 1:-1, 1:-1])

            # Ensure obstacles are still considered
            mask = (self.grid_obstacles[:, 1:-1, 1:-1] != OBSTACLE)
            self.value_grid[:, 1:-1, 1:-1] = torch.where(mask, update_grid, self.value_grid[:, 1:-1, 1:-1])

            # Compute difference for convergence check
            delta = torch.abs(self.value_grid - old_value_grid).max()
            deltas.append(delta.item())

            if delta < threshold:
                print(f"Converged in {_+1} iterations.")
                break

        return deltas  # Return convergence data for debugging

# Usage example
x_dim, y_dim, num_cells, batch_size = 10, 10, 100, 100  
occupancy_grid = SpatialDiffusionOccupancyGrid(x_dim, y_dim, num_cells, batch_size)

# Simulate visiting some cells
occupancy_grid.update_visited(6, 1)
occupancy_grid.update_visited(6, 2)
occupancy_grid.update_visited(6, 3)
occupancy_grid.update_visited(6, 4)
occupancy_grid.update_visited(6, 5)
occupancy_grid.update_visited(6, 6)

# Add obstacles
occupancy_grid.update_obstacles(5, 5)
occupancy_grid.update_obstacles(5, 4)
occupancy_grid.update_obstacles(5, 3)
occupancy_grid.update_obstacles(5, 2)
occupancy_grid.update_obstacles(5, 1)
occupancy_grid.update_obstacles(7, 5)
occupancy_grid.update_obstacles(7, 4)
occupancy_grid.update_obstacles(7, 3)
occupancy_grid.update_obstacles(7, 2)
occupancy_grid.update_obstacles(7, 1)

# Run value diffusion
deltas = occupancy_grid.value_iteration()

# Visualize value grid for the first batch
import matplotlib.pyplot as plt
plt.imshow(occupancy_grid.value_grid[0].cpu().numpy(), cmap='hot')
plt.colorbar(label="Value")
plt.title("Spatial Diffusion Occupancy Grid (Batch 1)")
plt.show()
