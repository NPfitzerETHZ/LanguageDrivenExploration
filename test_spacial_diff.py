import torch
import torch.nn.functional as F
from occupancy_grid import OccupancyGrid, OBSTACLE, VISITED
import matplotlib.pyplot as plt

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
    
    def compute_local_free_space(self, radius=2):
        """
        Computes a local openness score for each free cell by counting accessible free cells within a given radius.
        Obstacles (1s) are ignored in the summation.
        """
        kernel_size = 2 * radius + 1
        grid_free = (self.grid_obstacles == 0).float().unsqueeze(1)  # Convert to float, add channel dim -> (100,1,12,12)

        # Define convolutional kernel
        kernel = torch.ones((1, 1, kernel_size, kernel_size), device=self.device)  # (out_channels, in_channels, H, W)

        # Apply convolution (counts free cells in the neighborhood)
        free_space = torch.nn.functional.conv2d(grid_free, kernel, padding=radius).squeeze(1)  # Remove channel dim -> (100,12,12)

        # Mask obstacles (set their openness score to 0)
        free_space[self.grid_obstacles == OBSTACLE] = 0  

        # Normalize to [0,1] considering only free space cells
        max_free_space = torch.max(free_space)
        if max_free_space > 0:
            free_space /= max_free_space  

        return 1 - free_space # Shape (100,12,12)
    
    
    def compute_escape_potential(self):
        """
        Computes an openness score using PyTorch without explicit loops.
        Uses Dijkstra-like wavefront propagation with tensor-based operations.
        """
        max = 10.
        grid = (self.grid_obstacles == 0).float().unsqueeze(1)  # Shape: (100,1,12,12)

        batch_size, _, h, w = grid.shape
        device = grid.device

        # Step 1: Identify initial free cells in large open areas
        kernel = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
        padded_grid = F.pad(grid, (1, 1, 1, 1), mode='constant', value=1)
        open_neighbors = torch.nn.functional.conv2d(padded_grid, kernel)  # Shape: (100,1,12,12)

        # Initialize escape potential with a high value
        escape_potential = torch.full((batch_size, 1, h, w), max, device=device)

        # Find open regions with at least 3 free neighbors
        open_areas = (grid == 1) & (open_neighbors >= 3)
        escape_potential[open_areas] = 0  # Initialize large open areas

        front = open_areas.clone().float()  # Active cells (1 if in front, 0 otherwise)
        distance = torch.zeros_like(grid)  # Track distances

        while front.any():
            # Create a padded version of front for convolution
            padded_front = F.pad(front, (1, 1, 1, 1), mode='constant', value=0)
            # Convolve to expand the wavefront
            new_front = (F.conv2d(padded_front, kernel) > 0) & (grid == 1) & (escape_potential == max)
            
            # Update distances (only for newly reached cells)
            distance[new_front] = distance[front.bool()].min() + 2
            escape_potential[new_front] = distance[new_front]

            # Move the front forward
            front = new_front.float()

        # Normalize and invert the escape potential
        max_escape = escape_potential.max()
        if max_escape > 0:
            escape_potential = (escape_potential / max_escape) - (self.grid_obstacles == OBSTACLE).float().unsqueeze(1)
        
        return escape_potential.squeeze(1) + self.grid_obstacles
    

    def compute_dead_end_penalty(self):
        """
        Identifies dead-end paths and propagates a penalty outward using PyTorch.
        Avoids explicit loops via convolution-based neighbor counting and wavefront expansion.
        """

        grid = (self.grid_obstacles == 0).float().unsqueeze(1)  # Shape: (100,1,12,12)

        batch_size, _, h, w = grid.shape
        device = grid.device

        # Step 1: Identify initial free cells in large open areas
        kernel = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
        padded_grid = F.pad(grid, (1, 1, 1, 1), mode='constant', value=1)
        open_neighbors = torch.nn.functional.conv2d(padded_grid, kernel)  # Shape: (100,1,12,12)

        penalty = torch.full((batch_size, 1, h, w), 0.0, device=device)
        # Identify dead-end cells (cells with exactly 1 open neighbor)
        dead_ends = (grid == 1) & (open_neighbors <= 1)
        penalty[dead_ends] = 1  # Assign initial penalty to dead-ends

        # Wavefront propagation without explicit loops
        front = dead_ends.clone().float()  # Active cells (1 if in front, 0 otherwise)
        distance = torch.zeros_like(grid)  # Track distances

        while front.any():
            # Create a padded version of front for convolution
            padded_front = F.pad(front, (1, 1, 1, 1), mode='constant', value=0)
            # Convolve to expand the wavefront
            new_front = (F.conv2d(padded_front, kernel) > 0) & (grid == 1) & (penalty == 0)
            
            # Update distances (only for newly reached cells)
            distance[new_front] = distance[front.bool()].min() + 100
            penalty[new_front] = distance[new_front] ** 0.08

            # Move the front forward
            front = new_front.float()

        # Normalize the penalty
        max_penalty = penalty.max()
        if max_penalty > 0:
            penalty /= max_penalty  # Normalize to [0,1]

        return 1 - penalty.squeeze(1)
    

    # def compute_escape_potential(self, num_iterations=5):
    #     """
    #     Computes an openness score by propagating values from open areas using an iterative convolution-based approximation.
    #     The input grid should have shape (100, 1, 12, 12), and the output will have shape (100, 12, 12).
    #     """
    #     grid_free = (self.grid_obstacles == 0).float().unsqueeze(1)  # Shape: (100,1,12,12)
    #     batch_size, _, h, w = grid_free.shape
    #     device = grid_free.device

    #     # Step 1: Identify initial free cells in large open areas
    #     kernel = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
    #     open_neighbors = torch.nn.functional.conv2d(grid_free, kernel, padding=1)  # Shape: (100,1,12,12)

    #     # Initialize escape potential with a high value
    #     escape_potential = torch.full((batch_size, 1, h, w), 10.0, device=device)
    #     cost = torch.full((batch_size, 1, h, w), 0.0, device=device)

    #     mask = (grid_free == 1) & (open_neighbors >= 3).int()
    #     print(mask[0])

    #     # Mark large open areas (at least 3 free neighbors)
    #     escape_potential[(grid_free == 0)] = 10  # Obstacles stay at max value
    #     escape_potential[(grid_free == 1) & (open_neighbors >= 3)] = 0  # Large open areas start at 0
    #     # Step 2: Iteratively propagate the escape potential outward
    #     for _ in range(num_iterations):
    #         cost = torch.nn.functional.conv2d(mask, kernel, padding=1) + cost
    #         # Ensure propagation only occurs where needed
    #         new_potential = torch.minimum(escape_potential, cost)
    #         escape_potential = torch.where(grid_free == 0, torch.tensor(10.0, device=device), new_potential)
    #         print(escape_potential[0])

    #     # Step 3: Normalize and invert the escape potential
    #     print(escape_potential[0])
    #     max_escape = escape_potential.amax(dim=(1, 2, 3), keepdim=True).clamp(min=1e-6)  # Avoid division by zero
    #     escape_potential = (1 - (escape_potential / max_escape)).squeeze(1)  # Shape: (100,12,12)

    #     return escape_potential


# Usage example
x_dim, y_dim, num_cells, batch_size = 10, 10, 100, 100  
occupancy_grid = SpatialDiffusionOccupancyGrid(x_dim, y_dim, num_cells, batch_size)

# Simulate visiting some cells
# occupancy_grid.update_visited(6, 1)
# occupancy_grid.update_visited(6, 2)
# occupancy_grid.update_visited(6, 3)
# occupancy_grid.update_visited(6, 4)
# occupancy_grid.update_visited(6, 5)
# occupancy_grid.update_visited(6, 6)$

# occupancy_grid.update_visited(10, 1)
# occupancy_grid.update_visited(9, 1)
# occupancy_grid.update_visited(8, 1)

# Add obstacles
# occupancy_grid.update_obstacles(5, 5)
# occupancy_grid.update_obstacles(5, 4)
# occupancy_grid.update_obstacles(5, 3)
# occupancy_grid.update_obstacles(5, 2)
# occupancy_grid.update_obstacles(5, 1)
# occupancy_grid.update_obstacles(7, 6)
# occupancy_grid.update_obstacles(8, 6)
# occupancy_grid.update_obstacles(9, 6)
# occupancy_grid.update_obstacles(10,6)

# #occupancy_grid.update_obstacles(7, 2)
# occupancy_grid.update_obstacles(8, 2)
# occupancy_grid.update_obstacles(9, 2)
# occupancy_grid.update_obstacles(10,2)

import numpy as np
num_obstacles = 5
obstacle_coords = np.random.randint(0, x_dim, size=(num_obstacles, 2))

# Update occupancy grid with random obstacles
for x, y in obstacle_coords:
    occupancy_grid.update_obstacles(x, y)



#occupancy_grid.update_obstacles(5,6)
#occupancy_grid.update_obstacles(6,6)

# Run value diffusion
deltas = occupancy_grid.value_iteration()

free_space = occupancy_grid.compute_local_free_space()
free_space_2 = occupancy_grid.compute_local_free_space(radius=1)
escape_potential = occupancy_grid.compute_escape_potential()
dead_end = occupancy_grid.compute_dead_end_penalty()
#potential = free_space * 0.4 + escape_potential * 0.4 + dead_end * 0.2
potential = dead_end * 0.5 + free_space * 0.5
# Convert tensors to NumPy for visualization
free_space_np = free_space[0].cpu().numpy()
free_space_2_np = free_space_2[0].cpu().numpy()
escape_potential_np = escape_potential[0].cpu().numpy()
dead_end_np = dead_end[0].cpu().numpy()
value_grid_np = occupancy_grid.value_grid[0].numpy()
potential_np = potential[0].cpu().numpy()

# Create subplots
fig, axes = plt.subplots(1, 5, figsize=(20, 5))

# Plot Free Space
im1 = axes[0].imshow(free_space_np, cmap='hot')
axes[0].set_title("Free Space")
fig.colorbar(im1, ax=axes[0])

# Plot Escape Potential
im2 = axes[1].imshow(free_space_2_np, cmap='hot')
axes[1].set_title("Free Space 2")
fig.colorbar(im2, ax=axes[1])

# Plot Dead-End Penalty
im3 = axes[2].imshow(dead_end_np, cmap='hot')
axes[2].set_title("Dead-End Penalty")
fig.colorbar(im3, ax=axes[2])

im4 = axes[3].imshow(value_grid_np, cmap='hot')
axes[3].set_title("Spacial diffusion")
fig.colorbar(im3, ax=axes[3])

# Plot Combined Potential
im5 = axes[4].imshow(potential_np, cmap='hot')
axes[4].set_title("Total Potential")
fig.colorbar(im5, ax=axes[4])

# Show the plots
plt.suptitle("Occupancy Grids")
plt.show()
