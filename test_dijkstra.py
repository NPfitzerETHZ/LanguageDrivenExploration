import numpy as np
import heapq
from scipy.ndimage import distance_transform_edt


import numpy as np
import heapq
from scipy.ndimage import distance_transform_edt


def compute_local_free_space(grid, radius=5):
    """
    Computes a local openness score for each free cell by counting accessible free cells within a given radius.
    """
    h, w = grid.shape
    free_space = np.zeros_like(grid, dtype=float)
    
    for i in range(h):
        for j in range(w):
            if grid[i, j] == 0:  # Free cell
                count = 0
                for di in range(-radius, radius + 1):
                    for dj in range(-radius, radius + 1):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < h and 0 <= nj < w and grid[ni, nj] == 0:
                            count += 1
                free_space[i, j] = count
    
    max_free_space = np.max(free_space)
    if max_free_space > 0:
        free_space /= max_free_space  # Normalize to [0,1]
    
    return free_space


def compute_escape_potential(grid):
    """
    Computes an openness score by propagating values from open areas using Dijkstra's algorithm.
    """
    h, w = grid.shape
    escape_potential = np.full((h, w), 10)
    pq = []
    
    # Initialize queue with all free cells in large open areas
    for i in range(h):
        for j in range(w):
            if grid[i, j] == 0:
                neighbors = [(i + di, j + dj) for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]]
                open_neighbors = sum(1 for ni, nj in neighbors if 0 <= ni < h and 0 <= nj < w and grid[ni, nj] == 0)
                if open_neighbors >= 3:  # Large open area
                    escape_potential[i, j] = 0
                    heapq.heappush(pq, (0, i, j))
    
    print(escape_potential)
    
    # Propagate openness score outward
    while pq:
        cost, x, y = heapq.heappop(pq)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < h and 0 <= ny < w and grid[nx, ny] == 0:
                new_cost = cost + 1
                if new_cost < escape_potential[nx, ny]:
                    escape_potential[nx, ny] = new_cost
                    heapq.heappush(pq, (new_cost, nx, ny))
    
    max_escape = np.max(escape_potential)
    if max_escape > 0:
        escape_potential = 1 - (escape_potential / max_escape)  # Normalize and invert
    
    return escape_potential


def compute_dead_end_penalty(grid):
    """
    Identifies dead-end paths and propagates a penalty outward.
    """
    h, w = grid.shape
    penalty = np.zeros((h, w), dtype=float)
    pq = []
    
    # Identify dead-end cells
    for i in range(h):
        for j in range(w):
            if grid[i, j] == 0:
                neighbors = [(i + di, j + dj) for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]]
                free_neighbors = sum(1 for ni, nj in neighbors if 0 <= ni < h and 0 <= nj < w and grid[ni, nj] == 0)
                if free_neighbors == 1:  # Dead-end
                    penalty[i, j] = 1
                    heapq.heappush(pq, (1, i, j))  # Start BFS from here
    
    # Spread penalty outward
    while pq:
        cost, x, y = heapq.heappop(pq)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < h and 0 <= ny < w and grid[nx, ny] == 0 and penalty[nx, ny] == 0:
                new_cost = cost + 1
                penalty[nx, ny] = new_cost
                heapq.heappush(pq, (new_cost, nx, ny))

    # Normalize penalty
    max_penalty = np.max(penalty)
    if max_penalty > 0:
        penalty /= max_penalty  # Normalize to [0,1]
    
    return penalty


def compute_openness(grid):
    """
    Computes the final openness score by combining local free space, escape potential, and dead-end penalties.
    """
    free_space = compute_local_free_space(grid)
    escape_potential = compute_escape_potential(grid)
    dead_end_penalty = compute_dead_end_penalty(grid)
    
    # Weighted sum
    openness = (escape_potential) # - (0.3 * dead_end_penalty) #(0.5 * free_space) + 
    #openness = 0.5*free_space + 0.5*escape_potential - 0.0*dead_end_penalty
    openness = np.clip(openness, 0.0, 1)  # Ensure values are between 0 and 1
    return openness


# # Example Usage
# grid = np.array([
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
#     [1, 0, 1, 0, 1, 0, 1, 1, 0, 1],
#     [1, 0, 1, 0, 1, 0, 0, 1, 0, 1],
#     [1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
#     [1, 0, 1, 0, 0, 0, 0, 1, 0, 1],
#     [1, 0, 1, 1, 0, 1, 1, 1, 0, 1],
#     [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# ])  # 1 = obstacle, 0 = free space

# Example Usage
grid = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 1, 1, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 1, 1, 0, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]) 

openness = compute_openness(grid)
import matplotlib.pyplot as plt
plt.imshow(openness, cmap='hot', interpolation='nearest')
plt.colorbar(label='Openness Score')
plt.show()
