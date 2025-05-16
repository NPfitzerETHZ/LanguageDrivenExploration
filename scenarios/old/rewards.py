import torch

class CountBasedReward:
    def __init__(self, k=5, max_buffer_size=50, state_dim=2):
        self.k = k
        self.max_buffer_size = max_buffer_size
        self.state_dim = state_dim
        
        # Initialize buffer as (batch_size, max_buffer_size, state_dim)
        self.visited_states = None
        self.current_size = 0  # Track how many states are filled

    def update(self, states: torch.Tensor):
        """
        Append new states to the buffer while maintaining a shape of (batch_size, N, 2).
        :param states: Tensor of shape (batch_size, 2)
        """
        batch_size = states.shape[0]
        
        if states.dim() != 2 or states.shape[1] != self.state_dim:
            raise ValueError(f"Expected shape (batch_size, 2), got {states.shape}")
        
        # Initialize buffer if it's the first call
        if self.visited_states is None:
            self.visited_states = torch.zeros(batch_size, self.max_buffer_size, self.state_dim, device=states.device)

        if self.visited_states.shape[0] != batch_size:
            raise ValueError(f"Inconsistent batch size: expected {self.visited_states.shape[0]}, got {batch_size}")
        
        # Roll buffer left (removes oldest state) and insert new states
        if self.current_size < self.max_buffer_size:
            self.visited_states[:, self.current_size] = states
            self.current_size += 1
        else:
            self.visited_states = torch.roll(self.visited_states, shifts=-1, dims=1)
            self.visited_states[:, -1] = states  

    def compute(self, state: torch.Tensor):
        """
        Compute bonus for the given state using k-NN.
        :param state: Tensor of shape (batch_size, 2)
        :return: Tensor of shape (batch_size,) with bonus values.
        """
        if state.dim() != 2 or state.shape[1] != self.state_dim:
            raise ValueError(f"Expected shape (batch_size, 2), got {state.shape}")

        if self.visited_states is None or self.current_size == 0:
            return 0.1*torch.ones(state.shape[0], device=state.device)
        all_states = self.visited_states[:, :self.current_size]  # Shape (batch_size, N, 2)

        # Compute pairwise distances: shape (batch_size, N)
        dists = torch.cdist(state.unsqueeze(1), all_states).squeeze(1)
        # k-NN distance calculation
        k = min(self.k, dists.shape[1])
        sorted_dists, _ = torch.sort(dists, dim=1)
        avg_dists = sorted_dists[:, :k].mean(dim=1)

        bonus = 0.1*torch.maximum(torch.zeros_like(avg_dists), 1 - torch.exp(-(avg_dists - 0.01) / 0.006))
        return bonus

    def reset(self):
        """Reset the visited states buffer to an empty state."""
        self.visited_states = None
        self.current_size = 0

class EntropyBasedReward:
    def __init__(self, radius, bandwidth=None, max_buffer_size=30, state_dim=2, dynamic_bandwidth=True):
        """
        :param bandwidth: If None, use standard deviation-based bandwidth. Otherwise, set a fixed float.
        :param dynamic_bandwidth: If True, adapt bandwidth based on stored states.
        """
        self.radius = radius
        self.bandwidth = bandwidth  # Fixed bandwidth if provided
        self.dynamic_bandwidth = dynamic_bandwidth
        self.max_buffer_size = max_buffer_size
        self.state_dim = state_dim
        
        self.visited_states = None
        self.current_size = None

    def update(self, states: torch.Tensor):
        batch_size = states.shape[0]
        
        if states.dim() != 2 or states.shape[1] != self.state_dim:
            raise ValueError(f"Expected shape (batch_size, {self.state_dim}), got {states.shape}")
        
        # Initialize buffer
        if self.visited_states is None:
            self.visited_states = torch.zeros(batch_size, self.max_buffer_size, self.state_dim, device=states.device)

        if self.current_size is None:
            self.current_size = torch.zeros(batch_size,dtype=torch.int,device=states.device)

        if self.visited_states.shape[0] != batch_size:
            raise ValueError(f"Inconsistent batch size: expected {self.visited_states.shape[0]}, got {batch_size}")
        
        # Add new states
        mask = self.current_size < self.max_buffer_size
        self.visited_states[mask,self.current_size[mask]] = states[mask]
        self.current_size[mask] += 1

        self.visited_states[mask.logical_not()] = torch.roll(self.visited_states[mask.logical_not()], shifts=-1, dims=1)
        self.visited_states[mask.logical_not(),-1] = states[mask.logical_not()]

    def compute(self, state: torch.Tensor):

        if state.dim() != 2 or state.shape[1] != self.state_dim:
            raise ValueError(f"Expected shape (batch_size, {self.state_dim}), got {state.shape}")

        if self.visited_states is None or self.current_size is None :
            return 0.05 * torch.ones(state.shape[0], device=state.device)

        # Compute dynamic bandwidth if enabled
        if self.dynamic_bandwidth and self.visited_states.shape[1] > 1:
            std_dev = torch.std(self.visited_states, dim=1, unbiased=False)  # Shape: (batch_size, state_dim)
            bandwidth = torch.mean(std_dev, dim=-1, keepdim=True)  # Shape: (batch_size, 1)
            bandwidth = torch.clamp(bandwidth, min=1e-3)  # Avoid zero bandwidth
        else:
            bandwidth = self.bandwidth if self.bandwidth else 0.2

        # Compute pairwise squared distances
        diff = state[:, None, :] - self.visited_states  # (batch_size, N, state_dim)
        squared_dist = torch.sum(diff ** 2, dim=-1)  # (batch_size, N)
        adjusted_dist = torch.maximum(squared_dist - self.radius**2, torch.zeros_like(squared_dist)) # Account for Lidar radius

        # Apply Gaussian kernel density
        kernel_values = torch.exp(-adjusted_dist / (2 * bandwidth ** 2))  # (batch_size, N)
        density_estimate = kernel_values.mean(dim=-1)  # (batch_size,)

        # Compute bonus (higher bonus for lower density)
        #bonus = 0.1 / torch.abs((density_estimate) / 0.14)
        #bonus = 0.1*torch.exp(-(density_estimate) / 0.2)
        bonus = 0.25*(0.8 - torch.exp((density_estimate-1)/ 0.6))

        #bonus = 0.1*torch.maximum(torch.zeros_like(density_estimate), 1 - torch.exp(-(density_estimate - 0.26*3/2) / 0.14))
        return bonus
    
    def reset(self):
        """Reset the visited states buffer to an empty state."""
        self.visited_states.zero_()
        self.current_size = 0

    def reset(self,env_index):
        self.visited_states[env_index].zero_()
        self.current_size[env_index] = 0

class JointEntropyBasedReward:

    def __init__(self, radius, n_agents, bandwidth=None, max_buffer_size=30, state_dim=2, dynamic_bandwidth=True):
        """
        :param bandwidth: If None, use standard deviation-based bandwidth. Otherwise, set a fixed float.
        :param dynamic_bandwidth: If True, adapt bandwidth based on stored states.
        """
        self.radius = radius
        self.n_agents = n_agents
        self.bandwidth = bandwidth  # Fixed bandwidth if provided
        self.dynamic_bandwidth = dynamic_bandwidth
        self.max_buffer_size = max_buffer_size
        self.state_dim = state_dim
        
        self.visited_states = None
        self.current_size = None

    def update(self, states: torch.Tensor):

        # states: Shape(batch_size,n_agents*state_dim)
        batch_size = states.shape[0]
        
        if states.dim() != 2 or states.shape[1] != self.state_dim*self.n_agents:
            raise ValueError(f"Expected shape (batch_size, {self.state_dim}), got {states.shape}")
        
        # Initialize buffer
        if self.visited_states is None:
            self.visited_states = torch.zeros(batch_size, self.max_buffer_size, self.state_dim*self.n_agents, device=states.device)
        
        if self.current_size is None:
            self.current_size = torch.zeros(batch_size,device=states.device)

        if self.visited_states.shape[0] != batch_size:
            raise ValueError(f"Inconsistent batch size: expected {self.visited_states.shape[0]}, got {batch_size}")
        
        # Add new states
        mask = self.current_size < self.max_buffer_size
        self.visited_states[mask,self.current_size[mask]] = states[mask]
        self.current_size[mask] += 1

        self.visited_states[not mask] = torch.roll(self.visited_states[not mask], shifts=-1, dims=1)
        self.visited_states[not mask,-1] = states[not mask]

    def compute(self, state: torch.Tensor):

        if state.dim() != 2 or state.shape[1] != self.state_dim*self.n_agents:
            raise ValueError(f"Expected shape (batch_size, {self.state_dim}), got {state.shape}")

        if self.visited_states is None or self.current_size == 0:
            return 0.05 * torch.ones(state.shape[0], device=state.device)

        all_states = self.visited_states[:, :self.current_size]  # Shape (batch_size, N, n_agents*state_dim)

        # Compute dynamic bandwidth if enabled
        if self.dynamic_bandwidth:
            std_dev = torch.std(all_states, dim=1, unbiased=False)  # Shape: (batch_size, n_agents*state_dim)
            bandwidth = torch.mean(std_dev, dim=-1, keepdim=True)  # Shape: (batch_size, 1)
            bandwidth = torch.clamp(bandwidth, min=1e-3)  # Avoid zero bandwidth
        else:
            bandwidth = self.bandwidth if self.bandwidth else 0.2

        # Compute pairwise squared distances
        diff = state[:, None, :] - all_states  # (batch_size, N, n_agents*state_dim)
        squared_dist = torch.sum(diff ** 2, dim=-1)  # (batch_size, N)
        adjusted_dist = torch.maximum(squared_dist - self.radius**2, torch.zeros_like(squared_dist)) # Account for Lidar radius

        # Apply Gaussian kernel density
        kernel_values = torch.exp(-adjusted_dist / (2 * bandwidth ** 2))  # (batch_size, N)
        density_estimate = kernel_values.mean(dim=-1)  # (batch_size,)

        # Compute bonus (higher bonus for lower density)
        #bonus = 0.1 / torch.abs((density_estimate) / 0.14)
        #bonus = 0.1*torch.exp(-(density_estimate) / 0.2)
        bonus = 0.25*(0.8 - torch.exp((density_estimate-1)/ 0.6))
        #bonus = 0.1*torch.maximum(torch.zeros_like(density_estimate), 1 - torch.exp(-(density_estimate - 0.26*3/2) / 0.14))
        return bonus
    
    def reset(self):
        """Reset the visited states buffer to an empty state."""
        self.visited_states.zero_()
        self.current_size = 0

    def reset(self,env_index):
        self.visited_states[env_index].zero_()
        self.current_size[env_index] = 0
    
