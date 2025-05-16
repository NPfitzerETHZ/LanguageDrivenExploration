import torch

class VelocityHistory:
    def __init__(self, batch_size: int, history_length: int, vel_dim: int, device='cpu'):
        """
        A class to store and update velocity history efficiently using PyTorch.
        
        Args:
            history_length (int): Number of past velocity states to retain.
            vel_dim (int): Dimensionality of the velocity vector (e.g., 2D or 3D).
            device (str): Device to store the tensor ('cpu' or 'cuda').
        """
        self.history_length = history_length
        self.vel_dim = vel_dim
        self.device = device
        self.batch_size = batch_size

        # Initialize the velocity history tensor with zeros
        self.history = torch.zeros((batch_size, history_length, vel_dim), device=device)

    def update(self, new_velocity: torch.Tensor):
        """
        Update the velocity history by shifting and adding the latest velocity.
        
        Args:
            new_velocity (torch.Tensor): New velocity tensor of shape (vel_dim,).
        """

        # Shift history and insert new velocity at the end
        self.history = torch.roll(self.history, shifts=-1, dims=1)
        self.history[:,-1] = new_velocity  # Store latest velocity

    def get_flattened(self) -> torch.Tensor:
        """
        Returns the velocity history as a flattened tensor for each batch instance.
        Shape: (batch_size, history_length * vel_dim)
        """
        return self.history.reshape(self.history.shape[0], -1)
    
    def reset_all(self):
        """Resets the velocity history to zeros."""
        self.history.zero_()
    
    def reset(self,env_idx):
        """Resets the velocity history of a given index to zero"""
        self.history[env_idx].zero_()


class PositionHistory:
    def __init__(self, batch_size: int, history_length: int, pos_dim: int, device='cpu'):
        """
        A class to store and update position history efficiently using PyTorch.
        
        Args:
            history_length (int): Number of past position states to retain.
            vel_dim (int): Dimensionality of the position vector (e.g., 2D or 3D).
            device (str): Device to store the tensor ('cpu' or 'cuda').
        """
        self.history_length = history_length
        self.pos_dim = pos_dim
        self.device = device
        self.batch_size = batch_size

        # Initialize the position history tensor with zeros
        self.history = torch.zeros((batch_size, history_length, pos_dim), device=device)

    def update(self, new_position: torch.Tensor):
        """
        Update the position history by shifting and adding the latest position.
        
        Args:
            new_position (torch.Tensor): New position tensor of shape (vel_dim,).
        """

        # Shift history and insert new position at the end
        self.history = torch.roll(self.history, shifts=-1, dims=1)
        self.history[:,-1] = new_position  # Store latest position

    def get_flattened(self) -> torch.Tensor:
        """
        Returns the position history as a flattened tensor for each batch instance.
        Shape: (batch_size, history_length * vel_dim)
        """
        return self.history.reshape(self.history.shape[0], -1)
    
    def reset_all(self):
        """Resets the position history to zeros."""
        self.history.zero_()

    def reset(self,env_idx):
        """Resets the position history of a given index to zero"""
        self.history[env_idx].zero_()

    
class JointPosHistory:

    def __init__(self, n_agents, batch_size: int, history_length: int, pos_dim: int, device='cpu'):
        """
        A class to store and update position history efficiently using PyTorch.
        
        Args:
            history_length (int): Number of past position states to retain.
            vel_dim (int): Dimensionality of the position vector (e.g., 2D or 3D).
            device (str): Device to store the tensor ('cpu' or 'cuda').
        """
        self.n_agents = n_agents
        self.history_length = history_length
        self.pos_dim = pos_dim
        self.device = device
        self.batch_size = batch_size

        # Initialize the position history tensor with zeros
        self.history = torch.zeros((batch_size, history_length, self.n_agents*self.pos_dim), device=device)

    def update(self, new_position: torch.Tensor):
        """
        Update the position history by shifting and adding the latest position.
        
        Args:
            new_position (torch.Tensor): New position tensor of shape (pos_dim*n_agents,).
        """

        # Shift history and insert new position at the end
        self.history = torch.roll(self.history, shifts=-1, dims=1)
        self.history[:,-1] = new_position  # Store latest position

    def get_flattened(self) -> torch.Tensor:
        """
        Returns the position history as a flattened tensor for each batch instance.
        Shape: (batch_size, history_length * pos_dim * n_agents)
        """
        return self.history.reshape(self.history.shape[0], -1)
    
    def reset(self):
        """Resets the joint position history to zeros."""
        self.history.zero_()

    def reset(self,env_idx):
        """Resets the joint position history of a given index to zero"""
        self.history[env_idx].zero_()

# class CellPosHistory:

#     def __init__(self, batch_size: int, history_length: int, pos_dim: int, device='cpu'):
#         # Idea:
#         # Have a history of previously visited cells (relative to current)
#         # Have a history of cell quality (number of neighbouring free spaces vs. full)
#         # Maybe have a grid of cell quality? Updated at each step? is it too much compute? It would be kind of a value iteration.
#         # Triggering reward?