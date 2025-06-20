import torch

X = 0
Y = 1

class CoreGrid:

    def __init__(self, x_dim, y_dim, x_scale, y_scale, num_cells, batch_size, device='cpu'):

        self.x_dim = x_dim  # World width normalized
        self.y_dim = y_dim  # World height normalized
        self.x_scale = x_scale # World width scale
        self.y_scale = y_scale # World height scale
        
        self.num_cells = num_cells  # Total number of grid cells
        self.device = device
        self.batch_size = batch_size

        self.grid_width = int(num_cells ** 0.5)  # Assuming a square grid
        self.grid_height = self.grid_width  # Square grid assumption
        self.cell_size_x = self.x_dim / self.grid_width
        self.cell_size_y = self.y_dim / self.grid_height
        self.cell_radius = ((self.cell_size_x/2)**2+(self.cell_size_y/2)**2)**0.5

        self.padded_grid_width = self.grid_width + 2 # Added padding to set obstacles around the env.
        self.padded_grid_height = self.grid_height + 2

        self.border_mask = torch.zeros((self.padded_grid_height, self.padded_grid_width), dtype=torch.bool, device=self.device)
        self.border_mask[0, :] = True
        self.border_mask[-1, :] = True
        self.border_mask[:, 0] = True
        self.border_mask[:, -1] = True
    
    def world_to_grid(self, pos, padding):
        """
        Convert continuous world coordinates to discrete grid coordinates.
        Ensures that the world origin (0,0) maps exactly to the center of the occupancy grid.
        """
        if padding: 
            grid_x = torch.round((pos[..., 0] / (self.cell_size_x * self.x_scale)) + (self.grid_width - 1) / 2).int().clamp(0, self.grid_width - 1) + 1
            grid_y = torch.round((pos[..., 1] / (self.cell_size_y * self.y_scale)) + (self.grid_height - 1) / 2).int().clamp(0, self.grid_height - 1) + 1
        else:
            grid_x = torch.round((pos[..., 0] / (self.cell_size_x * self.x_scale)) + (self.grid_width - 1) / 2).int().clamp(0, self.grid_width - 1)
            grid_y = torch.round((pos[..., 1] / (self.cell_size_y * self.y_scale)) + (self.grid_height - 1) / 2).int().clamp(0, self.grid_height - 1)

        return grid_x, grid_y
    
    def grid_to_world(self, grid_x, grid_y):

        """
        Convert discrete grid coordinates to continuous world coordinates.
        Ensures that the center of each grid cell corresponds to a world coordinate.

        Args:
            grid_x (torch.Tensor): Grid x-coordinates.
            grid_y (torch.Tensor): Grid y-coordinates.

        Returns:
            torch.Tensor: World coordinates (x, y).
        """
        world_x = (grid_x - (self.grid_width - 1) / 2) * self.cell_size_x * self.x_scale
        world_y = (grid_y - (self.grid_height - 1) / 2) * self.cell_size_y * self.y_scale

        return torch.stack((world_x, world_y), dim=-1)
        
    
    def sample_mini_grid(self,pos,mini_grid_radius):

        grid_x, grid_y = self.world_to_grid(pos, padding=True)
        x_min = (grid_x - mini_grid_radius).int()
        y_min = (grid_y - mini_grid_radius).int()

        x_range = torch.arange(mini_grid_radius*2+1, device=self.device).view(1, -1) + x_min.view(-1, 1)
        y_range = torch.arange(mini_grid_radius*2+1, device=self.device).view(1, -1) + y_min.view(-1, 1)

        # Clamp to avoid out-of-bounds indexing
        x_range = torch.clamp(x_range, min=0, max=self.grid_width)
        y_range = torch.clamp(y_range, min=0, max=self.grid_height)

        return x_range, y_range