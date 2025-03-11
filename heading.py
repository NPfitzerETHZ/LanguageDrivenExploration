import random
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

from occupancy_grid import OccupancyGrid

model = SentenceTransformer('thenlper/gte-base')

# Define word categories
target_terms = [
    "Objective", "Goal", "Mark", "Point", "Focus", "Destination", "Aim", "Spot", 
    "Site", "Position", "Location", "Zone", "Subject", "Waypoint", "Landmark", 
    "Endpoint", "Reference point"
]

start_terms = [
    "is located in", "can be found in", "is positioned in", "is situated in", 
    "lies in", "resides in", "is placed in", "is set in", "is within", 
    "exists in", "is inside", "occupies", "rests in", "stands in"
]

# position_terms = {
#     "corner": ["Edge", "corner",],
#     "center": ["center","core", "middle", "epicenter"],
#     "side": ["side","flank","boundary","border","margin"]
# }

position_terms = [
    "Edge", "corner", "side","flank","border"
]

direction_terms = {
    "left": ["West", "Western", "Left", "Westerly", "Westernmost", "Leftmost", "Leftward", "Left-hand"],
    "top": ["Upper", "Northern", "Top", "Nordic"],
    "right": ["East", "Eastern", "Right", "Easterly", "Easternmost", "Rightmost", "Rightward", "Right-hand"],
    "bottom": ["Bottom", "Lower", "South", "Southern"]
}


direction_map = {
    "left": (0., 0.),
    "top": (0., 1.),
    "right": (1., 0.),
    "bottom": (0., 0.)
}


# Define opposite directions for filtering
opposite_directions = {
    "left": "right",
    "right": "left",
    "top": "bottom",
    "bottom": "top"
}

proximity_terms = {
    "close": ["very close to", "near", "adjacent to", "beside", "a short distance from",
                "moderately close to", "not far from", "at the edge of", "on the outskirts of"],
    "far": ["far from", "distant from", "a considerable distance from", "nowhere near",
            "well beyond", "on the opposite side of", "separated from", "remote from", "away from"]
}

space_terms = [
    "of the area", "of the region", "of the zone", "of the territory", "of the surroundings",
    "of the environment", "of the field", "of the landscape", "of the setting", "of the domain",
    "of the sector", "of the vicinity", "of the grounds",
    "of the premises", "of the scene"
]

search_shapes = [
    "The search area is", "The search zone is", "The search boundary is", "The search space is",
    "The search perimeter is", "The search field is", "The search scope is", "The search territory is",
    "The search extent is", "The investigation area is"
]

size_terms = {
    "large": ["Vast", "Expansive", "Immense", "Enormous", "Extensive", "Broad", "Wide",
                "Colossal", "Gigantic", "Massive", "Sprawling"],
    "small": ["Tiny", "Miniature", "Compact", "Narrow", "Petite", "Minute", "Modest", "Limited",
                "Diminutive", "Micro", "Restricted"],
    "medium": ["Moderate", "Average", "Intermediate", "Mid-sized", "Balanced", "Medium-scale",
                "Midsize", "Fair-sized", "Middle-range", "Standard"]
}
size_map = {
    "large": 1.0,
    "medium": 0.66,
    "small": 0.33
}

MINI_GRID_DIM = 3

class HeadingOccupancyGrid(OccupancyGrid):
    def __init__(self, x_dim, y_dim, num_cells, batch_size, num_targets, mini_grid_dim=3, device='cpu'):
        super().__init__(x_dim, y_dim, num_cells, batch_size, num_targets, mini_grid_dim, device)

        # Handle strings with lists, not tensors
        self.headings_pos_string = [""] * batch_size
        self.heading_size_string = [""] * batch_size
        self.heading_keywords = [["", ""]] * batch_size

        # Use proper tensor initialization for numerical values
        self.heading_pos_grid = torch.zeros((batch_size, 2), dtype=torch.int, device=self.device)
        self.headings_pos_vec = torch.zeros((batch_size, 2), dtype=torch.float, device=self.device)

        # Example embedding size (define EMBEDDING_SIZE in your context)
        EMBEDDING_SIZE = 768
        self.heading_embedding = torch.zeros((batch_size, EMBEDDING_SIZE), device=self.device)

    def update_sentence_from_vector(self, env_index):

        if env_index is None:
            env_index = torch.arange(self.batch_size,dtype=torch.int, device=self.device)
        else:
            env_index = torch.tensor(env_index,device=self.device).view(-1)

        batch_size = env_index.shape[0]

        targets = random.choices(target_terms, k=batch_size)
        insides = random.choices(start_terms, k=batch_size)
        spaces = random.choices(space_terms, k=batch_size)
        searches = random.choices(search_shapes, k=batch_size)
        positions = random.choices(position_terms, k=batch_size)
        size_categories = random.choices(["large", "small", "medium"], k=batch_size)
        sizes = [random.choice(size_terms[cat]) for cat in size_categories]

        # Extracting keywords (lists)
        for i, idx in enumerate(env_index):
            dir_1 = self.heading_keywords[idx][0]
            dir_2 = self.heading_keywords[idx][1]

            # Store results in lists instead of tensors
            self.headings_pos_string[idx] = [
                f"The {targets[i]} {insides[i]} the {dir_1} {dir_2} {positions[i]} {spaces[i]}."
            ]

            print(self.headings_pos_string[idx])
            
            self.heading_size_string[idx] = [
                f"{searches[i]} {sizes[i]}."
            ]

    def update_sentence_vectors(self, env_index):

        if env_index is None:
            env_index = torch.arange(self.batch_size,dtype=torch.int, device=self.device)
        else:
            env_index = torch.tensor(env_index,device=self.device).view(-1)

        batch_size = env_index.shape[0]

        dir_types_1 = random.choices(list(direction_terms.keys()), k=batch_size)
        dir_1 = [random.choice(direction_terms[d]) for d in dir_types_1]

        valid_d_types_2 = [
            random.choice([d for d in direction_terms if d != d1 and d != opposite_directions[d1]])
            for d1 in dir_types_1
        ]
        dir_2 = [random.choice(direction_terms[d]) for d in valid_d_types_2]

        # Convert to NumPy array before tensor
        val_tensor = np.array([direction_map[d1] for d1 in dir_types_1]) + np.array([direction_map[d2] for d2 in valid_d_types_2])
        uncertainty = (np.random.rand(batch_size, 2) * 0.5) * -val_tensor
        val_tensor += uncertainty

        # Convert to PyTorch tensor
        val_tensor = torch.tensor(val_tensor, dtype=torch.float32, device=self.device)

        # Store keywords in a list
        for i, idx in enumerate(env_index.tolist()):
            self.heading_keywords[idx] = [dir_1[i], dir_2[i]]

        # Grid positions (rounded)
        self.heading_pos_grid[:, 0] = (val_tensor[:, 0] * self.grid_width).floor().to(torch.int)
        self.heading_pos_grid[:, 1] = (val_tensor[:, 1] * self.grid_height).floor().to(torch.int)

    def collect_embedding(self, env_index):

        if env_index is None:
            env_index = torch.arange(self.batch_size,dtype=torch.int, device=self.device)
        else:
            env_index = torch.tensor(env_index,device=self.device).view(-1)
        
        self.heading_embedding = model.encode(self.headings_pos_string)
    
    def _initalize_heading(self, env_index):

        if env_index is None:
            env_index = torch.arange(self.batch_size,dtype=torch.int, device=self.device)
        else:
            env_index = torch.tensor(env_index,device=self.device).view(-1,1)
        
        self.update_sentence_vectors(env_index)
        self.update_sentence_from_vector(env_index)

        target_x = self.heading_pos_grid[env_index,0]
        target_y = self.heading_pos_grid[env_index,1]

        print(target_x,target_y)

        x_min = (target_x - MINI_GRID_DIM // 2).int()
        y_min = (target_y - MINI_GRID_DIM // 2).int()
        
        x_range = torch.arange(MINI_GRID_DIM, device=self.device).view(1, -1) + x_min.view(-1, 1)
        y_range = torch.arange(MINI_GRID_DIM, device=self.device).view(1, -1) + y_min.view(-1, 1)

        x_range = torch.clamp(x_range, min=0, max=self.grid_width - 1)
        y_range = torch.clamp(y_range, min=0, max=self.grid_height - 1)

        self.grid_heading[env_index.unsqueeze(1).unsqueeze(2), y_range.unsqueeze(2), x_range.unsqueeze(1)] = 1

        target_pose = self.grid_to_world(target_x, target_y)

        return target_pose

    
