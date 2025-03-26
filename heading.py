import random
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from occupancy_grid import OccupancyGrid, TARGET, OBSTACLE, VISITED_TARGET
import torch.nn as nn

model_path = "llm0_decoder_model.pth"

LARGE = 10


class Decoder(nn.Module):
    def __init__(self, emb_size):
        super(Decoder, self).__init__()
        hidden_size = 256
        self.l0 = nn.Linear(emb_size, hidden_size)
        self.l1 = nn.Linear(hidden_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, 2)
        self.relu = nn.ReLU()

    def forward(self, embed):
        x = self.relu(self.l0(embed))
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        return self.l3(x)


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

MINI_GRID_RADIUS = 1
EMBEDDING_SIZE = 1024

class HeadingOccupancyGrid(OccupancyGrid):
    def __init__(self, x_dim, y_dim, num_cells, batch_size, num_targets, heading_mini_grid_radius=1, device='cpu'):
        super().__init__(x_dim, y_dim, num_cells, batch_size, num_targets, heading_mini_grid_radius, device)

        self.llm = SentenceTransformer('thenlper/gte-large', device=device)
        
        self.decoder_mlp = Decoder(emb_size=EMBEDDING_SIZE).to(self.device)
        self.decoder_mlp.load_state_dict(torch.load(model_path, map_location=self.device))
        self.decoder_mlp.eval()

        self.heading_embeddings = torch.zeros((self.batch_size,self.num_targets,EMBEDDING_SIZE),device=self.device)
    
    def spawn_llm_map(self, env_index, n_obstacles, n_agents, n_targets, target_class, llm_activate, padding = True):

        env_index = env_index.view(-1,1)
        packet_size = env_index.shape[0]
        target_poses = torch.zeros((packet_size,n_targets,2),device=self.device)
        unknown_targets = [] # Targets not hinted through a heading

        if padding: pad = 1 
        else: pad = 0
        
        for t in range(self.num_targets):
            # Convert world coordinates to grid indices
            rando = random.random()
            #randos = torch.randint(-self.heading_mini_grid_radius,self.heading_mini_grid_radius + 1,(packet_size,2), device=self.device).int() # Create a bit of chaos
            randos = torch.randint(0,self.heading_mini_grid_radius + 1,(packet_size,2), device=self.device).int()
            if rando < self.heading_to_target_ratio and llm_activate:

                dir_types_1 = random.choices(list(direction_terms.keys()), k=packet_size)
                dir_1 = [random.choice(direction_terms[d]) for d in dir_types_1]

                valid_d_types_2 = [
                    random.choice([d for d in direction_terms if d != d1 and d != opposite_directions[d1]])
                    for d1 in dir_types_1
                ]
                dir_2 = [random.choice(direction_terms[d]) for d in valid_d_types_2]

                targets = random.choices(target_terms, k=packet_size)
                insides = random.choices(start_terms, k=packet_size)
                spaces = random.choices(space_terms, k=packet_size)
                searches = random.choices(search_shapes, k=packet_size)
                positions = random.choices(position_terms, k=packet_size)
                size_categories = random.choices(["large", "small", "medium"], k=packet_size)
                sizes = [random.choice(size_terms[cat]) for cat in size_categories]

                batch_strings = [f"The {targets[i]} {insides[i]} the {dir_1[i]} {dir_2[i]} {positions[i]} {spaces[i]}." for i in range(packet_size)]
                # for i, idx in enumerate(env_index.tolist()):

                #     string_pos = f"The {targets[i]} {insides[i]} the {dir_1[i]} {dir_2[i]} {positions[i]} {spaces[i]}."
                #     #string_size = f"{searches[i]} {sizes[i]}."

                embedding = torch.tensor(self.llm.encode(batch_strings), device = self.device).squeeze(0)
                self.heading_embeddings[env_index.squeeze(1),t] = embedding
                vec = self.decoder_mlp(embedding).view(-1,2)
                vec_x = torch.clamp(vec[:,0] * (self.grid_width - 1) + pad + randos[:,0], min=pad, max=self.grid_width - 1).int()
                vec_y = torch.clamp(vec[:,1] * (self.grid_height - 1) + pad + randos[:,1], min=pad, max=self.grid_height - 1).int()

                heading_center_x = vec_x # + randos[:,0]
                heading_center_y = vec_y # + randos[:,1]

                # self.headings[env_index, t, 0] = torch.clamp(heading_center_x.unsqueeze(1), min=self.heading_mini_grid_radius + pad, max=self.grid_width - 1 - self.heading_mini_grid_radius)
                # self.headings[env_index, t, 1] = torch.clamp(heading_center_y.unsqueeze(1), min=self.heading_mini_grid_radius + pad, max=self.grid_height - 1 - self.heading_mini_grid_radius)

                self.headings[env_index, t, 0] = heading_center_x.unsqueeze(1)
                self.headings[env_index, t, 1] = heading_center_y.unsqueeze(1)

                # x_min = (self.headings[env_index, t, 0].squeeze(0) - self.heading_mini_grid_radius).int()
                # y_min = (self.headings[env_index, t, 1].squeeze(0) - self.heading_mini_grid_radius).int()
                
                # x_range = torch.arange(self.heading_mini_grid_radius*2+1, device=self.device).view(1, -1) + x_min.view(-1, 1)
                # y_range = torch.arange(self.heading_mini_grid_radius*2+1, device=self.device).view(1, -1) + y_min.view(-1, 1)

                # x_range = torch.clamp(x_range, min=0, max=self.grid_width - 1)
                # y_range = torch.clamp(y_range, min=0, max=self.grid_height - 1)

                #self.grid_heading[env_index.unsqueeze(2), y_range.unsqueeze(2), x_range.unsqueeze(1)] = 1
                self.gaussian_heading(env_index,self.headings[env_index,t],t,self.heading_mini_grid_radius,self.heading_mini_grid_radius)
                self.grid_targets[env_index, vec_y.unsqueeze(1), vec_x.unsqueeze(1)] = (TARGET + target_class[env_index]) 

                target_poses[:,t] = self.grid_to_world(vec_x-pad, vec_y-pad)

            else:
                self.headings[env_index, t] = VISITED_TARGET
                unknown_targets.append(t)

        n_unknown_targets = len(unknown_targets)
        n_known_targets = self.num_targets - n_unknown_targets

        grid_size = self.grid_width * self.grid_height

        assert grid_size >= n_obstacles + n_agents + n_targets , "Not enough room for all entities"

        # Generate random values and take the indices of the top `n_obstacles` smallest values
        rand_values = torch.rand(packet_size, grid_size, device=self.device)

        agent_sample_start = torch.randint(0, grid_size - n_agents, (packet_size,), device=self.device)
        agent_sample_range = torch.arange(0, n_agents, device=self.device).view(1, -1) + agent_sample_start.view(-1, 1)  # Shape: (packet_size, n_agents)
        agent_batch_indices = torch.arange(packet_size, device=self.device).view(-1, 1).expand_as(agent_sample_range)
        rand_values[agent_batch_indices, agent_sample_range] = LARGE

        # Filter out heading targets
        if n_known_targets > 0:
            all_possible_values = torch.arange(grid_size - 1, device=self.device).unsqueeze(0).repeat(packet_size, 1)
            agent_positions_expanded = agent_sample_range.unsqueeze(1).repeat(1,grid_size-1,1)
            mask = all_possible_values.unsqueeze(-1) == agent_positions_expanded
            mask = mask.sum(dim=-1).bool()
            probabilities = (~mask).float()
            probabilities /= probabilities.sum(dim=1, keepdim=True)
            target_sample_start = torch.multinomial(
                probabilities, n_known_targets, replacement=False
            )
            row_indices = torch.arange(rand_values.size(0)).unsqueeze(1)
            rand_values[row_indices,target_sample_start] = LARGE - 1

        # Extract obstacle and agent indices
        sort_indices = torch.argsort(rand_values, dim=1)
        obstacle_indices = sort_indices[:, :n_obstacles]  # First n_obstacles indices
        target_indices = sort_indices[:,n_obstacles:n_obstacles+n_unknown_targets]
        agent_indices = sort_indices[:, -n_agents:]

        # Convert flat indices to (x, y) grid coordinates
        obstacle_grid_x = (obstacle_indices % self.grid_width).view(packet_size, n_obstacles, 1)
        obstacle_grid_y = (obstacle_indices // self.grid_width).view(packet_size, n_obstacles, 1)

        target_grid_x = (target_indices % self.grid_width).view(packet_size, n_unknown_targets,1)
        target_grid_y = (target_indices // self.grid_width).view(packet_size, n_unknown_targets,1)

        agent_grid_x = (agent_indices % self.grid_width).view(packet_size, n_agents, 1)
        agent_grid_y = (agent_indices // self.grid_width).view(packet_size, n_agents, 1)

        # Update grid_obstacles and grid_targets for the given environments and adjust for padding
        if padding:
            self.grid_obstacles[env_index.unsqueeze(1), obstacle_grid_y+1, obstacle_grid_x+1] = OBSTACLE  # Mark obstacles
            self.grid_targets[env_index.unsqueeze(1), target_grid_y+1, target_grid_x+1] = (TARGET + target_class[env_index.unsqueeze(1)]) # Mark targets 
        else:
            self.grid_obstacles[env_index.unsqueeze(1), obstacle_grid_y, obstacle_grid_x] = OBSTACLE  # Mark obstacles
            self.grid_targets[env_index.unsqueeze(1), target_grid_y, target_grid_x] = (TARGET + target_class[env_index.unsqueeze(1).unsqueeze(2)]) # Mark targets 

        # Convert to world coordinates
        obstacle_centers = self.grid_to_world(obstacle_grid_x, obstacle_grid_y)  # Ensure shape (packet_size, n_obstacles, 2)
        target_centers = self.grid_to_world(target_grid_x, target_grid_y)  # Ensure shape (packet_size, n_targets, 2)
        for i, j in enumerate(unknown_targets):
            target_poses[:,j] = target_centers[:,i,0]
        agent_centers = self.grid_to_world(agent_grid_x,agent_grid_y)

        return obstacle_centers.squeeze(-2), agent_centers.squeeze(-2), target_poses
    
    def observe_embeddings(self):

        return self.heading_embeddings.flatten(start_dim=1,end_dim=-1)

    
