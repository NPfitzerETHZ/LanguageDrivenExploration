import random
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from scenarios.scripts.occupancy_grid import OccupancyGrid, TARGET, OBSTACLE, VISITED_TARGET, VISITED
import torch.nn as nn

from pathlib import Path

model_path = Path(__file__).parent / "llm0_decoder_model.pth"

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

ordinal_prefixes = ["The first", "The second", "The third", "The fourth", "The fifth",
                            "The sixth", "The seventh", "The eighth", "The ninth", "The tenth"]

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

llm = None  # initially None, not loaded

def load_llm(device='cpu'):
    global llm
    from sentence_transformers import SentenceTransformer
    llm = SentenceTransformer('thenlper/gte-large', device=device)

MINI_GRID_RADIUS = 1
EMBEDDING_SIZE = 1024

class MultiHeadingOccupancyGrid(OccupancyGrid):
    
    def __init__(self, x_dim, y_dim, num_cells, batch_size, num_targets, heading_mini_grid_radius=1, device='cpu'):
        super().__init__(x_dim, y_dim, num_cells, batch_size, num_targets, heading_mini_grid_radius, device)
        
        self.decoder_mlp = Decoder(emb_size=EMBEDDING_SIZE).to(self.device)
        self.decoder_mlp.load_state_dict(torch.load(model_path, map_location=self.device))
        self.decoder_mlp.eval()

        self.heading_embeddings = torch.zeros((self.batch_size,EMBEDDING_SIZE),device=self.device)
        self.heading_sentences = [[ "" for _ in range(self.num_targets) ] for _ in range(self.batch_size)]
        
        self.searching_hinted_target = torch.zeros((self.batch_size,self.num_targets), dtype=torch.bool, device=self.device)
    
    def spawn_llm_map(self, env_index, n_obstacles, n_agents, n_targets, target_class, llm_activate, padding = True):

        env_index = env_index.view(-1,1)
        packet_size = env_index.shape[0]
        target_poses = torch.zeros((packet_size,n_targets,2),device=self.device)
        unknown_targets = [] # Targets not hinted through a heading

        if padding: pad = 1 
        else: pad = 0
        
        hinted_target_count = 0
        headings_strings = [""] * packet_size
        
        for t in range(self.num_targets):
            # Convert world coordinates to grid indices
            rando = random.random()
            #randos = torch.randint(-self.heading_mini_grid_radius,self.heading_mini_grid_radius + 1,(packet_size,2), device=self.device).int() # Create a bit of chaos
            randos = torch.randint(0,self.heading_mini_grid_radius + 1,(packet_size,2), device=self.device).int()
            if rando < self.heading_to_target_ratio and llm_activate:
                
                self.searching_hinted_target[env_index,t] = True

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

                vec = torch.zeros((packet_size,2), device=self.device)
                for i, idx in enumerate(env_index):
                    sentence = f"{targets[i]} {insides[i]} the {dir_1[i]} {dir_2[i]} {positions[i]} {spaces[i]}."
                    self.heading_sentences[idx][t] = sentence
                    headings_strings[i] += f"{ordinal_prefixes[hinted_target_count]} {targets[i]} {insides[i]} the {dir_1[i]} {dir_2[i]} {positions[i]} {spaces[i]}."
                    d_1 = torch.tensor(direction_map[dir_types_1[i]], device=self.device)
                    d_2 = torch.tensor(direction_map[valid_d_types_2[i]], device=self.device)
                    vec[i] = d_1 + d_2
                hinted_target_count += 1
                
                x_rando_sign = (0.5 - vec[:,0])/torch.abs(0.5 - vec[:,0])
                y_rando_sign = (0.5 - vec[:,1])/torch.abs(0.5 - vec[:,1])

                vec[:,0] = torch.clamp(vec[:,0] * (self.grid_width - 1) + pad + x_rando_sign*randos[:,0], min=pad, max=self.grid_width - 1).int()
                vec[:,1] = torch.clamp(vec[:,1] * (self.grid_height - 1) + pad + y_rando_sign*randos[:,1], min=pad, max=self.grid_height - 1).int()

                self.gaussian_heading(env_index,vec,t,self.heading_mini_grid_radius,self.heading_mini_grid_radius)
                self.grid_targets[env_index, vec[:,1].unsqueeze(1).int(), vec[:,0].unsqueeze(1).int()] = (TARGET + target_class[env_index]) 

                target_poses[:,t] = self.grid_to_world(vec[:,0]-pad, vec[:,1]-pad)

            else:
                unknown_targets.append(t)
            
        if hinted_target_count > 0:
            embedding = torch.tensor(llm.encode(headings_strings), device = self.device).squeeze(0)
            self.heading_embeddings[env_index.squeeze(1)] = embedding
            

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
    
    def update_instruction(self, env_index):
        
        """
        Given a tensor of env indices, format each sentence snippet in heading_sentences with an ordinal prefix
        and concatenate them into a single instruction string for each environment index.

        Args:
            env_index (Tensor): 1D tensor of environment indices.

        Returns:
            List[str]: A list of formatted instruction strings, one for each index.
        """
        self.heading_embeddings[env_index] = 0.
        instructions = []
        update_indices = []
        
        for idx in env_index:
            idx = idx.item()
            sentence_list = self.heading_sentences[idx]

            formatted_sentences = []
            i = 0
            for sentence in sentence_list:
                if len(sentence) > 0:
                    prefix = ordinal_prefixes[i] if i < len(ordinal_prefixes) else f"The {i+1}th"
                    formatted_sentences.append(f"{prefix} {sentence.strip()}")
                    i += 1
            instruction = " ".join(formatted_sentences)
            if len(instruction) > 0:
                instructions.append(instruction)
                update_indices.append(idx)
            
        if instructions:
            embedding = torch.tensor(llm.encode(instructions), device=self.device)
            if embedding.ndim == 2:  # shape: (N, embedding_dim)
                self.heading_embeddings[update_indices] = embedding
            elif embedding.ndim == 1:  # edge case: single sentence, (embedding_dim,)
                self.heading_embeddings[update_indices[0]] = embedding
    
    def update_gaussian_heading(self, all_time_covered_targets: torch.Tensor):

        # Mask for all time found targets
        mask = all_time_covered_targets  # (batch, n_targets)
        # Remove potential for found headings
        self.grid_gaussian_heading[mask] = 0.0
        # Recompute total grid potential
        self.grid_heading = self.grid_gaussian_heading.max(dim=1).values
        # Visited cells are emptied out
        visit_mask = (self.grid_visited == VISITED)
        self.grid_heading[visit_mask] = 0.0
        
        # Recompute embedding if a hinted target is found.
        # 1) Find indices of found targets - Find where mask and self.searching_hinted_target are both true
        env_index, target_index = torch.where(mask & self.searching_hinted_target)
        # 2) Clear the cached sentence associated to that heading
        self.searching_hinted_target[env_index,target_index] = False # No longer searching for this target.
        for env_i, tgt_i in zip(env_index.tolist(), target_index.tolist()):
            self.heading_sentences[env_i][tgt_i] = ""
        # 3) Compute new instruction with remaining hinted targets (if any) and generate new embedding
        self.update_instruction(env_index)
    
    def reset_all(self):
        self.heading_sentences = [[ "" for _ in range(self.num_targets) ] for _ in range(self.batch_size)]
        self.searching_hinted_target.zero_()
        self.heading_embeddings.zero_()
        return super().reset_all()
    
    def reset_env(self, env_index):
        
        self.heading_sentences[env_index] = [ "" for _ in range(self.num_targets) ]   
        self.heading_embeddings[env_index].zero_()
        self.searching_hinted_target[env_index].zero_()
        return super().reset_env(env_index)
            

