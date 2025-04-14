import random
import torch
import numpy as np
import json
from occupancy_grid import OccupancyGrid, TARGET, OBSTACLE, VISITED_TARGET, VISITED
from vmas.simulator.core import Landmark
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import List

LARGE = 10
EMBEDDING_SIZE = 1024
DECODER_OUTPUT_SIZE = 25

train_dict = None
total_dict_size = None
data_grid_size = None
decoder_model = None

class Decoder(nn.Module):
    def __init__(self, emb_size, out_size, hidden_size=256):
        super().__init__()
        self.norm_input = nn.LayerNorm(emb_size)
        self.l0 = nn.Linear(emb_size, hidden_size)
        self.l1 = nn.Linear(hidden_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, out_size)
        self.norm_hidden = nn.LayerNorm(hidden_size)
        self.act = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.norm_input(x)
        x = self.dropout(self.act(self.l0(x)))
        x = self.norm_hidden(x)
        x = self.dropout(self.act(self.l1(x)))
        return torch.tanh(self.l2(x))

def load_decoder(model_path, device):
    
    global decoder_model
    decoder_model = Decoder(emb_size= EMBEDDING_SIZE, out_size=DECODER_OUTPUT_SIZE)
    decoder_model.load_state_dict(torch.load(model_path, map_location=device))
    decoder_model.eval()
    
def load_task_data(json_path, use_decoder, device='cpu'):

    global train_dict
    global total_dict_size
    with open(json_path, 'r') as f:
        data = [json.loads(line) for line in f]

    np.random.shuffle(data)

    def process_dataset(dataset):
        output = {}

        if all("embedding" in entry for entry in dataset):
            embeddings = [entry["embedding"] for entry in dataset]
            output["task_embedding"] = torch.tensor(embeddings, dtype=torch.float32, device=device)
        
        if all("gemini_response" in entry for entry in dataset):
            sentences = [entry["gemini_response"] for entry in dataset]
            output["sentence"] = sentences

        if all("grid" in entry for entry in dataset) and not use_decoder:
            grids = [[*entry["grid"]] for entry in dataset]
            output["grid"] = torch.tensor(grids, dtype=torch.float32, device=device)
        else:
            grids = [decoder_model(torch.tensor(entry["embedding"],device=device)) for entry in dataset]
            output["grid"] = torch.stack(grids)

        if all("class" in entry for entry in dataset):
            classes = [entry["class"] for entry in dataset]
            output["class"] = torch.tensor(classes, dtype=torch.float32, device=device)
        else:
            print("Target Class wasn't found in the dataset so reverting back to randomized classes")

        if all("max_targets" in entry for entry in dataset):
            max_targets = [entry["max_targets"] for entry in dataset]
            output["max_targets"] = torch.tensor(max_targets, dtype=torch.float32, device=device)
        else:
            print("Max target not found in the dataset, so reverting back to randomized max num")

        return output

    train_dict = process_dataset(data)
    total_dict_size = next(iter(train_dict.values())).shape[0]
    

MINI_GRID_RADIUS = 1
EMBEDDING_SIZE = 1024
DATA_GRID_SHAPE = (5,5)
DATA_GRID_NUM_TARGET_PATCHES = 1

class GeneralPurposeOccupancyGrid(OccupancyGrid):
    
    def __init__(self, x_dim, y_dim, num_cells, batch_size, num_targets, heading_mini_grid_radius=1, device='cpu'):
        super().__init__(x_dim, y_dim, num_cells, batch_size, num_targets, heading_mini_grid_radius, device)

        self.embeddings = torch.zeros((self.batch_size,EMBEDDING_SIZE),device=self.device)
        self.sentences = [[ "" for _ in range(self.num_targets) ] for _ in range(self.batch_size)]
        self.searching_hinted_target = torch.zeros((self.batch_size,), dtype=torch.bool, device=self.device)
        self.use_embedding_ratio = 0.95
        self.heading_lvl_threshold = 0.5
        
        self.target_attribute_embedding_found = False
        self.max_target_embedding_found = False
    
    def sample_dataset(self,env_index, packet_size, target_class, max_target_count, num_target_groups):
        
        sample_indices = torch.randperm(total_dict_size)[:packet_size]
        
        # Sample tensors
        task_dict = {key: value[sample_indices] for key, value in train_dict.items() if key in train_dict and key != "sentence"}
        # Sample sentences
        task_dict["sentence"] = [sentence for sentence in train_dict["sentence"]]

        if "task_embedding" in task_dict:
            self.embeddings[env_index] = task_dict["task_embedding"].unsqueeze(1)
        
        if "sentence" in task_dict:
            for idx in env_index:
                self.sentences[idx] = task_dict["sentence"][idx]

        if "class" in task_dict:
            target_class[env_index] = task_dict["class"].unsqueeze(1).int()
            self.target_attribute_embedding_found = True
        else: 
            target_class[env_index] = torch.randint(0, num_target_groups, (packet_size,1), dtype=torch.int, device=self.device)

        if "max_targets" in task_dict:
            max_target_count[env_index] = task_dict["max_targets"]
            self.max_target_embedding_found = True
        else:
            max_target_count[env_index] = self.num_targets # This will change

        if "grid" in task_dict:
            new_grids_scaled = F.interpolate(
                task_dict["grid"].reshape(-1, *DATA_GRID_SHAPE).unsqueeze(1),
                size=(self.grid_height, self.grid_width),
                mode='nearest'
            )
            pad_w = self.padded_grid_height - self.grid_width
            pad_h = self.padded_grid_height - self.grid_height
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            new_grids_scaled = F.pad(
                new_grids_scaled,
                (pad_left, pad_right, pad_top, pad_bottom),
                mode='constant',
                value=0
            )
            self.grid_heading[env_index] = new_grids_scaled

    def get_target_pose_in_heading(
        self,
        env_index: torch.Tensor,
        packet_size: int,
        valid_heading_threshold: float):
        
        flat_grid = self.grid_heading[env_index].view(packet_size, -1)  # (B, H*W)
        
        # Mask out values below the threshold
        valid_mask = flat_grid > valid_heading_threshold  # (B, H*W)
        masked_grid = flat_grid * valid_mask  # Zero out invalid cells

        num_valid = valid_mask.sum(dim=1)  # (B,)
        no_valid = num_valid == 0

        # Initialize selection mask
        selection_mask = torch.zeros_like(flat_grid, dtype=torch.bool)

        for i in range(packet_size):
            if num_valid[i] > 0:
                probs = masked_grid[i]
                if probs.sum() == 0:
                    continue  # Avoid division by zero if all weights are zero
                sampled_idx = torch.multinomial(probs, 1)
                selection_mask[i, sampled_idx] = True

        chosen_flat_idx = selection_mask.float().argmax(dim=1)
        y = chosen_flat_idx // self.padded_grid_width
        x = chosen_flat_idx % self.padded_grid_width

        return torch.stack([x, y], dim=1), no_valid  # (B, 2)
    
    def generate_random_grid(
        self,
        env_index,
        packet_size,
        n_agents,
        n_obstacles,
        unknown_targets,
        target_class,
        target_poses, padding):
        
        n_unknown_targets = len(unknown_targets)
        n_known_targets = self.num_targets - n_unknown_targets

        grid_size = self.grid_width * self.grid_height

        assert grid_size >= n_obstacles + n_agents + self.num_targets , "Not enough room for all entities"

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
        pad = 1 if padding else 0
        self.grid_obstacles[env_index.unsqueeze(1), obstacle_grid_y+pad, obstacle_grid_x+pad] = OBSTACLE  # Mark obstacles# Mark targets 
        # Convert to world coordinates
        obstacle_centers = self.grid_to_world(obstacle_grid_x, obstacle_grid_y)  # Ensure shape (packet_size, n_obstacles, 2)
        target_centers = self.grid_to_world(target_grid_x, target_grid_y)  # Ensure shape (packet_size, n_targets, 2)
        
        for i, (j,t_dict) in enumerate(unknown_targets.items()):
            for t, mask in t_dict.items():
                
                target_poses[mask,j,t] = target_centers[mask,i,0]
                self.grid_targets[env_index[mask], target_grid_y[mask,j].squeeze(1)+pad, target_grid_x[mask,j].squeeze(1)+pad] = (TARGET + j)
                
        agent_centers = self.grid_to_world(agent_grid_x,agent_grid_y)
        
        return agent_centers, obstacle_centers, target_poses
         
    def spawn_llm_map(
        self,
        env_index: torch.Tensor,
        n_obstacles: int,
        n_agents: int,
        target_groups: List[List[Landmark]],
        target_class: torch.Tensor,
        max_target_count: torch.Tensor,
        padding = True):
        
        """ This function handles the scenario reset. It is unbelievably complicated."""

        # Environments being reset
        env_index = env_index.view(-1,1)
        packet_size = env_index.shape[0]
        
        # Target Tree: Each class can have X targets
        num_target_groups = len(target_groups)
        num_targets_per_class = len(target_groups[0])
        
        # Vector to hold new target positions
        target_poses = torch.zeros((packet_size,num_target_groups,num_targets_per_class,2),device=self.device)
        
        # Dictionary to hold targets not hinted through a heading
        unknown_targets = {} 

        # Padding around the grid, to avoid hitting the edges too much.
        if padding: pad = 1 
        else: pad = 0
        
        # Increase robustness by ommitting the embedding sometimes, forces the team to revert back to regular exploration
        rando = random.random()
        use_embedding = rando < self.use_embedding_ratio
        
        if use_embedding: # Case we sample the dataset for the new scenario + Embedding
            if train_dict is not None and total_dict_size is not None:
                self.sample_dataset(env_index, packet_size, target_class, max_target_count, num_target_groups)
        
        else: # Case that we are not using the embedding for robustness
            max_target_count[env_index] = num_targets_per_class
            target_class[env_index] = torch.randint(0, num_target_groups, (packet_size,1), dtype=torch.int, device=self.device)
            # Embedding is already zero
            # Sentence is already [""]
        
        # Cycle through each target and assign new positions
        for j in range(num_target_groups):
            unknown_target_dict = {}
            for t in range(num_targets_per_class):

                # Mask to hold environment indices that are targetting class j
                mask = (target_class[env_index] == j).squeeze(1)
                envs = env_index[mask]
                
                # Targets 
                self.searching_hinted_target[envs] = True
                
                # Cancel mask: Environments which are not targetting class j or don't meet the valid heading threshold
                declined_targets_mask = (~mask).clone()
                
                # Environments which are not targetting class j
                unknown_target_dict[t] = declined_targets_mask 
                
                # Case we are using an embedding and some environments are targetting j
                if use_embedding and mask.any():
                    
                    # Get new target positions
                    vec, no_ones = self.get_target_pose_in_heading(envs,envs.numel(),valid_heading_threshold=self.heading_lvl_threshold)
                    
                    # If the heading doesn't meet the validity threshold,
                    # we cancel the embedding and target (j,t) is placed randomly
                    if no_ones.any() > 0: 
                        declined_targets_mask[mask] += no_ones
                        unknown_target_dict[t] = declined_targets_mask
                        target_class[envs] = 0 # Default target when no embedding?
                        self.embeddings[envs[no_ones]].zero_()
                        self.grid_heading[envs[no_ones]].zero_()
                        self.searching_hinted_target[envs[no_ones]] = False
                        
                        for idx in envs[no_ones]:
                            self.sentences[idx] = ""
                    
                    # Place the target in the grid
                    self.grid_targets[envs, vec[:,1].unsqueeze(1).int(), vec[:,0].unsqueeze(1).int()] = (TARGET + j) 
                    # Store world position
                    target_poses[mask,j,t] = self.grid_to_world(vec[:,0]-pad, vec[:,1]-pad)
                
                # Case we are not using an embedding or no environments are targetting j    
                else:
                    declined_targets_mask += mask # aka: all environments
                    unknown_target_dict[t] = declined_targets_mask 
                     # Check this
            unknown_targets[j] = unknown_target_dict
                        
        # Generate random obstacles, agents (allways in a line somewhere) and unknown targets
        agent_centers, obstacle_centers, target_poses = self.generate_random_grid(env_index, packet_size, n_agents, n_obstacles, unknown_targets, target_class, target_poses, padding)

        return obstacle_centers.squeeze(-2), agent_centers.squeeze(-2), target_poses
    
    def compute_heading_bonus(self,pos, heading_exploration_rew_coeff = 1.0):
        """Heading grid is either +1: region of interest, or -1: region to avoid"""

        grid_x, grid_y = self.world_to_grid(pos, padding=True)

        in_heading_cell = (self.grid_heading[torch.arange(pos.shape[0]),grid_y,grid_x] > self.heading_lvl_threshold).float()
        in_danger_cell = (self.grid_heading[torch.arange(pos.shape[0]),grid_y,grid_x] < 0).float()
        
        visit_lvl = self.grid_visits_sigmoid[torch.arange(pos.shape[0]), grid_y, grid_x]
        new_cell_bonus = (visit_lvl < 0.2).float() * heading_exploration_rew_coeff

        return in_heading_cell * new_cell_bonus - in_danger_cell * new_cell_bonus
    
    def observe_embeddings(self):

        return self.embeddings.flatten(start_dim=1,end_dim=-1)
    
    def reset_all(self):
        self.sentences = [[ "" for _ in range(self.num_targets) ] for _ in range(self.batch_size)]
        self.embeddings.zero_()
        self.searching_hinted_target.zero_()
        return super().reset_all()
    
    def reset_env(self, env_index):
        
        self.sentences[env_index] = [ "" for _ in range(self.num_targets) ]   
        self.embeddings[env_index].zero_()
        self.searching_hinted_target[env_index].zero_()
        return super().reset_env(env_index)
            

