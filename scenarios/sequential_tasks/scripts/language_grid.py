import random
import torch
import numpy as np
import json

from scenarios.grids.core_grid import CoreGrid
from scenarios.grids.environment_grids import TARGET, OBSTACLE
from scenarios.grids.internal_grids import InternalOccupancyGrid
from scenarios.grids.environment_grids import EnvironmentGrid

from vmas.simulator.core import Landmark
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import List, Dict

LARGE = 10
DECODER_OUTPUT_SIZE = 100
CONFIDENCE_HIGH = 0.
CONFIDENCE_LOW = 2.

MINI_GRID_RADIUS = 2
DATA_GRID_SHAPE = (10,10)
DATA_GRID_NUM_TARGET_PATCHES = 1

# Tasks
EXPLORE = 0
NAVIGATE = 1
DEFEND_TIGHT = 2
DEFEND_WIDE = 3
IDLE = 4

train_dict = None
total_dict_size = None
data_grid_size = None
decoder_model = None

class Decoder(nn.Module):
    def __init__(self, emb_size, out_size, hidden_size=128):
        super().__init__()
        self.l0 = nn.Linear(emb_size, hidden_size)
        self.l1 = nn.Linear(hidden_size, out_size)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.act(self.l0(x))
        return torch.sigmoid(self.l1(x))

def load_decoder(model_path, embedding_size, device):
    
    global decoder_model
    decoder_model = Decoder(emb_size= embedding_size, out_size=DECODER_OUTPUT_SIZE)
    decoder_model.load_state_dict(torch.load(model_path, map_location=device))
    decoder_model.eval()
    
def load_task_data(
    json_path,
    use_decoder,
    use_grid_data,
    device='cpu'):
    global train_dict
    global total_dict_size

    # Resolve path to ensure it's absolute and correct regardless of cwd
    project_root = Path(__file__).resolve().parents[3]  # Adjust depending on depth of current file
    full_path = project_root / json_path

    with full_path.open('r') as f:
        data = json.load(f)

    np.random.shuffle(data)

    def process_dataset(dataset):
        output = {}

        if all("states" in entry for entry in dataset):
            states = [entry["states"] for entry in dataset]
            output["states"] = states
        
        if all("y" in entry for entry in dataset):
            task = [entry["y"] for entry in dataset]
            output["task"] = torch.tensor(task, dtype=torch.float32, device=device)
        
        if all("h" in entry for entry in dataset):
            embeddings = [torch.tensor(entry["h"],dtype=torch.float32, device=device) for entry in dataset]
            output["subtasks"] = embeddings
        
        if all("summary" in entry for entry in dataset):
            sentences = [entry["summary"] for entry in dataset]
            output["summary"] = sentences

        if all("grid" in entry for entry in dataset) and use_grid_data:
            grids = [[*entry["grid"]] for entry in dataset]
            output["grid"] = torch.tensor(grids, dtype=torch.float32, device=device)
            
        elif use_decoder:
            grids = [decoder_model(torch.tensor(entry["embedding"], device=device)) for entry in dataset]
            grid_tensor = torch.stack(grids)  # shape: (batch_size, grid_dim^2)

            # Step 1: Min-max normalize each sample individually
            min_vals = grid_tensor.min(dim=1, keepdim=True).values
            max_vals = grid_tensor.max(dim=1, keepdim=True).values
            denom = max_vals - min_vals
            normalized = torch.where(denom > 0, (grid_tensor - min_vals) / denom, torch.zeros_like(grid_tensor))

            # Step 2: Apply fixed threshold (e.g., 0.8)
            threshold = 0.8
            above_thresh = normalized >= threshold

            # Step 3: Subtract threshold only for values above it
            rescaled = torch.zeros_like(normalized)
            rescaled[above_thresh] = normalized[above_thresh] - threshold

            # Step 4: Normalize remaining values to [0, 1] per sample
            new_max_vals = rescaled.max(dim=1, keepdim=True).values
            rescaled = torch.where(new_max_vals > 0, rescaled / new_max_vals, torch.zeros_like(rescaled))

            output["grid"] = rescaled

        else:
            grids = [[0.0] * DATA_GRID_SHAPE[0] * DATA_GRID_SHAPE[1]  for _ in dataset]

        return output

    train_dict = process_dataset(data)
    total_dict_size = len(next(iter(train_dict.values())))
    

def apply_density_diffusion(grid, kernel_size=3, sigma=1.0):
    # Create a Gaussian kernel for diffusion
    import math

    def gaussian_kernel(k, sigma):
        ax = torch.arange(-k // 2 + 1., k // 2 + 1.)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel

    kernel = gaussian_kernel(kernel_size, sigma).to(grid.device)
    kernel = kernel.expand(grid.size(1), 1, kernel_size, kernel_size)

    # Apply convolution with padding
    padding = kernel_size // 2
    blurred = F.conv2d(grid, kernel, padding=padding, groups=grid.size(1))

    # Renormalize to preserve total density (area)
    total_mass_before = grid.sum(dim=(2, 3), keepdim=True)
    total_mass_after = blurred.sum(dim=(2, 3), keepdim=True)
    blurred = blurred * (total_mass_before / (total_mass_after + 1e-8))

    return blurred

class LanguageGrid(CoreGrid):
    
    def __init__(self, x_dim, y_dim, x_scale, y_scale, num_cells, batch_size, num_targets, num_targets_per_class, visit_threshold, embedding_size, use_embedding_ratio, device='cpu'):
        
        super().__init__(x_dim, y_dim, x_scale, y_scale, num_cells, batch_size, device)
        
        # Internal Grid Memory
        self.internal_grid = InternalOccupancyGrid(x_dim,y_dim,x_scale,y_scale,num_cells,visit_threshold,batch_size,device)
        # Environment Grid
        self.environment = EnvironmentGrid(x_dim,y_dim,x_scale,y_scale,num_cells,batch_size,device)

        # Useful parameters
        self.num_targets = num_targets
        self.use_embedding_ratio = use_embedding_ratio
        self.heading_lvl_threshold = 0.5
        self.target_attribute_embedding_found = False
        self.max_target_embedding_found = False
        self.confidence_embedding_found = False
        self.num_targets_per_class = num_targets_per_class
        
        # Language Driven Grids
        self.searching_hinted_target = torch.zeros((self.batch_size,), dtype=torch.bool, device=self.device)
        self.grid_gaussian_heading = torch.zeros((batch_size,num_targets_per_class,self.padded_grid_height, self.padded_grid_width), device=self.device)
        self.grid_heading = torch.zeros((batch_size,self.padded_grid_height, self.padded_grid_width), device=self.device)
        self.num_heading_cells = torch.zeros((batch_size,), device=self.device)
        self.heading_coverage_ratio = torch.zeros((batch_size,), device=self.device)
        
        # Task
        self.embedding_size = embedding_size
        self.task_embeddings = torch.zeros((self.batch_size,embedding_size),device=self.device)
        self.subtask_embeddings = torch.zeros((self.batch_size,embedding_size),device=self.device)
        self.states = torch.zeros((self.batch_size,), device=self.device)
        self.summary = [ "" for _ in range(self.batch_size)]
        
    def sample_dataset(self,env_index, packet_size):
        
        # --- pick indices ------------------------------------------------------------     # or any key with same length
        if packet_size <= total_dict_size:
            # Normal case: sample *without* replacement
            sample_indices = torch.randperm(total_dict_size, device=self.device)[:packet_size]
        else:
            # Need repeats → build “base” + “extra” indices
            repeats, remainder = divmod(packet_size, total_dict_size)

            # 1) repeat every index the same number of times
            base = torch.arange(total_dict_size, device=self.device).repeat(repeats)

            # 2) top-up with a random subset for the leftover slots
            extra = torch.randperm(total_dict_size, device=self.device)[:remainder] \
                    if remainder > 0 else torch.empty(0, dtype=torch.long, device=self.device)

            sample_indices = torch.cat([base, extra])
        
        # Sample tensors
        task_dict = {key: value[sample_indices] for key, value in train_dict.items() if key in train_dict and key not in ["states", "subtasks", "summary"]}
        # Sample sentences
        indices_list = sample_indices.tolist()
        task_dict["summary"] = [train_dict["summary"][i] for i in indices_list]
        task_dict["subtasks"] = [train_dict["subtasks"][i] for i in indices_list]
        task_dict["states"] = [train_dict["states"][i] for i in indices_list]
        
        subtask_indices = torch.zeros(packet_size, dtype=torch.long, device=self.device)

        if "task" in task_dict:
            self.task_embeddings[env_index] = task_dict["task"].unsqueeze(1)
        
        if "summary" in task_dict:
            for i , idx in enumerate(env_index):
                self.summary[idx] = task_dict["summary"][i]

        if "grid" in task_dict:
            raw_grids = task_dict["grid"].reshape(-1, *DATA_GRID_SHAPE).unsqueeze(1)
            
            # Optional: apply diffusion on original resolution
            #raw_grids = apply_density_diffusion(raw_grids, kernel_size=5, sigma=2.0)
            new_grids_scaled = F.interpolate(
                raw_grids,
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

            # Flip vertically to match bottom-left origin
            new_grids_scaled = torch.flip(new_grids_scaled, dims=[2])

            # Normalize so sum over spatial dims == 1
            num_heading_cells = (new_grids_scaled.sum(dim=(2, 3), keepdim=True) + 1e-8)
            new_grids_scaled = new_grids_scaled / num_heading_cells

            self.num_heading_cells[env_index] = num_heading_cells.view(-1,1)
            self.grid_heading[env_index] = new_grids_scaled
        
        if "subtasks" in task_dict:
            for i , idx in enumerate(env_index):
                num_subtasks = task_dict["subtasks"][i].shape[0]
                subtask_idx =  random.randint(0, num_subtasks - 1) if num_subtasks > 0 else 0
                subtask_indices[i] = subtask_idx
                self.subtask_embeddings[idx] = task_dict["subtasks"][i][subtask_idx]
        
        if "states" in task_dict: 
            for i , idx in enumerate(env_index):
                state = task_dict["states"][i][subtask_indices[i]]
                if state == 'E':
                    self.states[idx] = EXPLORE
                elif state == 'N':
                    self.states[idx] = NAVIGATE
                elif state == 'P1':
                    self.states[idx] = DEFEND_TIGHT
                elif state == 'P2':
                    self.states[idx] = DEFEND_WIDE
                elif state == 'F':
                    self.states[idx] = IDLE
                else:
                    raise ValueError(f"Unknown state {state} in task data")
                
    def get_target_pose_in_heading(
        self,
        env_index: torch.Tensor,         # (B,)
        packet_size: int,
        padding: bool
    ) -> torch.Tensor:                   # → (B, 2)  [x, y] integer grid coords
        # --------------------------------------------------------------------------
        flat_grid   = self.grid_heading[env_index].view(packet_size, -1)     # (B, H*W)
        valid_mask  = flat_grid > 5e-4
        masked_grid = flat_grid * valid_mask
        num_valid   = valid_mask.sum(dim=1)

        # probability of falling back to a uniform pick
        p_uniform = 0    # 50 % @ LOW → 0 % @ HIGH

        rand_u      = torch.rand(packet_size, device=flat_grid.device)
        do_uniform  = (rand_u < p_uniform) | (num_valid == 0)

        # --------------------------------------------------------------------------
        chosen_idx  = torch.empty(packet_size, dtype=torch.long,
                                device=flat_grid.device)

        # --- uniform branch --------------------------------------------------------
        if do_uniform.any():
            n_uni                     = do_uniform.sum()
            chosen_idx[do_uniform]    = torch.randint(
                0, self.num_cells, (n_uni,), device=flat_grid.device
            )

        # --- weighted-sampling branch ---------------------------------------------
        if (~do_uniform).any():
            probs          = masked_grid[~do_uniform]
            probs_sum      = probs.sum(dim=1, keepdim=True)
            probs          = probs / probs_sum               # safe: probs_sum>0 by construction
            chosen_idx[~do_uniform] = torch.multinomial(probs, 1).squeeze(1)

        # --------------------------------------------------------------------------
        pad           = 1 if padding else 0
        x = torch.empty_like(chosen_idx)
        y = torch.empty_like(chosen_idx)

        # coords for uniform picks (use non-padded grid, then add pad)
        if do_uniform.any():
            uni_idx        = chosen_idx[do_uniform]
            y[do_uniform]  = uni_idx // self.grid_width  + pad
            x[do_uniform]  = uni_idx %  self.grid_width  + pad

        # coords for weighted picks (grid is already padded)
        if (~do_uniform).any():
            wtd_idx        = chosen_idx[~do_uniform]
            y[~do_uniform] = wtd_idx // self.padded_grid_width
            x[~do_uniform] = wtd_idx %  self.padded_grid_width

        return torch.stack((x, y), dim=1)
    
    def generate_random_grid(
        self,
        env_index: torch.Tensor,
        packet_size: int,
        n_agents: int,
        n_obstacles: int,
        unknown_targets: Dict[int,torch.Tensor],
        target_poses: torch.Tensor,
        not_explore_mask: torch.Tensor,
        padding):

        grid_size = self.grid_width * self.grid_height
        assert grid_size >= n_obstacles + n_agents + self.num_targets , "Not enough room for all entities"

        # Generate random values and take the indices of the top `n_obstacles` smallest values
        rand_values = torch.rand(packet_size, grid_size, device=self.device)
        
        for j, mask in unknown_targets.items():
            for t in range(self.num_targets_per_class):
                vec = target_poses[~mask,j,t]
                grid_x, grid_y = self.world_to_grid(vec,padding=False)
                indices = grid_y * self.grid_width + grid_x
                rand_values[~mask,indices] = LARGE - 1

        # Extract obstacle and agent indices
        sort_indices = torch.argsort(rand_values, dim=1)
        obstacle_indices = sort_indices[:, :n_obstacles]  # First n_obstacles indices
        #agent_indices = sort_indices[:, -n_agents:]
        # Random Agents
        agent_indices = sort_indices[:, n_obstacles:n_obstacles+n_agents]
        
        # Convert flat indices to (x, y) grid coordinates
        obstacle_grid_x = (obstacle_indices % self.grid_width).view(packet_size, n_obstacles, 1)
        obstacle_grid_y = (obstacle_indices // self.grid_width).view(packet_size, n_obstacles, 1)
        
        agent_grid_x = (agent_indices % self.grid_width).view(packet_size, n_agents, 1)
        agent_grid_y = (agent_indices // self.grid_width).view(packet_size, n_agents, 1)
        
        # Update grid_obstacles for the given environments and adjust for padding
        pad = 1 if padding else 0
        self.environment.grid_obstacles[env_index.unsqueeze(1), obstacle_grid_y+pad, obstacle_grid_x+pad] = OBSTACLE  # Mark obstacles# Mark targets 
        # Convert to world coordinates
        obstacle_centers = self.grid_to_world(obstacle_grid_x, obstacle_grid_y)  # Ensure shape (packet_size, n_obstacles, 2)
        agent_centers = self.grid_to_world(agent_grid_x,agent_grid_y)
        
        target_indices = sort_indices[:, n_obstacles:n_obstacles + self.num_targets]
        target_grid_x = (target_indices % self.grid_width)
        target_grid_y = (target_indices // self.grid_width)
        target_center = self.grid_to_world(target_grid_x, target_grid_y)
          
        t = 0
        for j, mask in unknown_targets.items():
            target_poses[mask,j,:] = target_center[mask,t:t+self.num_targets_per_class]
            self.environment.grid_targets[env_index[mask], target_grid_y[mask,t:t+self.num_targets_per_class]+pad, target_grid_x[mask,t:t+self.num_targets_per_class]+pad] = (TARGET + j)
            #self.grid_visits_sigmoid[env_index[mask], target_grid_y[mask,t:t+self.num_targets_per_class]+pad, target_grid_x[mask,t:t+self.num_targets_per_class]+pad] = 1.0
            t += self.num_targets_per_class
            
        # If task is NAVIGATE or DEFEND, we need to spawn one agent near a target
        sample = None
            
        # Sample a target index for each environment
        target_class_indices = torch.randint(0, target_poses.size(1), (packet_size,), device=self.device)
        target_indices = torch.randint(0, target_poses.size(2), (packet_size,), device=self.device)
        batch_indices = torch.arange(packet_size, device=self.device)
        sample = target_poses[batch_indices,target_class_indices,target_indices][not_explore_mask]
        
        # Select one target for each environment
        target_grid_x, target_grid_y = self.world_to_grid(target_poses[batch_indices,target_class_indices,target_indices], padding=False)
        
        # Place the agent near the target
        agent_grid_x = target_grid_x + torch.randint(-MINI_GRID_RADIUS, MINI_GRID_RADIUS + 1, (packet_size,), device=self.device)
        agent_grid_y = target_grid_y + torch.randint(-MINI_GRID_RADIUS, MINI_GRID_RADIUS + 1, (packet_size,), device=self.device)
        
        # Ensure the agent is within bounds # check bounds 
        agent_grid_x = torch.clamp(agent_grid_x, 0, self.grid_width - 1)
        agent_grid_y = torch.clamp(agent_grid_y, 0, self.grid_height - 1)
        
        # Update agent_centers with the new positions
        agent_centers[not_explore_mask,0] = self.grid_to_world(agent_grid_x[not_explore_mask], agent_grid_y[not_explore_mask]).unsqueeze(1)
            
        return agent_centers, obstacle_centers, target_poses, sample
         
    def spawn_llm_map(
        self,
        env_index: torch.Tensor,
        n_obstacles: int,
        n_agents: int,
        target_groups: List[List[Landmark]],
        target_class: torch.Tensor,
        padding = True):
        
        """ This function handles the scenario reset. It is unbelievably complicated."""

        # Environments being reset
        env_index = env_index.view(-1,1)
        packet_size = env_index.shape[0]
        
        # Target Tree: Each class can have X targets
        num_target_groups = len(target_groups)
        if num_target_groups > 0:
            num_targets_per_class = len(target_groups[0])
        else:
            num_targets_per_class = 0
        
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
            target_class[env_index] = torch.zeros((packet_size,1), dtype=torch.int, device=self.device)
            if train_dict is not None and total_dict_size is not None:
                self.sample_dataset(env_index, packet_size)

        # Cycle through each target and assign new positions
        for j in range(num_target_groups):
            mask = (target_class[env_index] == j).squeeze(1)
            
            if use_embedding:

                # Cancel mask: Environments which are not targetting class j but target is still randomized
                declined_targets_mask = (~mask).clone()
                unknown_targets[j] = declined_targets_mask
                
                envs = env_index[mask]
                self.searching_hinted_target[envs] = True
        
                if mask.any():
                    for t in range(num_targets_per_class):
                        # Get new target positions
                        vec = self.get_target_pose_in_heading(envs,envs.numel(), padding)
                        # Place the target in the grid (and mark as visited, this a test)
                        self.environment.grid_targets[envs, vec[:,1].unsqueeze(1).int(), vec[:,0].unsqueeze(1).int()] = (TARGET + j)
                        self.gaussian_heading(envs,t,vec)
                        # Store world position
                        target_poses[mask,j,t] = self.grid_to_world(vec[:,0]-pad, vec[:,1]-pad)
            else:
                unknown_targets[j] = mask

                        
        # Generate random obstacles, agents (allways in a line somewhere) and unknown targets
        not_explore_mask = ((self.states[env_index] != EXPLORE) & (self.states[env_index] != IDLE)).squeeze(1)
        not_explore_idx = env_index[not_explore_mask].squeeze(1)
        agent_centers, obstacle_centers, target_poses, sample = self.generate_random_grid(env_index, packet_size, n_agents, n_obstacles, unknown_targets, target_poses, not_explore_mask, padding)
        
        #If task is NAVIGATION or DEFEND, Update internal grid with found target
        self.internal_grid.update(sample, MINI_GRID_RADIUS, self.environment.grid_targets, not_explore_idx)
        return obstacle_centers.squeeze(-2), agent_centers.squeeze(-2), target_poses
    
    
    def gaussian_heading(self, env_index: torch.Tensor, t_index: int, pos: torch.Tensor, sigma_coef=0.05):
        """
        pos: (batch_size, 2)
        env_index: (batch_size,)
        """

        batch_size = pos.shape[0]
        sigma_x = sigma_coef * self.grid_width
        sigma_y = sigma_coef * self.grid_height

        # Create meshgrid once for all grid points
        x_range = torch.arange(self.padded_grid_width, device=pos.device).float()
        y_range = torch.arange(self.padded_grid_height, device=pos.device).float()
        grid_x, grid_y = torch.meshgrid(y_range, x_range, indexing='xy')  # shape: (H, W)

        grid_x = grid_x.unsqueeze(0)  # (1, W, H)
        # Expand to batch size
        grid_y = grid_y.unsqueeze(0)  # (1, W, H)

        pos_x = pos[:,0].unsqueeze(1).unsqueeze(2)  # (B, 1, 1)
        pos_y = pos[:,1].unsqueeze(1).unsqueeze(2)  # (B, 1, 1)

        dist_x = ((grid_x - pos_x) / sigma_x) ** 2
        dist_y = ((grid_y - pos_y) / sigma_y) ** 2

        heading_val = (1 / (2 * torch.pi * sigma_x * sigma_y)) * torch.exp(-0.5 * (dist_x + dist_y))  # (B, W, H)
        heading_val = heading_val / heading_val.view(batch_size, -1).max(dim=1)[0].view(-1, 1, 1)

        # Update grid_heading only if the new value is higher
        for i in range(batch_size):
            self.grid_gaussian_heading[env_index[i],t_index] = heading_val[i]
    
    def observe_task_embeddings(self):

        return self.task_embeddings.flatten(start_dim=1,end_dim=-1)
    
    def observe_subtask_embeddings(self):

        return self.subtask_embeddings.flatten(start_dim=1,end_dim=-1)
    
    def update_heading_coverage_ratio(self):
        """ Update the ratio of heading cells covered by the agent. """
        num_heading_cells_covered = ((self.grid_heading > 0) & (self.internal_grid.grid_visits_sigmoid > 0)).sum(dim=(1, 2))
        self.heading_coverage_ratio = num_heading_cells_covered / self.num_heading_cells
    
    def update_multi_target_gaussian_heading(self, all_time_covered_targets: torch.Tensor, target_class):

        # All found heading regions are reset
        mask = all_time_covered_targets[torch.arange(0,self.batch_size),target_class]  # (batch, n_targets)
        self.grid_gaussian_heading[mask] = 0.0
        
    def compute_coverage_ratio_bonus(self, coverage_action):
        """Reward if coverage action is close to self.heading_coverage_ratio"""
        coverage_ratio = self.heading_coverage_ratio.view(-1, 1)
        coverage_ratio_bonus = torch.exp(-torch.abs(coverage_action - coverage_ratio) / 0.2) - 0.5 # 
        return coverage_ratio_bonus
        
    def compute_region_heading_bonus(self, pos:torch.Tensor, heading_exploration_rew_coeff = 1.0):
        """Reward is independent of the grid_heading values, rather fixed by the coefficient"""

        grid_x, grid_y = self.world_to_grid(pos, padding=True)

        in_heading_cell = (self.grid_heading[torch.arange(pos.shape[0]),grid_y,grid_x] > self.heading_lvl_threshold).float()
        in_danger_cell = (self.grid_heading[torch.arange(pos.shape[0]),grid_y,grid_x] < 0).float()
        
        visit_lvl = self.internal_grid.grid_visits_sigmoid[torch.arange(pos.shape[0]), grid_y, grid_x]
        new_cell_bonus = (visit_lvl < 0.2).float() * heading_exploration_rew_coeff

        return in_heading_cell * new_cell_bonus - in_danger_cell * new_cell_bonus
    
    def compute_region_heading_bonus_normalized(self, pos:torch.Tensor, heading_exploration_rew_coeff = 1.0):
        """Reward potential is constant. Individual cell reward depends on heading grid size."""

        grid_x, grid_y = self.world_to_grid(pos, padding=True)

        heading_lvl = self.grid_heading[torch.arange(pos.shape[0]),grid_y,grid_x] 
        
        visit_lvl = self.internal_grid.grid_visits_sigmoid[torch.arange(pos.shape[0]), grid_y, grid_x]
        new_cell_bonus = (visit_lvl < 0.2).float() * heading_exploration_rew_coeff

        return heading_lvl * new_cell_bonus
    
    def compute_gaussian_heading_bonus(self, pos:torch.Tensor, heading_exploration_rew_coeff = 1.0):
        
        """ Reward increases as we approach the center of the heading region"""
        heading_exploration_rew_coeff /= (0.05 * self.grid_height * self.grid_width)
        grid_x, grid_y = self.world_to_grid(pos, padding=True)
        heading_merged = self.grid_gaussian_heading.max(dim=1).values
        heading_val = heading_merged[torch.arange(pos.shape[0]),grid_y,grid_x]
        visit_lvl = self.internal_grid.grid_visits_sigmoid[torch.arange(pos.shape[0]), grid_y, grid_x]
        new_cell_bonus = (visit_lvl < 0.2).float() * heading_exploration_rew_coeff
        
        return new_cell_bonus * heading_val
        #return heading_val * heading_exploration_rew_coeff  # Open question: Should the heading bonus degrade after visiting the cell or not? 
    
    def reset_all(self):

        self.grid_gaussian_heading.zero_()
        self.grid_heading.zero_()
        self.searching_hinted_target.zero_()
        self.num_heading_cells.zero_()
        self.heading_coverage_ratio.zero_()
        
        self.task_embeddings.zero_()
        self.subtask_embeddings.zero_()
        self.states.zero_()
        self.summary = [ ""  for _ in range(self.batch_size)]
        
        self.internal_grid.reset_all()
        self.environment.reset_all()
    
    def reset_env(self, env_index):
        
        self.grid_gaussian_heading[env_index].zero_()
        self.grid_heading[env_index].zero_()
        self.searching_hinted_target[env_index].zero_()
        self.num_heading_cells[env_index].zero_()
        self.heading_coverage_ratio[env_index].zero_()
        
        self.task_embeddings[env_index].zero_()
        self.subtask_embeddings[env_index].zero_()
        self.states[env_index].zero_()
        self.summary[env_index] = ""
        
        
        self.internal_grid.reset_env(env_index)
        self.environment.reset_env(env_index)
            
