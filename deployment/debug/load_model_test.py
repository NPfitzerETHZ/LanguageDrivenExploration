import torch
import copy

import os
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf

from sentence_transformers import SentenceTransformer
llm = SentenceTransformer('thenlper/gte-large', device="cpu")

from benchmarl.experiment import ExperimentConfig, Experiment
from benchmarl.algorithms import MappoConfig
from benchmarl.models.mlp import MlpConfig
from benchmarl.environments import VmasTask
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scenarios.scripts.grids.general_purpose_occupancy_grid import GeneralPurposeOccupancyGrid, load_task_data  
from scenarios.simple_language_deployment_scenario import MyLanguageScenario
from scenarios.scripts.histories import VelocityHistory, PositionHistory
import copy

from typing import Callable, Optional
from benchmarl.environments import VmasTask
from benchmarl.utils import DEVICE_TYPING
from torchrl.envs import EnvBase, VmasEnv
from datetime import datetime
from tensordict import TensorDict
import csv

def convert_ne_to_xy(north, east):
    """
    Convert coordinates from north–east (NE) ordering to x–y ordering.
    Here, x corresponds to east and y corresponds to north.
    """
    return east, north


def convert_xy_to_ne(x, y):
    """
    Convert coordinates from x–y ordering to north–east (NE) ordering.
    Here, north corresponds to y and east corresponds to x.
    """
    return y, x

def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device: DEVICE_TYPING,
    ) -> Callable[[], EnvBase]:
        config = copy.deepcopy(self.config)
        if self is VmasTask.NAVIGATION: # This is the only modification we make ....
            scenario = MyLanguageScenario() # .... ends here
        else:
            scenario = self.name.lower()
        return lambda: VmasEnv(
            scenario=scenario,
            num_envs=num_envs,
            continuous_actions=continuous_actions,
            seed=seed,
            device=device,
            categorical_actions=True,
            clamp_actions=True,
            **config,
        )

import importlib
def load_class(class_path: str):
    """
    Given a full class path string like 'torch.nn.modules.linear.Linear',
    dynamically load and return the class object.

    Args:
        class_path (str): Full path to the class.

    Returns:
        type: The loaded class.

    Raises:
        ImportError: If the module cannot be imported.
        AttributeError: If the class does not exist in the module.
    """
    try:
        module_path, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Cannot load class '{class_path}': {e}")


def get_experiment(config):
    
    VmasTask.get_env_fun = get_env_fun
    
    experiment_config = ExperimentConfig(**config["experiment_config"].value)
    experiment_config.restore_file = str("/Users/nicolaspfitzer/ProrokLab/CustomScenarios/checkpoints/benchmarl/single_agent_llm_deployment.pt")
    algorithm_config = MappoConfig(**config["algorithm_config"].value)
    model_config = MlpConfig(**config["model_config"].value)
    task = VmasTask.NAVIGATION.get_from_yaml()
    task.config = config["task_config"].value
    
    model_config.activation_class = load_class(model_config.activation_class)
    model_config.layer_class = load_class(model_config.layer_class)
    
    experiment = Experiment(
        config=experiment_config,
        task=task,
        algorithm_config=algorithm_config,
        model_config=model_config,
        seed=0
    )
    
    return experiment

class Agent():
    def __init__(self, robot_id: int, pos_history_length: int, device: DEVICE_TYPING):
        self.robot_id = robot_id
        self.device = device
        self.state_received = False
        # State variables updated by current_state_callback
        self.current_pos_n = 0.0
        self.current_pos_e = 0.0
        self.current_vel_n = 0.0
        self.current_vel_e = 0.0
        
        self.pos_history_length = pos_history_length
        self.pos_dim = 2
        self._create_pos_history()
        
        # # Create publisher and subscriber for the single robot
        # self.pub = self.create_publisher(ReferenceState, f"/robomaster_{self.robot_id}/reference_state", 1)
        # self.create_subscription(CurrentState, f"/robomaster_{self.robot_id}/current_state", self.current_state_callback, 1)

        # # Create reference state message
        # self.reference_state = ReferenceState()
    
    def _create_pos_history(self):  
        self.pos_history = PositionHistory(
            batch_size=1,
            history_length=self.pos_history_length,
            pos_dim=self.pos_dim,
            device=self.device)
    
    def get_current_state(self):
        # This function should return the current state of the agent
        # For now, we will just return a dummy state
        return {
            "position_n": self.current_pos_n,
            "position_e": self.current_pos_e,
            "velocity_n": self.current_vel_n,
            "velocity_e": self.current_vel_e
        }
    

class VmasModels():

    def __init__(self, config: DictConfig, log_dir: Path):
        #super().__init__("vmas_ros_interface")
        ros_config = config["ros_config"]
        self.a_range = ros_config.a_range
        self.v_range = ros_config.v_range
        self.sentence = "Hello"
        self.device = config.device
        
        # Observation Config
        self.observe_pos_history = config["task_config"].value.observe_pos_history
        self.use_gnn = config["task_config"].value.use_gnn
        self.llm_activate = config["task_config"].value.llm_activate
        
        # Create Agents
        self.n_agents = config["task_config"].value.n_agents
        self.observe_pos_history = config["task_config"].value.observe_pos_history
        self.agents = []
        for i in range(self.n_agents):
            agent = Agent(
                robot_id=i,
                pos_history_length=config["task_config"].value.history_length,
                device=config.device
            )
            self.agents.append(agent)
            
        # Grid Config
        grid_config = config["grid_config"]
        self.x_semidim = grid_config.x_semidim
        self.y_semidim = grid_config.y_semidim
        self.num_grid_cells = grid_config.num_grid_cells
        self.mini_grid_radius = grid_config.mini_grid_radius
        self.n_targets = grid_config.n_targets
        self.n_targets_per_class = grid_config.n_targets_per_class
        self.observe_targets = config["task_config"].value.observe_targets
        self.num_covered_targets = torch.zeros(1, dtype=torch.int, device=self.device)
        self._create_occupancy_grid()
        
        # History Config
        self.pos_dim = 2
        self.pos_history_length = config["task_config"].value.history_length
        
        # Load the LLM model
        embedding = torch.tensor(llm.encode([self.sentence]), device=self.device).squeeze(0)
        self.occupancy_grid.embeddings[0] = embedding
        experiment = get_experiment(config)
        
        self.policy = experiment.policy
        
        actor = self.policy[0]
        forward_model = actor.module[0]
        self.mlp = copy.deepcopy(actor.module[0].module[0])
        self.probabilistic_module = actor.module[1]

        # Timer to update and publish commands
        self.dt = 1.0 / 20.0
        self.mytime = 0.0
        #self.timer = self.create_timer(self.dt, self.timer_callback)
        #self.get_logger().info("ROS2 starting ..")

        # Setup CSV logging
                # Setup logging to CSV in runtime dir
        self.log_file = open(log_dir / "vmas_log.csv", "w", newline='')
        self.csv_writer = csv.writer(self.log_file)
        self.csv_writer.writerow(["experiment_time", "real_time", "robot_id", "cmd_vel_n", "cmd_vel_e", 
                                  "pos_n", "pos_e", "vel_n", "vel_e"])
    
    def _create_occupancy_grid(self):
        self.occupancy_grid = GeneralPurposeOccupancyGrid(
            batch_size=1,
            x_dim=2, # [-1,1]
            y_dim=2, # [-1,1]
            num_cells=self.num_grid_cells,
            num_targets=self.n_targets,
            num_targets_per_class=self.n_targets_per_class,
            heading_mini_grid_radius=self.mini_grid_radius*2,
            device=self.device)
        self._covering_range = self.occupancy_grid.cell_radius
    
    def _create_pos_history(self):  
        self.pos_history = PositionHistory(batch_dim=1,history_length=self.pos_history_length,pos_dim=self.pos_dim,device=self.device)
    
    def compute_action(self, agent):
        
        # Get the current state of the agent
        current_state = agent.get_current_state()
        pos_x, pos_y = convert_ne_to_xy(current_state["position_n"], current_state["position_e"])
        vel_x, vel_y = convert_ne_to_xy(current_state["velocity_n"], current_state["velocity_e"])
        pos_x = pos_x / self.x_semidim
        pos_y = pos_y / self.y_semidim
        pos = torch.tensor([pos_x, pos_y], device=self.device).unsqueeze(0)
        vel = torch.tensor([vel_x, vel_y], device=self.device).unsqueeze(0)
        pos_hist = agent.pos_history.get_flattened() if self.observe_pos_history else None
        
        # Collect all observation components
        obs_components = []
        
        # Sentence Embedding Observation
        if self.llm_activate:
            obs_components.append(self.occupancy_grid.observe_embeddings())
            
        # Targets
        if self.observe_targets:
            obs_components.append(self.occupancy_grid.get_grid_target_observation(pos,self.mini_grid_radius))
        
        # Histories
        if self.observe_pos_history:
            obs_components.append(pos_hist[: pos.shape[0], :])
            agent.pos_history.update(pos)
        
        if self.llm_activate:
            obs_components.append(self.num_covered_targets.unsqueeze(1))
            
        # Grid Observation
        obs_components.append(self.occupancy_grid.get_grid_visits_obstacle_observation(pos,self.mini_grid_radius))

        # Pose (GNN works different)    
        if not self.use_gnn:
            obs_components.append(pos)
            obs_components.append(vel)

        # Concatenate observations along last dimension
        obs = torch.cat([comp for comp in obs_components if comp is not None], dim=-1)
        # Update the occupancy grid with the agent's position
        self.occupancy_grid.update(pos)
        
        input_td = TensorDict({
            ("agents", "observation"): obs
        }, batch_size=[1])
        
        output_td = self.policy(input_td)

        # 4. Get the action
        action = output_td[("agents", "action")]
        log_prob = output_td[("agents", "log_prob")]
        
        return action, log_prob

    # def timer_callback(self):
    #     # Run repeatedly to send commands to the robot
    #     if not self.state_received:
    #         self.get_logger().info("Current state not received yet.")
    #         return


    #     # Evaluate the model with the converted inputs and relative target
    #     cmd_vel_tensor = MDEval.compute_action(
    #         pos_x,
    #         pos_y,
    #         vel_x,
    #         vel_y,
    #         target_location=[relative_target_x, relative_target_y],
    #         policy=self.policy,
    #         u_range=self.v_range,
    #         deterministic=True,
    #     )
    #     cmd_vel = cmd_vel_tensor.tolist()

    #     # Convert model output back to north-east ordering.
    #     # Since the model returns (vx, vy) in x-y order, we convert it back:
    #     vel_n, vel_e = convert_xy_to_ne(*cmd_vel)
    #     self.reference_state.vn = vel_n
    #     self.reference_state.ve = vel_e
    #     self.reference_state.yaw = math.pi / 2
    #     self.reference_state.an = self.a_range
    #     self.reference_state.ae = self.a_range
    #     self.reference_state.header.stamp = self.get_clock().now().to_msg()

    #     # Get the real-world time in ISO format
    #     real_time_str = datetime.now().isoformat()

    #     # Log to console
    #     self.get_logger().info(
    #         f"Robot {self.robot_id} - Commanded velocity ne: {vel_n}, {vel_e} - "
    #         f"pos: [{self.current_pos_n}, {self.current_pos_e}] - vel: [{self.current_vel_n}, {self.current_vel_e}]"
    #     )

    #     # Log to CSV file including both simulation time and real-world time
    #     self.csv_writer.writerow([self.mytime, real_time_str, self.robot_id, self.reference_state.vn, self.reference_state.ve,
    #                               self.current_pos_n, self.current_pos_e, self.current_vel_n, self.current_vel_e])
    #     self.log_file.flush()

    #     # Publish the command
    #     self.pub.publish(self.reference_state)

    #     # Update simulation time
    #     self.mytime += self.dt

@hydra.main(config_path="/Users/nicolaspfitzer/ProrokLab/CustomScenarios/configs", 
            config_name="benchmarl_mappo", version_base="1.1")
def main(config: DictConfig):

    print('Config:', OmegaConf.to_yaml(config))

    # Create runtime log dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runtime_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    log_dir = runtime_dir / f"run_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    ros_interface_node = VmasModels(
        config=config,
        log_dir=log_dir
    )
    ros_interface_node.compute_actions()

if __name__ == '__main__':
    main()