import rclpy
from rclpy.node import Node

import sys
import math
import signal
import csv
from datetime import datetime  # For real-world time logging

from std_msgs.msg import String
from std_msgs.msg import UInt8
sys.path.insert(0, "/home/npfitzer/robomaster_ws/install/freyja_msgs/lib/python3.10/site-packages")
from freyja_msgs.msg import ReferenceState
from freyja_msgs.msg import CurrentState
from freyja_msgs.msg import WaypointTarget

from tensordict import TensorDict

from benchmarl.experiment import ExperimentConfig, Experiment
from benchmarl.algorithms import MappoConfig
from benchmarl.models.mlp import MlpConfig
from benchmarl.models import GnnConfig, SequenceModelConfig
from benchmarl.environments import VmasTask

import torch
import copy

import shutil
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf

from sentence_transformers import SentenceTransformer
llm = SentenceTransformer('thenlper/gte-large', device="cpu")

from typing import Callable, Optional
from benchmarl.environments import VmasTask
from benchmarl.utils import DEVICE_TYPING
from torchrl.envs import EnvBase, VmasEnv  
import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 
from scenarios.grids.world_occupancy_grid import WorldOccupancyGrid
from scenarios.centralized.multi_agent_llm_exploration import MyLanguageScenario
from scenarios.centralized.scripts.histories import VelocityHistory, PositionHistory
from torchrl.envs.utils import ExplorationType, set_exploration_type
import copy


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

def _load_experiment_cpu(self):
    loaded_dict = torch.load(self.config.restore_file, map_location=torch.device("cpu"))
    self.load_state_dict(loaded_dict)
    return self

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
    Experiment._load_experiment = _load_experiment_cpu
    OmegaConf.set_struct(config["model_config"].value.model_configs[0].gnn_kwargs, False)
    
    print(config["experiment_config"].value)
    
    experiment_config = ExperimentConfig(**config["experiment_config"].value)
    experiment_config.restore_file = str("/Users/nicolaspfitzer/ProrokLab/CustomScenarios/checkpoints/benchmarl/gnn_multi_agent_first/gnn_multi_agent_llm_deployment.pt")
    task = VmasTask.NAVIGATION.get_from_yaml()
    task.config = config["task_config"].value
    algorithm_config = MappoConfig(**config["algorithm_config"].value)
    
    use_gnn = config["task_config"].value.use_gnn
    
    if not use_gnn: 
        model_config = MlpConfig(**config["model_config"].value)
        model_config.activation_class = load_class(config["model_config"].value.activation_class)
        model_config.layer_class = load_class(config["model_config"].value.layer_class)
    else:
        gnn_cfg = config["model_config"].value.model_configs[0]
        mlp_cfg = config["model_config"].value.model_configs[1]
        gnn_config = GnnConfig(**gnn_cfg)
        gnn_config.gnn_class = load_class(gnn_cfg.gnn_class)
        # We add an MLP layer to process GNN output node embeddings into actions
        mlp_config = MlpConfig(**mlp_cfg)
        mlp_config.activation_class = load_class(mlp_cfg.activation_class)
        mlp_config.layer_class = load_class(mlp_cfg.layer_class)
        model_config = SequenceModelConfig(model_configs=[gnn_config, mlp_config], intermediate_sizes=config["model_config"].value.intermediate_sizes)
    
    experiment = Experiment(
        config=experiment_config,
        task=task,
        algorithm_config=algorithm_config,
        model_config=model_config,
        seed=0
    )
    
    return experiment

class Agent():
    def __init__(
    self,
    node,
    robot_id: int,
    weight: float,
    pos_history_length: int,
    grid: WorldOccupancyGrid,
    num_covered_targets: torch.Tensor,
    task_config,
    ros_config,
    device: DEVICE_TYPING):
        
        self.node = node
        self.robot_id = robot_id
        self.weight = weight
        
        # Timer to update and publish commands
        self.dt = 1.0 / 20.0
        self.mytime = 0.0
        self.step_count = 0
        self.max_steps = 200  # Change as needed
        
        # State buffer
        self.state_buffer = []
        
        self.a_range = ros_config.a_range
        self.v_range = ros_config.v_range
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
        
        self.timer = self.node.create_timer(self.dt, self.timer_callback)
        
        # Shared grid
        self.grid = grid
        # Shared target count
        self.num_covered_targets = num_covered_targets
        
        # Task config
        self.use_gnn = task_config.use_gnn
        self.observe_pos_history = task_config.observe_pos_history
        self.llm_activate = task_config.llm_activate
        self.mini_grid_radius = task_config.mini_grid_radius
        self.observe_targets = task_config.observe_targets
        
        # Get topic prefix from config or use default
        topic_prefix = getattr(ros_config, "topic_prefix", "/robomaster_")
        
        # Create publisher for the robot
        self.pub = self.node.create_publisher(
            ReferenceState, 
            f"{topic_prefix}{self.robot_id}/reference_state", 
            1  
        )
        
        # Create subscription with more descriptive variable name
        self.state_subscription = self.node.create_subscription(
            CurrentState, 
            f"{topic_prefix}{self.robot_id}/current_state", 
            self.current_state_callback, 
            1
        )
        
        # Log the subscription
        self.node.get_logger().info(f"Robot {self.robot_id} subscribing to: {topic_prefix}{self.robot_id}/current_state")
    
        # Create reference state message
        self.reference_state = ReferenceState()
    
    def _create_pos_history(self):  
        self.pos_history = PositionHistory(
            batch_size=1,
            history_length=self.pos_history_length,
            pos_dim=self.pos_dim,
            device=self.device)
    
    def current_state_callback(self, msg: CurrentState):
        # Extract current state values from the state vector
        self.current_pos_n = msg.state_vector[0]
        self.current_pos_e = msg.state_vector[1]
        self.current_vel_n = msg.state_vector[3]
        self.current_vel_e = msg.state_vector[4]
        self.state_received = True
    
    def get_current_state(self):
        # This function should return the current state of the agent
        # For now, we will just return a dummy state
        return {
            "position_n": self.current_pos_n,
            "position_e": self.current_pos_e,
            "velocity_n": self.current_vel_n,
            "velocity_e": self.current_vel_e
        }
    
    def collect_observation(self):
        
        # Get the current state of the agent
        current_state = self.get_current_state()
        pos_x, pos_y = convert_ne_to_xy(current_state["position_n"], current_state["position_e"])
        vel_x, vel_y = convert_ne_to_xy(current_state["velocity_n"], current_state["velocity_e"])
        pos_x = pos_x / self.node.x_semidim
        pos_y = pos_y / self.node.y_semidim
        vel_x = vel_x / (2 * self.node.x_semidim)
        vel_y = vel_y / (2 * self.node.y_semidim)

        pos = torch.tensor([pos_x, pos_y], device=self.device).unsqueeze(0)
        vel = torch.tensor([vel_x, vel_y], device=self.device).unsqueeze(0)
        pos_hist = self.pos_history.get_flattened() if self.observe_pos_history else None
        
        obs_components = []

        # Sentence embedding (not logged)
        if self.llm_activate:
            obs_components.append(self.grid.observe_embeddings())  # excluded from logging

        # Targets
        if self.observe_targets:
            target_obs = self.grid.get_grid_target_observation(pos, self.mini_grid_radius)
            self.node.get_logger().info(f"Target observation: {target_obs.cpu().numpy()}")
            obs_components.append(target_obs)

        # Histories
        if self.observe_pos_history:
            hist_obs = pos_hist[: pos.shape[0], :]
            self.node.get_logger().info(f"Position history: {hist_obs.cpu().numpy()}")
            obs_components.append(hist_obs)
            self.pos_history.update(pos)

        # Grid Observation
        grid_obs = self.grid.get_grid_visits_obstacle_observation(pos, self.mini_grid_radius)
        self.node.get_logger().info(f"Grid visits/obstacles observation: {grid_obs.cpu().numpy()}")
        obs_components.append(grid_obs)

        # Number of covered targets (if LLM active)
        if self.llm_activate:
            obs_components.append(self.num_covered_targets.unsqueeze(1))  # not logged

        # Update State Buffer and Grid
        obs = torch.cat([comp for comp in obs_components if comp is not None], dim=-1)
        self.state_buffer.append({"obs": obs, "pos": pos, "vel":vel})
        # Keep only the most recent `self.state_buffer_length` entries
        if len(self.state_buffer) > self.state_buffer_length:
            self.state_buffer = self.state_buffer[-self.state_buffer_length:]
        self.grid.update(pos)
        
    
    def compute_action(self):
        # Get the current state of the agent
        current_state = self.get_current_state()
        pos_x, pos_y = convert_ne_to_xy(current_state["position_n"], current_state["position_e"])
        vel_x, vel_y = convert_ne_to_xy(current_state["velocity_n"], current_state["velocity_e"])
        pos_x = pos_x / self.node.x_semidim
        pos_y = pos_y / self.node.y_semidim
        vel_x = vel_x / (2 * self.node.x_semidim)
        vel_y = vel_y / (2 * self.node.y_semidim)

        pos = torch.tensor([pos_x, pos_y], device=self.device).unsqueeze(0)
        vel = torch.tensor([vel_x, vel_y], device=self.device).unsqueeze(0)
        pos_hist = self.pos_history.get_flattened() if self.observe_pos_history else None

        obs_components = []

        # Sentence embedding (not logged)
        if self.llm_activate:
            obs_components.append(self.grid.observe_embeddings())  # excluded from logging

        # Targets
        if self.observe_targets:
            target_obs = self.grid.get_grid_target_observation(pos, self.mini_grid_radius)
            self.node.get_logger().info(f"Target observation: {target_obs.cpu().numpy()}")
            obs_components.append(target_obs)

        # Histories
        if self.observe_pos_history:
            hist_obs = pos_hist[: pos.shape[0], :]
            self.node.get_logger().info(f"Position history: {hist_obs.cpu().numpy()}")
            obs_components.append(hist_obs)
            self.pos_history.update(pos)

        # Grid Observation
        grid_obs = self.grid.get_grid_visits_obstacle_observation(pos, self.mini_grid_radius)
        self.node.get_logger().info(f"Grid visits/obstacles observation: {grid_obs.cpu().numpy()}")
        obs_components.append(grid_obs)

        # Number of covered targets (if LLM active)
        if self.llm_activate:
            obs_components.append(self.num_covered_targets.unsqueeze(1))  # not logged

        # Pose
        if not self.use_gnn:
            self.node.get_logger().info(f"Normalized position: {pos.cpu().numpy()}")
            self.node.get_logger().info(f"Normalized velocity: {vel.cpu().numpy()}")
            obs_components.append(pos)
            obs_components.append(vel)

        obs = torch.cat([comp for comp in obs_components if comp is not None], dim=-1)

        self.grid.update(pos)

        input_td = TensorDict({
            ("agents", "observation"): obs.to(dtype=torch.float32, device=self.device)
        }, batch_size=[1], device=self.device)

        deterministic = True
        exp_type = (
            ExplorationType.DETERMINISTIC if deterministic
            else ExplorationType.RANDOM
        )
        
        #    and run the policy
        with torch.no_grad(), set_exploration_type(exp_type):
            output_td = self.node.policy(input_td)

        action = output_td[("agents", "action")]
        log_prob = output_td[("agents", "log_prob")]

        return action, log_prob

    
    def timer_callback(self):
        # Run repeatedly to send commands to the robot
        if not self.state_received:
            self.node.get_logger().info("Current state not received yet.")
            return
        
        if self.step_count >= self.max_steps:
            self.node.get_logger().info(f"Robot {self.robot_id} reached {self.max_steps} steps. Stopping.")
            self.timer.cancel()
            self.send_zero_velocity()
            self.node.notify_agent_stopped(self.robot_id)
            return

        # Convert current state to model input (x-y ordering)
        pos_x, pos_y = convert_ne_to_xy(self.current_pos_n, self.current_pos_e)
        vel_x, vel_y = convert_ne_to_xy(self.current_vel_n, self.current_vel_e)

        cmd_vel, log_prob = self.compute_action()
        
        cmd_vel = cmd_vel.tolist()

        # Convert model output back to north-east ordering.
        # Since the model returns (vx, vy) in x-y order, we convert it back:
        vel_n, vel_e = convert_xy_to_ne(*cmd_vel[0])
        #self.reference_state.pn = 0
        #self.reference_state.pe = 0
        self.reference_state.vn = vel_n
        self.reference_state.ve = vel_e
        self.reference_state.yaw = math.pi / 2
        self.reference_state.an = self.a_range
        self.reference_state.ae = self.a_range
        self.reference_state.header.stamp = self.node.get_clock().now().to_msg()

        # Get the real-world time in ISO format
        real_time_str = datetime.now().isoformat()

        # Log to console
        self.node.get_logger().info(
            f"Robot {self.robot_id} - Commanded velocity ne: {vel_n}, {vel_e} - "
            f"pos: [{self.current_pos_n}, {self.current_pos_e}] - vel: [{self.current_vel_n}, {self.current_vel_e}]"
        )

        # Log to CSV file including both simulation time and real-world time
        self.node.csv_writer.writerow([self.mytime, real_time_str, self.robot_id, self.reference_state.vn, self.reference_state.ve,
                                  self.current_pos_n, self.current_pos_e, self.current_vel_n, self.current_vel_e])
        self.node.log_file.flush()

        # Publish the command
        self.pub.publish(self.reference_state)

        # Update simulation time
        self.mytime += self.dt
        self.step_count += 1


    def send_zero_velocity(self):
        # Send a zero velocity command
        self.reference_state.vn = 0.0
        self.reference_state.ve = 0.0
        self.reference_state.header.stamp = self.node.get_clock().now().to_msg()
        self.node.get_logger().info(f"Robot {self.robot_id} - Zero velocity command sent.")
        self.pub.publish(self.reference_state)
        self.node.log_file.flush()
    

class VmasModelsROSInterface(Node):

    def __init__(self, config: DictConfig, log_dir: Path):
        super().__init__("vmas_ros_interface")
        self.sentence = "Team, locate the target in the south east corner of the room."
        self.device = config.device

        self.agents_done = set()
        
        # Grid Config
        grid_config = config["grid_config"]
        self.x_semidim = grid_config.x_semidim
        self.y_semidim = grid_config.y_semidim

        self.task_x_semidim = config["task_config"].value.x_semidim
        self.task_y_semidim = config["task_config"].value.y_semidim
        self.embedding_size = config["task_config"].value.embedding_size

        self.num_grid_cells = config["task_config"].value.num_grid_cells
        self.mini_grid_radius = config["task_config"].value.mini_grid_radius
        self.n_target_classes = config["task_config"].value.n_target_classes
        self.n_targets_per_class = config["task_config"].value.n_targets_per_class
        self.observe_targets = config["task_config"].value.observe_targets
        self.agent_weight = config["task_config"].value.agent_weight
        self.grid_visit_threshold = config["task_config"].value.grid_visit_threshold
        self.num_covered_targets = torch.zeros(1, dtype=torch.int, device=self.device)
        self._create_occupancy_grid()
        
        # History Config
        self.pos_dim = 2
        self.pos_history_length = config["task_config"].value.history_length

        # Load experiment
        experiment = get_experiment(config)
        self.policy = experiment.policy

        # Setup CSV logging
                # Setup logging to CSV in runtime dir
        self.log_file = open(log_dir / "vmas_log.csv", "w", newline='')
        self.csv_writer = csv.writer(self.log_file)
        self.csv_writer.writerow(["experiment_time", "real_time", "robot_id", "cmd_vel_n", "cmd_vel_e", 
                                  "pos_n", "pos_e", "vel_n", "vel_e"])

        # Create Agents
        self.n_agents = config["task_config"].value.n_agents
        self.observe_pos_history = config["task_config"].value.observe_pos_history
        self.agents = []
        for i in range(self.n_agents):
            agent = Agent(
                node=self,
                robot_id=6,
                weight=self.agent_weight,
                pos_history_length=self.pos_history_length,
                grid=self.occupancy_grid,
                num_covered_targets=self.num_covered_targets,
                task_config = config["task_config"].value,
                ros_config = config["ros_config"],
                device=config.device
            )
            self.agents.append(agent)
        
        self.prompt_for_new_instruction()
        self.get_logger().info("ROS2 starting ..")
        
    def _create_occupancy_grid(self):
        self.occupancy_grid = WorldOccupancyGrid(
            batch_size=1,
            x_dim=self.task_x_semidim*2, # [-1,1]
            y_dim=self.task_y_semidim*2, # [-1,1]
            num_cells=self.num_grid_cells,
            num_targets=self.n_target_classes * self.n_targets_per_class,
            num_targets_per_class=self.n_targets_per_class,
            visit_threshold=self.grid_visit_threshold,
            embedding_size=self.embedding_size,
            device=self.device)
        self._covering_range = self.occupancy_grid.cell_radius
    
    def stop_all_agents(self):
        for agent in self.agents:
            agent.timer.cancel()
            agent.send_zero_velocity()
    
    def notify_agent_stopped(self, robot_id):
        self.agents_done.add(robot_id)
        if len(self.agents_done) == self.n_agents:
            self.get_logger().info("All agents stopped after 200 steps.")
            self.prompt_for_new_instruction()
    
    def prompt_for_new_instruction(self):
        new_sentence = input("Enter a new instruction for the agents: ")
        embedding = torch.tensor(llm.encode([new_sentence]), device=self.device).squeeze(0)
        self.occupancy_grid.embeddings[0] = embedding

        # Reset agent step counts and restart timers
        for agent in self.agents:
            agent.step_count = 0
            agent.timer.reset()  # or create a new timer if needed
        self.agents_done.clear()
        self.get_logger().info("Starting agents with new instruction.")



@hydra.main(config_path="/home/npfitzer/robomaster_ws/src/LanguageDrivenExploration/checkpoints/benchmarl/single_agent_first/", 
            config_name="benchmarl_mappo", version_base="1.1")
def main(config: DictConfig):
    rclpy.init()

    # Create runtime log dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runtime_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    log_dir = runtime_dir / f"run_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Copy all .py files and config to runtime dir
    for src in Path('./src/RoboMasterROSPackage/andy_controller').rglob("*.py"):
        dst = log_dir / src.name
        shutil.copy2(src, dst)
    for src in Path("./config").rglob("*.yaml"):
        dst = log_dir / src.name
        shutil.copy2(src, dst)

    ros_interface_node = VmasModelsROSInterface(
        config=config,
        log_dir=log_dir
    )

    def sigint_handler(sig, frame):
        ros_interface_node.get_logger().info('SIGINT received. Stopping timer and sending zero velocity...')
        
        ros_interface_node.stop_all_agents()
        
        # Spin once to publish the final message
        rclpy.spin_once(ros_interface_node, timeout_sec=0.5)

        # Clean up
        ros_interface_node.destroy_node()
        ros_interface_node.log_file.close()
        rclpy.shutdown()
        sys.exit(0)


    signal.signal(signal.SIGINT, sigint_handler)

    rclpy.spin(ros_interface_node)


if __name__ == '__main__':
    main()
