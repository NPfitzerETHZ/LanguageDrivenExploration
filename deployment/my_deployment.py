import sys
import os
import csv
import math
import copy
import shutil
import signal
import importlib
from pathlib import Path
from datetime import datetime
from typing import List, Callable, Optional

import torch
from torchrl.envs import EnvBase, VmasEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type
from sentence_transformers import SentenceTransformer
from omegaconf import DictConfig, OmegaConf
import hydra

from tensordict import TensorDict

# ROS2
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, UInt8
sys.path.insert(0, "/home/npfitzer/robomaster_ws/install/freyja_msgs/lib/python3.10/site-packages")
from freyja_msgs.msg import ReferenceState
from freyja_msgs.msg import CurrentState
from freyja_msgs.msg import WaypointTarget

# Local Modules
sys.path.insert(0, "/home/npfitzer/robomaster_ws/src/LanguageDrivenExploration")
from scenarios.grids.world_occupancy_grid import WorldOccupancyGrid
from scenarios.centralized.multi_agent_llm_exploration import MyLanguageScenario
from scenarios.centralized.scripts.histories import VelocityHistory, PositionHistory
from scenarios.centralized.scripts.observation import observation
from benchmarl.environments import VmasTask
from benchmarl.experiment import ExperimentConfig, Experiment
from benchmarl.algorithms import MappoConfig
from benchmarl.models import GnnConfig, SequenceModelConfig
from benchmarl.models.mlp import MlpConfig
from benchmarl.utils import DEVICE_TYPING

X = 0
Y = 1


def convert_ne_to_xy(north: float, east: float) -> tuple[float, float]:
    """
    Convert coordinates from north–east (NE) ordering to x–y ordering.
    Here, x corresponds to east and y corresponds to north.
    """
    return east, north


def convert_xy_to_ne(x: float, y: float) -> tuple[float, float]:
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
        # Override scenario for NAVIGATION task to use custom language-based scenario
        if self is VmasTask.NAVIGATION:
            scenario = MyLanguageScenario()
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

def load_class(class_path: str):
    """
    Given a full class path string like 'torch.nn.modules.linear.Linear',
    dynamically load and return the class object.
    """
    try:
        module_path, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Cannot load class '{class_path}': {e}")

def get_experiment(config: DictConfig) -> Experiment:
    
    VmasTask.get_env_fun = get_env_fun
    Experiment._load_experiment = _load_experiment_cpu
    OmegaConf.set_struct(config["model_config"].value.model_configs[0].gnn_kwargs, False)
    
    print(config["experiment_config"].value)
    
    experiment_config = ExperimentConfig(**config["experiment_config"].value)
    experiment_config.restore_file = str("/home/npfitzer/robomaster_ws/src/LanguageDrivenExploration/checkpoints/benchmarl/gnn_multi_agent_first/gnn_multi_agent_llm_deployment.pt")
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

class State:
    def __init__(self, pos, vel, device):
        self.device = device
        self.pos = torch.tensor(pos, dtype=torch.float32, device=self.device)
        self.vel = torch.tensor(vel, dtype=torch.float32, device=self.device)

class Agent:
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
        
        # Timer to update state
        self.obs_dt = 1.0 / 50.0
        self.mytime = 0.0
        
        # State buffer
        self.state_buffer = []
        self.state_buffer_length = 5
        
        self.a_range = ros_config.a_range
        self.v_range = ros_config.v_range
        self.device = device
        self.state_received = False
        # State variables updated by current_state_callback
        
        self.state = State(
            pos=[0.0, 0.0],
            vel=[0.0, 0.0],
            device=self.device
        )
        
        self.pos_history_length = pos_history_length
        self.pos_dim = 2
        
        if self.node.observe_pos_history:
            self._create_pos_history()
        
        self.timer = self.node.create_timer(self.obs_dt, self.collect_observation)
        
        # Shared grid
        self.grid = grid
        # Shared target count
        self.num_covered_targets = num_covered_targets
        
        # Task config
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
        current_pos_n = msg.state_vector[0]
        current_pos_e = msg.state_vector[1]
        current_vel_n = msg.state_vector[3]
        current_vel_e = msg.state_vector[4]
        
        self.state.pos[X], self.state.pos[Y] = convert_ne_to_xy(current_pos_n, current_pos_e)
        self.state.vel[X], self.state.vel[Y] = convert_ne_to_xy(current_vel_n, current_vel_e)
        
        self.state_received = True
    
    def collect_observation(self):
        
        obs = observation(self, self.node)
        self.state_buffer.append(obs)
        if len(self.state_buffer) > self.state_buffer_length:
            self.state_buffer = self.state_buffer[-self.state_buffer_length:]
        self.grid.update(self.state.pos)
        
    def send_zero_velocity(self):
        # Send a zero velocity command
        self.reference_state.vn = 0.0
        self.reference_state.ve = 0.0
        self.reference_state.header.stamp = self.node.get_clock().now().to_msg()
        self.node.get_logger().info(f"Robot {self.robot_id} - Zero velocity command sent.")
        self.pub.publish(self.reference_state)
        self.node.log_file.flush()
    

class VmasModelsROSInterface(Node):

    def __init__(self, config: DictConfig, log_dir: Path, llm: SentenceTransformer):
        super().__init__("vmas_ros_interface")
        self.sentence = "Team, locate the target in the south east corner of the room."
        self.device = config.device
        self.llm = llm

        self.agents_done = set()
        
        # Grid Config
        grid_config = config["grid_config"]
        self.x_semidim = grid_config.x_semidim
        self.y_semidim = grid_config.y_semidim
        
        # Task Config
        self.embedding_size = config["task_config"].value.embedding_size
        self.use_gnn = config["task_config"].value.use_gnn
        self.use_lidar = False
        self.num_grid_cells = config["task_config"].value.num_grid_cells
        self.mini_grid_radius = config["task_config"].value.mini_grid_radius
        self.n_target_classes = config["task_config"].value.n_target_classes
        self.n_targets_per_class = config["task_config"].value.n_targets_per_class
        self.observe_targets = config["task_config"].value.observe_targets
        self.agent_weight = config["task_config"].value.agent_weight
        self.grid_visit_threshold = config["task_config"].value.grid_visit_threshold
        self.use_expo_search_rew = config["task_config"].value.use_expo_search_rew
        self.num_covered_targets = torch.zeros(1, dtype=torch.int, device=self.device)
        self.n_agents = config["task_config"].value.n_agents
        self.observe_pos_history = config["task_config"].value.observe_pos_history
        self.observe_vel_history = False
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
        self.agents: List[Agent] = []
        id_list = [3,4,6,7]
        for i in range(self.n_agents):
            agent = Agent(
                node=self,
                robot_id=id_list[i],
                weight=self.agent_weight,
                pos_history_length=self.pos_history_length,
                grid=self.occupancy_grid,
                num_covered_targets=self.num_covered_targets,
                task_config = config["task_config"].value,
                ros_config = config["ros_config"],
                device=config.device
            )
            self.agents.append(agent)
        
        # Create action loop
        self.action_dt = 1.0 / 10.0
        self.step_count = 0
        self.max_steps = 200
        self.timer = self.create_timer(self.action_dt, self.timer_callback)
        
        self.get_logger().info("ROS2 starting ..")
        
    def _create_occupancy_grid(self):
        self.occupancy_grid = WorldOccupancyGrid(
            batch_size=1,
            x_dim=2, # [-1,1]
            y_dim=2, # [-1,1]
            x_scale=self.x_semidim,
            y_scale=self.y_semidim,
            num_cells=self.num_grid_cells,
            num_targets=self.n_target_classes * self.n_targets_per_class,
            num_targets_per_class=self.n_targets_per_class,
            visit_threshold=self.grid_visit_threshold,
            embedding_size=self.embedding_size,
            device=self.device)
        self._covering_range = self.occupancy_grid.cell_radius
    
    def timer_callback(self):
        # Run repeatedly to send commands to the robot
        if not all(agent.state_received for agent in self.agents):
            self.get_logger().info("Waiting for all agents to receive state.")
            return
        
        if self.step_count >= self.max_steps:
            self.get_logger().info(f"Robots reached {self.max_steps} steps. Stopping.")
            self.timer.cancel()
            self.stop_all_agents()
            self.prompt_for_new_instruction()
            return
        
        # Collect Agent observations
        obs_list = []
        pos_list = []
        vel_list = []

        for agent in self.agents:
            if not agent.state_buffer:
                self.get_logger().warn(f"No state in buffer for agent {agent.robot_id}")
                continue

            latest_state = agent.state_buffer.pop()
            if self.use_gnn:
                obs = latest_state["obs"].to(torch.float32)
                pos = latest_state["pos"].to(torch.float32)
                vel = latest_state["vel"].to(torch.float32)

                obs_list.append(obs)
                pos_list.append(pos)
                vel_list.append(vel)        
            else:
                obs_list.append(latest_state.to(torch.float32))
        
        if not obs_list:
            self.get_logger().warn("No valid observations collected. Skipping this timestep.")
            return

        obs_tensor = torch.cat(obs_list, dim=0)

        if self.use_gnn:
            positions = torch.cat(pos_list, dim=0)
            velocities = torch.cat(vel_list, dim=0)
            input_td = TensorDict({
                ("agents", "observation", "obs"): obs_tensor,
                ("agents", "observation", "pos"): positions,
                ("agents", "observation", "vel"): velocities,
            }, batch_size=[len(obs_list)])
        else:
            input_td = TensorDict({
                ("agents", "observation"): obs_tensor,
            }, batch_size=[len(obs_list)])
        
        with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
                output_td = self.policy(input_td)

        action = output_td[("agents", "action")]
        log_prob = output_td[("agents", "log_prob")]

        for i, agent in enumerate(self.agents):
            
            cmd_vel = action[i,:]
            cmd_vel = cmd_vel.tolist()

            # Convert model output back to north-east ordering.
            # Since the model returns (vx, vy) in x-y order, we convert it back:
            vel_n, vel_e = convert_xy_to_ne(*cmd_vel)
            #self.reference_state.pn = 0
            #self.reference_state.pe = 0
            agent.reference_state.vn = vel_n
            agent.reference_state.ve = vel_e
            agent.reference_state.yaw = math.pi / 2
            agent.reference_state.an = agent.a_range
            agent.reference_state.ae = agent.a_range
            agent.reference_state.header.stamp = self.get_clock().now().to_msg()

            # Get the real-world time in ISO format
            real_time_str = datetime.now().isoformat()

            # Log to console
            self.get_logger().info(
                f"Robot {agent.robot_id} - Commanded velocity ne: {vel_n}, {vel_e} - "
                f"pos: [{agent.state.pos[X]}, {agent.state.pos[Y]}] - vel: [{agent.state.vel[X]}, {agent.state.vel[Y]}]"
            )

            # Log to CSV file including both simulation time and real-world time
            self.csv_writer.writerow([agent.mytime, real_time_str, agent.robot_id, agent.reference_state.vn, agent.reference_state.ve,
                                    agent.state.pos[X],agent.state.pos[Y], agent.state.vel[X], agent.state.vel[Y]])
            self.log_file.flush()

            # Publish the command
            agent.pub.publish(agent.reference_state)

        # Update simulation time
        agent.mytime += self.action_dt
        self.step_count += 1
    
    def stop_all_agents(self):
        for agent in self.agents:
            agent.timer.cancel()
            agent.send_zero_velocity()
    
    def prompt_for_new_instruction(self):
        new_sentence = input("Enter a new instruction for the agents: ")
        try:
            embedding = torch.tensor(self.llm.encode([new_sentence]), device=self.device).squeeze(0)
        except Exception as e:
            self.get_logger().error(f"Failed to encode instruction: {e}")
            return
        self.occupancy_grid.embeddings[0] = embedding

        # Reset step count and restart timers
        self.step_count = 0
        for agent in self.agents:
            agent.mytime = 0
            agent.timer.reset()  # or create a new timer if needed
        self.agents_done.clear()
        self.get_logger().info("Starting agents with new instruction.")



@hydra.main(config_path="/home/npfitzer/robomaster_ws/src/LanguageDrivenExploration/checkpoints/benchmarl/gnn_multi_agent_first/", 
            config_name="benchmarl_mappo", version_base="1.1")
def main(config: DictConfig):
    rclpy.init()

    # Create runtime log dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runtime_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    log_dir = runtime_dir / f"run_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Copy all .py files and config to runtime dir
    for src in Path('./src/LanguageDrivenExploration/deployment').rglob("*.py"):
        dst = log_dir / src.name
        shutil.copy2(src, dst)
    for src in Path("./src/LanguageDrivenExploration/checkpoints/benchmarl/gnn_multi_agent/first").rglob("*.yaml"):
        dst = log_dir / src.name
        shutil.copy2(src, dst)
        
    llm = SentenceTransformer("thenlper/gte-large", device="cpu")

    ros_interface_node = VmasModelsROSInterface(
        config=config,
        log_dir=log_dir,
        llm=llm
    )
    ros_interface_node.prompt_for_new_instruction()

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
