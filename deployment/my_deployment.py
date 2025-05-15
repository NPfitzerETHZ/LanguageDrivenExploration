import sys
import csv
import math
import shutil
import signal
from pathlib import Path
from datetime import datetime
from typing import List

import torch
from torchrl.envs.utils import ExplorationType, set_exploration_type
from sentence_transformers import SentenceTransformer
from omegaconf import DictConfig
import hydra

from tensordict import TensorDict
from benchmarl.utils import DEVICE_TYPING

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
from scenarios.centralized.scripts.load_config import load_scenario_config_yaml
from deployment.utils import convert_ne_to_xy, convert_xy_to_ne, get_experiment

X = 0
Y = 1
MAP_NORMAL_SIZE = 2

class State:
    def __init__(self, pos, vel, device):
        self.device = device
        self.pos = torch.tensor(pos, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.vel = torch.tensor(vel, dtype=torch.float32, device=self.device).unsqueeze(0)

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
    deployement_config,
    device: DEVICE_TYPING):
        
        self.node = node
        self.robot_id = robot_id
        self.weight = weight
        
        # Timer to update state
        self.obs_dt = deployement_config.obs_dt
        self.mytime = 0.0
        
        # State buffer
        self.state_buffer = []
        self.state_buffer_length = 5
        
        self.a_range = deployement_config.a_range
        self.v_range = deployement_config.v_range
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
        
        self._create_history(self.node.observe_pos_history,self.node.observe_vel_history)
        
        # Shared grid
        self.grid = grid
        # Shared target count
        self.num_covered_targets = num_covered_targets
        
        # Task config
        self.llm_activate = task_config.llm_activate
        self.mini_grid_radius = task_config.mini_grid_radius
        self.observe_targets = task_config.observe_targets
        
        # Get topic prefix from config or use default
        topic_prefix = getattr(deployement_config, "topic_prefix", "/robomaster_")
        
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
    
    def _create_history(self, observe_pos_history, observe_vel_history): 
         
        if observe_pos_history:
            self.pos_history = PositionHistory(
                batch_size=1,
                history_length=self.pos_history_length,
                pos_dim=self.pos_dim,
                device=self.device)
        if observe_vel_history:
            self.vel_history = VelocityHistory(
                batch_size=1,
                history_length=self.vel_history_length,
                pos_dim=self.vel_dim,
                device=self.device)
    
    def current_state_callback(self, msg: CurrentState):
        # Extract current state values from the state vector
        current_pos_n = msg.state_vector[0]
        current_pos_e = msg.state_vector[1]
        current_vel_n = msg.state_vector[3]
        current_vel_e = msg.state_vector[4]
        
        self.state.pos[0,X], self.state.pos[0,Y] = convert_ne_to_xy(current_pos_n, current_pos_e)
        self.state.vel[0,X], self.state.vel[0,Y] = convert_ne_to_xy(current_vel_n, current_vel_e)
        
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

    def __init__(self, config: DictConfig, log_dir: Path):
        super().__init__("vmas_ros_interface")
        self.device = config.device 
        deployment_config = config["deployment"]
        self.llm = SentenceTransformer(deployment_config.llm_model, device="cpu")
        
        # Grid Config
        grid_config = config["grid_config"]
        self.x_semidim = grid_config.x_semidim
        self.y_semidim = grid_config.y_semidim
        
        # Task Config
        load_scenario_config_yaml(config,self)
        self._create_occupancy_grid()
        
        # History Config
        self.pos_dim = 2
        self.pos_history_length = config["task_config"].value.history_length

        # Load experiment and get policy
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
        id_list = deployment_config.id_list
        assert len(id_list) == self.n_agents
        for i in range(self.n_agents):
            agent = Agent(
                node=self,
                robot_id=id_list[i],
                weight=self.agent_weight,
                pos_history_length=self.pos_history_length,
                grid=self.occupancy_grid,
                num_covered_targets=self.num_covered_targets,
                task_config = config["task_config"].value,
                deployment_config = deployment_config,
                device=config.device
            )
            self.agents.append(agent)
        
        # Create action loop
        self.action_dt = deployment_config.action_dt
        self.max_steps = deployment_config.max_steps
        self.step_count = 0
        
        self.get_logger().info("ROS2 starting ..")
        
    def _create_occupancy_grid(self):
        
        self.occupancy_grid = WorldOccupancyGrid(
            batch_size=1,
            x_dim=MAP_NORMAL_SIZE, # [-1,1]
            y_dim=MAP_NORMAL_SIZE, # [-1,1]
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
        if not self._all_states_received():
            self.get_logger().info("Waiting for all agents to receive state.")
            return

        if self._reached_max_steps():
            self._handle_termination()
            return

        obs_list, pos_list, vel_list = self._collect_observations()
        if not obs_list:
            self.get_logger().warn("No valid observations collected. Skipping this timestep.")
            return

        input_td = self._prepare_input_tensor(obs_list, pos_list, vel_list)

        with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
            output_td = self.policy(input_td)

        action = output_td[("agents", "action")]

        self._issue_commands_to_agents(action)
        self.step_count += 1

    def _all_states_received(self):
        return all(agent.state_received for agent in self.agents)

    def _reached_max_steps(self):
        return self.step_count >= self.max_steps

    def _handle_termination(self):
        self.get_logger().info(f"Robots reached {self.max_steps} steps. Stopping.")
        self.timer.cancel()
        self.stop_all_agents()
        self.occupancy_grid.reset_all()
        self.prompt_for_new_instruction()

    def _collect_observations(self):
        obs_list, pos_list, vel_list = [], [], []

        for agent in self.agents:
            if not agent.state_buffer:
                self.get_logger().warn(f"No state in buffer for agent {agent.robot_id}")
                continue

            latest_state = agent.state_buffer.pop()
            if self.use_gnn:
                obs_list.append(latest_state["obs"].float())
                pos_list.append(latest_state["pos"].float())
                vel_list.append(latest_state["vel"].float())
            else:
                obs_list.append(latest_state.float())

        return obs_list, pos_list, vel_list

    def _prepare_input_tensor(self, obs_list, pos_list, vel_list):
        obs_tensor = torch.cat(obs_list, dim=0)

        if self.use_gnn:
            return TensorDict({
                ("agents", "observation", "obs"): obs_tensor,
                ("agents", "observation", "pos"): torch.cat(pos_list, dim=0),
                ("agents", "observation", "vel"): torch.cat(vel_list, dim=0),
            }, batch_size=[len(obs_list)])
        else:
            return TensorDict({
                ("agents", "observation"): obs_tensor,
            }, batch_size=[len(obs_list)])

    def _issue_commands_to_agents(self, action_tensor):
        real_time_str = datetime.now().isoformat()

        for i, agent in enumerate(self.agents):
            cmd_vel = action_tensor[i].tolist()
            vel_n, vel_e = convert_xy_to_ne(*cmd_vel)

            agent.reference_state.vn = vel_n
            agent.reference_state.ve = vel_e
            agent.reference_state.yaw = math.pi / 2
            agent.reference_state.an = agent.a_range
            agent.reference_state.ae = agent.a_range
            agent.reference_state.header.stamp = self.get_clock().now().to_msg()

            self.get_logger().info(
                f"Robot {agent.robot_id} - Commanded velocity ne: {vel_n}, {vel_e} - "
                f"pos: [{agent.state.pos[X]}, {agent.state.pos[Y]}] - "
                f"vel: [{agent.state.vel[X]}, {agent.state.vel[Y]}]"
            )

            self.csv_writer.writerow([
                agent.mytime, real_time_str, agent.robot_id,
                agent.reference_state.vn, agent.reference_state.ve,
                agent.state.pos[X], agent.state.pos[Y],
                agent.state.vel[X], agent.state.vel[Y]
            ])
            self.log_file.flush()

            agent.pub.publish(agent.reference_state)
            agent.mytime += self.action_dt

    def stop_all_agents(self):
        for agent in self.agents:
            agent.state_buffer = [] 
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

        # Reset step count and timers
        self.step_count = 0
        self.timer = self.create_timer(self.action_dt, self.timer_callback)
        for agent in self.agents:
            agent.mytime = 0
            agent.timer = self.create_timer(agent.obs_dt, agent.collect_observation)
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

    ros_interface_node = VmasModelsROSInterface(
        config=config,
        log_dir=log_dir
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
