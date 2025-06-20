import sys
import csv
import math
import shutil
import signal
from pathlib import Path
from datetime import datetime
from typing import List, Optional

import torch
from torchrl.envs.utils import ExplorationType, set_exploration_type
from omegaconf import DictConfig, OmegaConf
import hydra
import speech_recognition as sr

from tensordict import TensorDict
from benchmarl.utils import DEVICE_TYPING
from vmas.simulator.utils import TorchUtils
from sentence_transformers import SentenceTransformer

# ROS2
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, UInt8
sys.path.insert(0, "/home/npfitzer/robomaster_ws/install/freyja_msgs/lib/python3.10/site-packages")
from freyja_msgs.msg import ReferenceState
from freyja_msgs.msg import CurrentState
from freyja_msgs.msg import WaypointTarget

# Local Modules
from scenarios.grids.language_grid import LanguageGrid
from scenarios.centralized.scripts.histories import VelocityHistory, PositionHistory
from scenarios.centralized.scripts.observation import observation
from scenarios.centralized.scripts.load_config import load_scenario_config
from deployment.helper_utils import convert_ne_to_xy, convert_xy_to_ne, get_experiment
from trainers.benchmarl_setup_experiment import benchmarl_setup_experiment

X = 0
Y = 1
MAP_NORMAL_SIZE = 2

class State:
    def __init__(self, pos, vel, rot, device):
        self.device = device
        self.pos = torch.tensor(pos, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.vel = torch.tensor(vel, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.rot = torch.tensor(rot, dtype=torch.float32, device=self.device).unsqueeze(0)

class Agent:
    def __init__(
    self,
    node,
    robot_id: int,
    weight: float,
    pos_history_length: int,
    grid: LanguageGrid,
    num_covered_targets: torch.Tensor,
    task_config,
    deployment_config,
    device: DEVICE_TYPING):
        
        self.timer: Optional[rclpy.timer.Timer] = None
        self._scale = torch.tensor([node.x_semidim, node.y_semidim], device=device)
        
        self.node = node
        self.robot_id = robot_id
        self.weight = weight
        
        # Timer to update state
        self.obs_dt = deployment_config.obs_dt
        self.mytime = 0.0
        
        # State buffer
        self.state_buffer = []
        self.state_buffer_length = 5
        
        self.a_range = deployment_config.a_range
        self.v_range = deployment_config.v_range
        self.device = device
        self.state_received = False
        # State variables updated by current_state_callback
        
        self.state = State(
            pos=[0.0, 0.0],
            vel=[0.0, 0.0],
            rot=[0.0],
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
        topic_prefix = getattr(deployment_config, "topic_prefix", "/robomaster_")
        
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
        self.state.rot[0,0] = msg.state_vector[5]
        
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
    
class World:
    def __init__(self, agents: List[Agent], dt: float):
        self.agents = agents
        self.dt = dt

class VmasModelsROSInterface(Node):

    def __init__(self, config: DictConfig, log_dir: Path, use_speech_to_text: bool):
        super().__init__("vmas_ros_interface")
        self.device = config.device 
        deployment_config = config["deployment"]
        task_config = config["task"].params
        self.llm = SentenceTransformer(deployment_config.llm_model, device="cpu")
        self.use_speech_to_text = use_speech_to_text
        
        # Task Config
        self.task_x_semidim = task_config.x_semidim
        self.task_y_semidim = task_config.y_semidim
        load_scenario_config(task_config,self)
        self._create_occupancy_grid()
        self.num_covered_targets = torch.zeros(1, dtype=torch.int, device=self.device)
        self.max_target_count = torch.tensor([self.n_targets],dtype=torch.int, device=self.device)

        # Grid Config
        grid_config = config["grid_config"]
        self.x_semidim = grid_config.x_semidim
        self.y_semidim = grid_config.y_semidim

        # History Config
        self.pos_dim = 2
        task_config.use_grid_data = False
        self.pos_history_length = task_config.history_length

        # Load experiment and get policy
        experiment = benchmarl_setup_experiment(cfg=config)
        self.policy = experiment.policy

        # Setup CSV logging
        # Setup logging to CSV in runtime dir
        self.log_file = open(log_dir / "vmas_log.csv", "w", newline='')
        self.csv_writer = csv.writer(self.log_file)
        self.csv_writer.writerow(["experiment_time", "real_time", "robot_id", "cmd_vel_n", "cmd_vel_e", 
                                  "pos_n", "pos_e", "vel_n", "vel_e"])

        # Create Agents
        agents: List[Agent] = []
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
                task_config = task_config,
                deployment_config = deployment_config,
                device=self.device
            )
            agents.append(agent)
        
        
        # Create action loop
        self.action_dt = deployment_config.action_dt
        self.max_steps = deployment_config.max_steps
        self.step_count = 0
        self.world = World(agents,self.action_dt)
        
        self.get_logger().info("ROS2 starting ..")
    
    def _create_occupancy_grid(self):
        
        self.occupancy_grid = LanguageGrid(
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

        obs_dict = self._collect_observations()
        if not obs_dict:
            self.get_logger().warn("No valid observations collected. Skipping this timestep.")
            return

        input_td = self._prepare_input_tensor(obs_dict)

        with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
            output_td = self.policy(input_td)

        action = output_td[("agents", "action")]

        self._issue_commands_to_agents(action)
        self.step_count += 1

    def _all_states_received(self):
        return all(agent.state_received for agent in self.world.agents)

    def _reached_max_steps(self):
        return self.step_count >= self.max_steps

    def _handle_termination(self):
        self.get_logger().info(f"Robots reached {self.max_steps} steps. Stopping.")
        self.timer.cancel()
        self.stop_all_agents()
        if self.use_speech_to_text:
            self.prompt_for_new_speech_instruction()
        else:
            self.prompt_for_new_instruction()

    def _collect_observations(self):
        data_dict = {} 

        for agent in self.world.agents:
            if not agent.state_buffer:
                self.get_logger().warn(f"No state in buffer for agent {agent.robot_id}")
                return

            latest_state = agent.state_buffer.pop()
            for key, value in latest_state.items():
                data_dict.setdefault(key, []).append(value.float())

        return data_dict

    def _prepare_input_tensor(self, data_dict):
        obs_dict = {
            ("agents", "observation", key): torch.cat(tensor_list, dim=0)
            for key, tensor_list in data_dict.items()
        }

        return TensorDict(obs_dict, batch_size=[len(next(iter(data_dict.values())))])

    def clamp_velocity_to_bounds(self, action: torch.Tensor, agent):
        """
        • If self.use_kinematic_model is False
            action = [vx, vy]               (Cartesian velocity in task space)
        • If self.use_kinematic_model is True
            action = [‖v‖, ω]               (forward-speed ‖v‖ and yaw-rate ω)

        Returns
        -------
        (vel_xy, omega)
            vel_xy : list[float] – clamped (vx, vy)
            omega  : float | None – yaw-rate in rad s⁻¹ (None when kinematic model is OFF)
        """
        pos   = agent.state.pos[0]              # shape (2,)
        theta = agent.state.rot[0]              # shape (1,)
        vel   = action.clone() 

        # ── 1. Interpret the raw action ──────────────────────────────────────────
        if self.use_kinematic_model:
            vel_norm = action[X]
            omega    = action[Y]
            vel[X]   = vel_norm * torch.cos(theta)
            vel[Y]   = vel_norm * torch.sin(theta)
        else:
            omega = None            # no rotational component

        # ── 2. Scale to the deployment arena ─────────────────────────────────────
        vel[X] *= self.x_semidim / self.task_x_semidim
        vel[Y] *= self.y_semidim / self.task_y_semidim

        # ── 3. Keep the next position inside the walls ───────────────────────────
        bounds_min = torch.tensor(
            [-self.x_semidim, -self.y_semidim], device=action.device
        ) + self.agent_radius
        bounds_max = torch.tensor(
            [ self.x_semidim,  self.y_semidim], device=action.device
        ) - self.agent_radius

        next_pos   = pos + vel * self.action_dt
        below_min  = next_pos < bounds_min
        above_max  = next_pos > bounds_max

        vel[below_min] = (bounds_min[below_min] - pos[below_min]) / self.action_dt
        vel[above_max] = (bounds_max[above_max] - pos[above_max]) / self.action_dt

        # ── 4. Respect the agent-speed envelope ──────────────────────────────────
        vel = TorchUtils.clamp_with_norm(vel, agent.v_range)

        return vel.tolist(), None if omega is None else omega.item()

    def _wrap_to_pi(self, angle: float) -> float:
        """Return the equivalent angle in the range [-π, π)."""
        return (angle + math.pi) % (2 * math.pi) - math.pi


    def _issue_commands_to_agents(self, action_tensor):
        real_time_str = datetime.now().isoformat()

        for i, agent in enumerate(self.world.agents):
            cmd_vel_xy, cmd_omega = self.clamp_velocity_to_bounds(action_tensor[i], agent)
            vel_n, vel_e = convert_xy_to_ne(*cmd_vel_xy)

            # ── Heading reference ────────────────────────────────────────────────
            if self.use_kinematic_model:
                yaw_now = agent.state.rot.item()          # current estimate
                yaw_ref = self._wrap_to_pi(yaw_now + cmd_omega * self.action_dt)
            else:
                yaw_ref   = math.pi / 2                   # legacy fixed heading
                cmd_omega = 0.0

            ref = agent.reference_state
            ref.vn   = vel_n
            ref.ve   = vel_e
            ref.yaw  = yaw_ref
            ref.an   = agent.a_range
            ref.ae   = agent.a_range
            ref.header.stamp = self.get_clock().now().to_msg()

            # ── Console / CSV logging ────────────────────────────────────────────
            self.get_logger().info(
                f"Robot {agent.robot_id} | v_ne=({vel_n:.3f}, {vel_e:.3f}) "
                f"| ω={cmd_omega:.3f} | pos=({agent.state.pos[0,X].item():.3f}, {agent.state.pos[0,Y].item():.3f}) "
                f"| yaw={agent.state.rot[0].item():.3f}"
            )

            self.csv_writer.writerow([
                agent.mytime, real_time_str, agent.robot_id,
                ref.vn, ref.ve,
                agent.state.pos[0,X], agent.state.pos[0,Y],
                agent.state.vel[0,X], agent.state.vel[0,Y],
            ])
            self.log_file.flush()

            agent.pub.publish(ref)
            agent.mytime += self.action_dt


    def stop_all_agents(self):
        if getattr(self, "timer", None): self.timer.cancel()
        for agent in self.world.agents:
            if agent.timer is not None:
                agent.timer.cancel()
            agent.state_buffer.clear()
            agent.send_zero_velocity()
    
    def _parse_goal_from_input(self, txt: str) -> torch.Tensor:
        """
        Expected formats:
        •  "3.0 -2.5"
        •  "3.0, -2.5"
        Returns a (1, 2) float32 tensor on the correct device.
        """
        # replace comma with space, split, take first two items
        try:
            n_str, e_str = txt.replace(",", " ").split()[:2]
            goal = torch.tensor([[float(n_str), float(e_str)]],
                                dtype=torch.float32,
                                device=self.device)
            return goal
        except Exception:
            raise ValueError(
                "Invalid goal format. Please enter two numbers, e.g. '1.2 -0.8'"
            )
            
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
        for agent in self.world.agents:
            if agent.timer: agent.timer.cancel(); agent.timer = None
            agent.mytime = 0
            agent.timer = self.create_timer(agent.obs_dt, agent.collect_observation)
        self.get_logger().info("Starting agents with new instruction.")


    def prompt_for_new_speech_instruction(self):
        
        recognizer = sr.Recognizer()
        mic = sr.Microphone()

        self.get_logger().info("Listening for new instruction... (or press Enter to type instead)")

        try:
            with mic as source:
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
                print(f"\n Audio recieved, sending to Google Speech API""\n")
            new_sentence = recognizer.recognize_google(audio)
            print(f"\n[Speech Recognized] \"{new_sentence}\"\n")
            self.get_logger().info(f"Received spoken instruction: {new_sentence}")
        except (sr.WaitTimeoutError, sr.UnknownValueError, sr.RequestError) as e:
            self.get_logger().warn(f"Speech recognition failed: {e}")
            user_input = input("Speech recognition failed. Type instruction or press Enter to try again: ").strip()
            if user_input:
                new_sentence = user_input
            else:
                self.get_logger().info("Retrying speech recognition...")
                return self.prompt_for_new_speech_instruction()
        except Exception as e:
            self.get_logger().error(f"Unexpected error during speech recognition: {e}")
            return

        try:
            embedding = torch.tensor(self.llm.encode([new_sentence]), device=self.device).squeeze(0)
        except Exception as e:
            self.get_logger().error(f"Failed to encode instruction: {e}")
            return

        self.occupancy_grid.embeddings[0] = embedding

       # Reset step count and timers
        self.step_count = 0
        self.timer = self.create_timer(self.action_dt, self.timer_callback)
        for agent in self.world.agents:
            agent.mytime = 0
            if agent.timer: agent.timer.cancel(); agent.timer = None
            agent.timer = self.create_timer(agent.obs_dt, agent.collect_observation)
        self.get_logger().info("Starting agents with new instruction.")
    

def get_runtime_log_dir():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runtime_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    log_dir = runtime_dir / f"run_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir

# Run script with:
# python deployment/my_deployment.py --config-path=path_to/conf_folder --config-name=config_file_name restore_path=path/to/checkpoint.pt
@hydra.main(version_base=None,config_path="../conf",config_name="deployment/unicycle_single_agent")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg, resolve=True))   # full merged config
    rclpy.init()

    cfg.experiment.restore_file = cfg.restore_path
    use_speech_to_text = cfg.use_speech_to_text

    log_dir = get_runtime_log_dir()

    # Instantiate interface
    ros_interface_node = VmasModelsROSInterface(
        config=cfg,
        log_dir=log_dir,
        use_speech_to_text=use_speech_to_text
    )
    
    if use_speech_to_text:
        ros_interface_node.prompt_for_new_speech_instruction()
    else:
        ros_interface_node.prompt_for_new_instruction()

    def sigint_handler(sig, frame):
        ros_interface_node.get_logger().info('SIGINT received. Stopping timer and sending zero velocity...')
        ros_interface_node.stop_all_agents()
        rclpy.spin_once(ros_interface_node, timeout_sec=0.5)
        ros_interface_node.destroy_node()
        ros_interface_node.log_file.close()
        rclpy.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, sigint_handler)
    rclpy.spin(ros_interface_node)


if __name__ == '__main__':
    main()