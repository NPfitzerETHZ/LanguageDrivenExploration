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

# ROS2
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, UInt8

USING_FREYJA = False;
CREATE_MAP_FRAME = True;

if USING_FREYJA:
    sys.path.insert(0, "/home/npfitzer/robomaster_ws/install/freyja_msgs/lib/python3.10/site-packages")
    from freyja_msgs.msg import ReferenceState
    from freyja_msgs.msg import CurrentState
    from freyja_msgs.msg import WaypointTarget
else:
    from geometry_msgs.msg import Twist, PoseStamped, Pose
    from nav_msgs.msg import Odometry
    from tf_transformations import euler_from_quaternion


# Local Modules
from deployment.helper_utils import convert_ne_to_xy, convert_xy_to_ne
from trainers.benchmarl_setup_experiment import benchmarl_setup_experiment

X = 0
Y = 1

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
    deployment_config,
    device: DEVICE_TYPING):
        
        self.timer: Optional[rclpy.timer.Timer] = None
        
        self.goal = None
        self._scale = torch.tensor([node.x_semidim, node.y_semidim], device=device)
        
        self.node = node
        self.robot_id = robot_id
    
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
        if CREATE_MAP_FRAME:
            self.home_pn = None;
            self.home_pe = None;
        else:
            self.home_pn = 0.0;
            self.home_pe = 0.0;

        # State variables updated by current_state_callback
        self.state = State(
            pos=[0.0, 0.0],
            vel=[0.0, 0.0],
            rot=[0.0],
            device=self.device
        )
        
        # Get topic prefix from config or use default
        topic_prefix = getattr(deployment_config, "topic_prefix", "/robomaster_")
        
        # Create publisher for the robot
        if USING_FREYJA:
            self.pub = self.node.create_publisher(
                ReferenceState,
                f"{topic_prefix}{self.robot_id}/reference_state",
                1
            )
        else:
            self.pub = self.node.create_publisher(
                Twist,
                "/willow1/cmd_vel",
                1
            )

        
        # Create subscription with more descriptive variable name
        if USING_FREYJA:
            self.state_subscription = self.node.create_subscription(
                CurrentState,
                f"{topic_prefix}{self.robot_id}/current_state",
                self.freyja_current_state_callback,
                1
            )
        else:
            self.state_subscription = self.node.create_subscription(
                Odometry,
                "/willow/odometry/gps",
                self.odom_current_state_callback,
                1
        )
        
        # Log the subscription
        self.node.get_logger().info(f"Robot {self.robot_id} subscribing to: {topic_prefix}{self.robot_id}/current_state")
    
        # Create reference state message
        if USING_FREYJA:
            self.reference_state = ReferenceState()
        else:
            self.reference_state = Twist()

    def freyja_current_state_callback(self, msg: CurrentState):
        # Extract current state values from the state vector
        current_pos_n = msg.state_vector[0]
        current_pos_e = msg.state_vector[1]
        current_rot = msg.state_vector[5]
        current_vel_n = msg.state_vector[3]
        current_vel_e = msg.state_vector[4]
        
        self.state.pos[0,X], self.state.pos[0,Y] = convert_ne_to_xy(current_pos_n, current_pos_e)
        self.state.vel[0,X], self.state.vel[0,Y] = convert_ne_to_xy(current_vel_n, current_vel_e)
        self.state.rot[0] = current_rot
        
        self.state_received = True

    def odom_current_state_callback(self, msg: Odometry):
        # Extract current state values from the state vector
        euler = euler_from_quaternion([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]);

        current_pos_n = msg.pose.pose.position.x;
        current_pos_e = msg.pose.pose.position.y;
        current_vel_n = msg.twist.twist.linear.x;
        current_vel_e = msg.twist.twist.linear.y;

        if CREATE_MAP_FRAME and self.home_pe is None:
            self.home_pn = current_pos_n;
            self.home_pe = current_pos_e;

        current_pos_n = (current_pos_n - self.home_pn);
        current_pos_e = (current_pos_e - self.home_pe);
        print(f"Robot {self.robot_id} - Euler[2]: {euler[2]:.4f}, pos_n: {current_pos_n:.4f}, pos_e: {current_pos_e:.4f}")

        self.state.pos[0,X], self.state.pos[0,Y] = convert_ne_to_xy(current_pos_n, current_pos_e)
        self.state.vel[0,X], self.state.vel[0,Y] = convert_ne_to_xy(current_vel_n, current_vel_e)
        self.state.rot[0] = euler[2];

        self.state_received = True
    
    def collect_observation(self):
        
        if self.goal is not None and self.state_received:
            
            obs = {
                "pos": self.state.pos / self._scale,
                "rot": self.state.rot,
                "vel": self.state.vel / self._scale,
                "obs": (self.state.pos - self.goal) / self._scale,
            }
            
            self.state_buffer.append(obs)
            if len(self.state_buffer) > self.state_buffer_length:
                self.state_buffer = self.state_buffer[-self.state_buffer_length:]
       
        
    def send_zero_velocity(self):
        # Send a zero velocity command
        if USING_FREYJA:
            self.reference_state.vn = 0.0
            self.reference_state.ve = 0.0
            self.reference_state.header.stamp = self.node.get_clock().now().to_msg()
        else:
            self.reference_state.linear.x = 0.0
            self.reference_state.angular.z = 0.0

        self.node.get_logger().info(f"Robot {self.robot_id} - Zero velocity command sent.")
        self.pub.publish(self.reference_state)
        self.node.log_file.flush()
    
class World:
    def __init__(self, agents: List[Agent], dt: float):
        self.agents = agents
        self.dt = dt

class VmasModelsROSInterface(Node):

    def __init__(self, config: DictConfig, log_dir: Path):
        super().__init__("vmas_ros_interface")
        self.device = config.device 
        grid_config = config["grid_config"]
        deployment_config = config["deployment"]
        task_config = config["task"].params
        self.x_semidim = grid_config.x_semidim
        self.y_semidim = grid_config.y_semidim
        self.n_agents = task_config.n_agents
        
        self.goal = None
        self.task_x_semidim = task_config.x_semidim
        self.task_y_semidim = task_config.y_semidim
        self.agent_radius = task_config.agent_radius

        # Load experiment and get policy
        experiment = benchmarl_setup_experiment(cfg=config)
        self.policy = experiment.policy

        # Setup CSV logging
        # Setup logging to CSV in runtime dir
        self.log_file = open(log_dir / "vmas_log.csv", "w", newline='')
        self.csv_writer = csv.writer(self.log_file)
        self.csv_writer.writerow(["experiment_time", "real_time", "robot_id", "cmd_vel", "cmd_omega", "cmd_vel_n", "cmd_vel_e",
                                  "cur_pos_n", "cur_pos_e", "cur_vel_n", "cur_vel_e"])

        # Create Agents
        agents: List[Agent] = []
        id_list = deployment_config.id_list
        assert len(id_list) == self.n_agents
        for i in range(self.n_agents):
            agent = Agent(
                node=self,
                robot_id=id_list[i],
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
    
    def timer_callback(self):
        if not self._all_states_received():
            self.get_logger().info("Waiting for all agents to receive state.")
            return

        if self._reached_max_steps():
            self._handle_termination()
            return

        obs_list = self._collect_observations()
        if not obs_list:
            self.get_logger().warn("No valid observations collected. Skipping this timestep.")
            return

        input_td = self._prepare_input_tensor(obs_list)

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
        self.prompt_for_new_instruction()

    def _collect_observations(self):
        obs_list = []

        for agent in self.world.agents:
            if not agent.state_buffer:
                self.get_logger().warn(f"No state in buffer for agent {agent.robot_id}")
                return

            latest_state = agent.state_buffer[-1]
            feat = torch.cat([
                latest_state["pos"],
                latest_state["rot"],
                latest_state["vel"],
                latest_state["obs"],], dim=-1).float()   # shape (1, 6)
            obs_list.append(feat)

        return obs_list

    def _prepare_input_tensor(self, obs_list):
        obs_tensor = torch.cat(obs_list, dim=0)

        return TensorDict({
            ("agents", "observation"): obs_tensor,
        }, batch_size=[len(obs_list)])
    
    def clamp_velocity_to_bounds(self, action: torch.Tensor, agent) -> List[float]:
        """
        Clamp the velocity so the agent remains within the environment bounds,
        accounting for the agent's radius and timestep.
        """
        pos = agent.state.pos[0]  # shape: (2,)
        theta = agent.state.rot[0]  # shape: (1,)
        vel = action.clone()
        
        vel_norm = action[X]
        omega = action[Y]  
        
        vel[X] = vel_norm * torch.cos(theta)
        vel[Y] = vel_norm * torch.sin(theta)
        
        # Scale Action to deployment environment
        vel[X] = vel[X] * self.x_semidim / self.task_x_semidim
        vel[Y] = vel[Y] * self.y_semidim / self.task_y_semidim

        bounds_min = torch.tensor([-self.x_semidim, -self.y_semidim], device=action.device) + self.agent_radius
        bounds_max = torch.tensor([ self.x_semidim,  self.y_semidim], device=action.device) - self.agent_radius

        next_pos = pos + vel * self.action_dt

        # Compute clamped velocity based on how far the agent can move without crossing bounds
        below_min = next_pos < bounds_min
        above_max = next_pos > bounds_max

        # Adjust velocity where next position would violate bounds
        vel[below_min] = (bounds_min[below_min] - pos[below_min]) / self.action_dt
        vel[above_max] = (bounds_max[above_max] - pos[above_max]) / self.action_dt

        # Clamp to the agent's max velocity norm
        clamped_vel = TorchUtils.clamp_with_norm(vel, agent.v_range)
        return clamped_vel.tolist(), omega.item()

    def _wrap_to_pi(self, angle: float) -> float:
        """Return the equivalent angle in the range [-π, π)."""
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def _issue_commands_to_agents(self, action_tensor):
        real_time_str = datetime.now().isoformat()

        for i, agent in enumerate(self.world.agents):

            cmd_vel, cmd_omega = self.clamp_velocity_to_bounds(action_tensor[i], agent)
            vel_n, vel_e = convert_xy_to_ne(*cmd_vel)

            yaw_now   = agent.state.rot.item()                      # current estimate from localisation
            yaw_ref   = self._wrap_to_pi(yaw_now + cmd_omega * self.action_dt)

            if USING_FREYJA:
                agent.reference_state.vn = vel_n
                agent.reference_state.ve = vel_e
                agent.reference_state.yaw = yaw_ref
                agent.reference_state.an = agent.a_range
                agent.reference_state.ae = agent.a_range
                agent.reference_state.header.stamp = self.get_clock().now().to_msg()
            else:
                agent.reference_state.linear.x = math.sqrt(vel_n**2 + vel_e**2);
                agent.reference_state.angular.z = cmd_omega;

            self.get_logger().info(
                f"Robot {agent.robot_id} - Commanded velocity ne: {vel_n}, {vel_e} - "
                f"Commanded yaw rate: {cmd_omega}"
                f"pos: [{agent.state.pos[0,X]}, {agent.state.pos[0,Y]}] - "
                f"yaw: [{agent.state.rot[0]}] - "
                f"vel: [{agent.state.vel[0,X]}, {agent.state.vel[0,Y]}]"
            )

            self.csv_writer.writerow([
                agent.mytime, real_time_str, agent.robot_id,
                cmd_vel, cmd_omega, vel_n, vel_e,
                agent.state.pos[0,X], agent.state.pos[0,Y],
                agent.state.vel[0,X], agent.state.vel[0,Y]
            ])
            self.log_file.flush()

            agent.pub.publish(agent.reference_state)
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
        txt = input("Enter N,E goal for the agents: ")
        try:
            goal_ne = self._parse_goal_from_input(txt)
            gx, gy = convert_ne_to_xy(goal_ne[0,0].item(), goal_ne[0,1].item())
            goal = torch.tensor([[gx, gy]], dtype=torch.float32, device=self.device)
        except ValueError as e:
            self.get_logger().error(str(e))
            return

        # Reset step count and timers
        self.step_count = 0
        self.timer = self.create_timer(self.action_dt, self.timer_callback)
        for agent in self.world.agents:
            agent.mytime = 0
            if agent.timer: agent.timer.cancel(); agent.timer = None
            agent.timer = self.create_timer(agent.obs_dt, agent.collect_observation)
            agent.goal = goal
        self.get_logger().info("Starting agents with new instruction.")

def get_runtime_log_dir():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runtime_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    log_dir = runtime_dir / f"run_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir

# Manually parse config_path and config_name from CLI
def extract_initial_config():
    config_path, config_name = None, None
    for arg in sys.argv:
        if arg.startswith("config_path="):
            config_path = arg.split("=", 1)[1]
        elif arg.startswith("config_name="):
            config_name = arg.split("=", 1)[1]
    return config_path, config_name

# Run script with:
# python deployment/debug/debug_policy/navigation_deployment.py restore_path=/path_to_checkpoint.pt
@hydra.main(version_base=None,config_path="../../../conf",config_name="deployment/single_agent_navigation")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg, resolve=True))   # full merged config
    rclpy.init()

    cfg.experiment.restore_file = cfg.restore_path

    log_dir = get_runtime_log_dir()

    # Instantiate interface
    ros_interface_node = VmasModelsROSInterface(
        config=cfg,
        log_dir=log_dir
    )

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
