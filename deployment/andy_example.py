import rclpy
from rclpy.node import Node

import sys
import math
import signal
import csv
from datetime import datetime  # For real-world time logging

from std_msgs.msg import String
from std_msgs.msg import UInt8
from freyja_msgs.msg import ReferenceState
from freyja_msgs.msg import CurrentState
from freyja_msgs.msg import WaypointTarget

from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torchrl.modules import ProbabilisticActor, TanhNormal
from torchrl.modules.models.multiagent import MultiAgentMLP

import torch

import os
print(os.getcwd())
from andy_controller import evaluate_model as MDEval

import shutil
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf


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


class VmasModelsROSInterface(Node):

    def __init__(self, config: DictConfig, log_dir: Path):
        super().__init__("vmas_ros_interface")

        self.robot_id = config.robomaster_id
        self.a_range = config.a_range
        self.v_range = config.v_range
        self.model_name = config.model_name
        self.target_n = config.target_ne[0]
        self.target_e = config.target_ne[1]

        # Load model
        actor_net = nn.Sequential(
            MultiAgentMLP(
                n_agent_inputs=6,
                n_agent_outputs=2 * 2,
                n_agents=1,
                centralised=False,
                share_params=True,
                device=config.device,
                depth=2,
                num_cells=256,
                activation_class=nn.Tanh,
            ),
            NormalParamExtractor(),
        )
        policy_module = TensorDictModule(
            actor_net,
            in_keys=[("agents", "observation")],
            out_keys=[("agents", "loc"), ("agents", "scale")],
        )
        self.policy = ProbabilisticActor(
            module=policy_module,
            in_keys=[("agents", "loc"), ("agents", "scale")],
            out_keys=[("agents", "action")],
            distribution_class=TanhNormal,
            distribution_kwargs={
                "low": torch.ones([1, 1, 2]).to(config.device) * (-1),
                "high": torch.ones([1, 1, 2]).to(config.device) * (1)
            },
            return_log_prob=True,
        )

        actor_net.load_state_dict(torch.load(self.model_name, map_location=config.device))
        self.get_logger().info("Model ready!")

        # Initialize current state variables
        self.state_received = False
        # State variables updated by current_state_callback
        self.current_pos_n = 0.0
        self.current_pos_e = 0.0
        self.current_vel_n = 0.0
        self.current_vel_e = 0.0

        # Create publisher and subscriber for the single robot
        self.pub = self.create_publisher(ReferenceState, f"/robomaster_{self.robot_id}/reference_state", 1)
        self.create_subscription(CurrentState, f"/robomaster_{self.robot_id}/current_state", self.current_state_callback, 1)

        # Create reference state message
        self.reference_state = ReferenceState()

        # Timer to update and publish commands
        self.dt = 1.0 / 20.0
        self.mytime = 0.0
        self.timer = self.create_timer(self.dt, self.timer_callback)
        self.get_logger().info("ROS2 starting ..")

        # Setup CSV logging
                # Setup logging to CSV in runtime dir
        self.log_file = open(log_dir / "vmas_log.csv", "w", newline='')
        self.csv_writer = csv.writer(self.log_file)
        self.csv_writer.writerow(["experiment_time", "real_time", "robot_id", "cmd_vel_n", "cmd_vel_e", 
                                  "pos_n", "pos_e", "vel_n", "vel_e"])

    def timer_callback(self):
        # Run repeatedly to send commands to the robot
        if not self.state_received:
            self.get_logger().info("Current state not received yet.")
            return

        # Convert current state to model input (x-y ordering)
        pos_x, pos_y = convert_ne_to_xy(self.current_pos_n, self.current_pos_e)
        vel_x, vel_y = convert_ne_to_xy(self.current_vel_n, self.current_vel_e)

        # Convert absolute target location from NE to x-y ordering
        target_x, target_y = convert_ne_to_xy(self.target_n, self.target_e)
        # Calculate the relative target location in x-y coordinates
        relative_target_x = target_x - pos_x
        relative_target_y = target_y - pos_y

        # Evaluate the model with the converted inputs and relative target
        cmd_vel_tensor = MDEval.compute_action(
            pos_x,
            pos_y,
            vel_x,
            vel_y,
            target_location=[relative_target_x, relative_target_y],
            policy=self.policy,
            u_range=self.v_range,
            deterministic=True,
        )
        cmd_vel = cmd_vel_tensor.tolist()

        # Convert model output back to north-east ordering.
        # Since the model returns (vx, vy) in x-y order, we convert it back:
        vel_n, vel_e = convert_xy_to_ne(*cmd_vel)
        self.reference_state.vn = vel_n
        self.reference_state.ve = vel_e
        self.reference_state.yaw = math.pi / 2
        self.reference_state.an = self.a_range
        self.reference_state.ae = self.a_range
        self.reference_state.header.stamp = self.get_clock().now().to_msg()

        # Get the real-world time in ISO format
        real_time_str = datetime.now().isoformat()

        # Log to console
        self.get_logger().info(
            f"Robot {self.robot_id} - Commanded velocity ne: {vel_n}, {vel_e} - "
            f"pos: [{self.current_pos_n}, {self.current_pos_e}] - vel: [{self.current_vel_n}, {self.current_vel_e}]"
        )

        # Log to CSV file including both simulation time and real-world time
        self.csv_writer.writerow([self.mytime, real_time_str, self.robot_id, self.reference_state.vn, self.reference_state.ve,
                                  self.current_pos_n, self.current_pos_e, self.current_vel_n, self.current_vel_e])
        self.log_file.flush()

        # Publish the command
        self.pub.publish(self.reference_state)

        # Update simulation time
        self.mytime += self.dt

    def current_state_callback(self, msg: CurrentState):
        # Extract current state values from the state vector
        self.current_pos_n = msg.state_vector[0]
        self.current_pos_e = msg.state_vector[1]
        self.current_vel_n = msg.state_vector[3]
        self.current_vel_e = msg.state_vector[4]
        self.state_received = True

    def send_zero_velocity(self):
        # Send a zero velocity command
        self.reference_state.vn = 0.0
        self.reference_state.ve = 0.0
        self.reference_state.header.stamp = self.get_clock().now().to_msg()
        self.get_logger().info(f"Robot {self.robot_id} - Zero velocity command sent.")
        self.pub.publish(self.reference_state)
        self.log_file.flush()


@hydra.main(config_path="/home/andy/Documents/part_iii_project/robomaster_ws/src/RoboMasterROSPackage/config", 
            config_name="config_sim2real_deployment", version_base="1.2")
def main(config: DictConfig):
    rclpy.init()

    # Create runtime log dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runtime_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    log_dir = runtime_dir / f"run_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)

    print('Config:', OmegaConf.to_yaml(config))

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
        
        # Cancel the periodic timer first to prevent another callback
        ros_interface_node.timer.cancel()
        
        # Send zero velocity after timer is canceled
        ros_interface_node.send_zero_velocity()
        
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
