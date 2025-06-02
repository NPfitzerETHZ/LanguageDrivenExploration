import torch
import vmas.simulator.core
import vmas.simulator.utils
from vmas.simulator.dynamics.common import Dynamics

class KinematicUnicycle(Dynamics):
    """
    Kinematic unicycle model.
    State = [x, y, θ]ᵀ.
    Inputs:
        v      – linear velocity command  [m s⁻¹]
        omega  – yaw-rate command         [rad s⁻¹]
    """

    def __init__(
        self,
        world: vmas.simulator.core.World,
        use_velocity_controller: bool = False,
        integration: str = "rk4",  # "euler" or "rk4"
    ):
        super().__init__()
        assert integration in ("rk4", "euler")
        self.dt = world.dt
        self.integration = integration
        self.use_velocity_controller = use_velocity_controller

    # ─────────────────────────────── helpers ────────────────────────────────
    def f(self, state, omega_command, v_command):
        """Continuous-time dynamics ẋ = f(x,u)."""
        theta = state[:, 2]
        dx     = v_command * torch.cos(theta)
        dy     = v_command * torch.sin(theta)
        dtheta = omega_command
        return torch.stack((dx, dy, dtheta), dim=1)          # [batch, 3]

    def euler(self, state, omega_command, v_command):
        return self.dt * self.f(state, omega_command, v_command)

    def runge_kutta(self, state, omega_command, v_command):
        k1 = self.f(state,                   omega_command, v_command)
        k2 = self.f(state + self.dt * k1/2,  omega_command, v_command)
        k3 = self.f(state + self.dt * k2/2,  omega_command, v_command)
        k4 = self.f(state + self.dt * k3,    omega_command, v_command)
        return (self.dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

    # ───────────────────────────── interface ────────────────────────────────
    @property
    def needed_action_size(self) -> int:
        return 2          # [v, omega]

    def process_action(self):
        # Retrieve user commands
        v_command     = torch.clamp(self.agent.action.u[:, 0], min=0.0)
        omega_command = self.agent.action.u[:, 1]

        # Current full state [x, y, θ]
        state = torch.cat((self.agent.state.pos, self.agent.state.rot), dim=1)

        # Current velocities
        v_cur_x       = self.agent.state.vel[:, 0]
        v_cur_y       = self.agent.state.vel[:, 1]
        v_cur_angular = self.agent.state.ang_vel[:, 0]

        # Integrate
        if self.integration == "euler":
            delta_state = self.euler(state, omega_command, v_command)
        else:
            delta_state = self.runge_kutta(state, omega_command, v_command)
        
        new_theta = state[:, 2] + delta_state[:, 2]
        wrapped_theta = torch.atan2(torch.sin(new_theta), torch.cos(new_theta))
        delta_state[:, 2] = wrapped_theta - state[:, 2]
        
        if self.use_velocity_controller:
            # Compute target velocity from delta_state
            target_vel_x = delta_state[:, 0] / self.dt
            target_vel_y = delta_state[:, 1] / self.dt

            # Set desired linear velocity for PID
            self.agent.action.u = torch.stack((target_vel_x, target_vel_y), dim=1)

            # Use PID to compute forces (from velocity error)
            self.agent.controller.process_force()
            self.agent.state.force = self.agent.action.u[:, : self.needed_action_size]

            # Use standard angular acceleration to torque
            v_cur_angular = self.agent.state.ang_vel[:, 0]
            acc_angular = (delta_state[:, 2] - v_cur_angular * self.dt) / self.dt**2
            torque = self.agent.moment_of_inertia * acc_angular
            self.agent.state.torque = torque.unsqueeze(-1)
            return

        # Required accelerations
        acc_x       = (delta_state[:, 0] - v_cur_x * self.dt) / self.dt**2
        acc_y       = (delta_state[:, 1] - v_cur_y * self.dt) / self.dt**2
        acc_angular = (delta_state[:, 2] - v_cur_angular * self.dt) / self.dt**2

        # Convert to forces and torque
        force_x = self.agent.mass * acc_x
        force_y = self.agent.mass * acc_y
        torque  = self.agent.moment_of_inertia * acc_angular

        # Push to simulator
        self.agent.state.force[:, vmas.simulator.utils.X] = force_x
        self.agent.state.force[:, vmas.simulator.utils.Y] = force_y
        self.agent.state.torque = torque.unsqueeze(-1)
