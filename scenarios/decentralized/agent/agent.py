from __future__ import annotations

import torch
from typing import Iterable, Sequence, Optional, Callable

# VMAS core
from vmas.simulator.core import Agent as VmasAgent, Sphere, Color, World
from vmas.simulator.dynamics.holonomic import Holonomic
from vmas.simulator.controllers.velocity_controller import VelocityController

from scenarios.decentralized.scripts.histories import VelocityHistory, PositionHistory
from scenarios.kinematic_dynamic_models.kinematic_unicycle import KinematicUnicycle

# Local occupancy‑grid implementation (adjust import path if necessary)
from scenarios.grids.internal_grids import InternalOccupancyGrid

# -----------------------------------------------------------------------------
class DecentralizedAgent(VmasAgent):
    """An *all‑inclusive* agent subclass pre‑configured for decentralized
    exploration, coverage and target search experiments.

    It bundles together the base VMAS ``Agent`` with:
        • **Dynamics** – holonomic *or* kinematic unicycle (optionally with a
          low‑level velocity PID controller).
        • **Sensors** – pluggable factory so you can swap LiDAR or other sensor
          suites without touching the agent code.
        • **Internal occupancy grid** – tensor‑based visit / target / obstacle
          map, sized automatically for batched simulation.
        • **Reward & termination buffers** – zero‑initialised on the correct
          device and with the right batch dimension.
        • **State histories** – optional position / velocity circular buffers
          for autoregressive policies.
    """

    # --------------------------------------------------------------------- init
    def __init__(
        self,
        *,
        # ---------------------- identifiers & boilerplate ------------------
        name: str,
        world: World,
        batch_dim: int,
        device: torch.device,
        # --------------------------- geometry -----------------------------
        agent_radius: float = 0.05,
        agent_weight: float = 1.0,
        agent_u_range: float = 1.0,
        agent_f_range: float = 1.0,
        agent_v_range: float = 1.0,
        # ------------------------- behaviour flags -------------------------
        collide: bool = True,
        silent: bool = False,
        use_lidar: bool = True,
        use_kinematic_model: bool = False,
        use_velocity_controller: bool = False,
        # ------------------- occupancy‑grid configuration -----------------
        num_grid_cells: int = 21,
        x_semidim: float = 1.0,
        y_semidim: float = 1.0,
        grid_visit_threshold: int = 1,
        # ------------------------ state histories --------------------------
        observe_pos_history: bool = False,
        observe_vel_history: bool = False,
        pos_history_length: int = 8,
        vel_history_length: int = 8,
        pos_dim: int = 2,
        vel_dim: int = 2,
        # -------------------- sensors & controller hooks -------------------
        lidar_sensor_factory: Optional[Callable[[World], Iterable]] = None,
        pid_controller_params: Sequence[float] = (2.0, 6.0, 0.002),
        # ----------------------------- style -------------------------------
        color: Color = Color.GREEN,
    ) -> None:
        # ----------------------------------------------------------- dynamics
        dynamics = (
            KinematicUnicycle(world, use_velocity_controller)
            if use_kinematic_model
            else Holonomic()
        )

        # ------------------------------------------------------------- sensors
        sensors = (
            tuple(lidar_sensor_factory(world))
            if (use_lidar and lidar_sensor_factory is not None)
            else ()
        )

        # ----------------------------------------------------- call superclass
        super().__init__(
            name=name,
            collide=collide,
            silent=silent,
            shape=Sphere(radius=agent_radius),
            mass=agent_weight,
            u_range=agent_u_range,
            f_range=agent_f_range,
            v_range=agent_v_range,
            sensors=sensors,
            dynamics=dynamics,
            render_action=True,
            color=color,
        )

        # ------------------------------------------------- store configuration
        self.device = device
        self.observe_pos_history = observe_pos_history
        self.observe_vel_history = observe_vel_history
        self.pos_history_length = pos_history_length
        self.vel_history_length = vel_history_length
        self.pos_dim = pos_dim
        self.vel_dim = vel_dim

        # -------------------------------------------------- occupancy grid
        self.occupancy_grid = InternalOccupancyGrid(
            x_dim=2,  # local logical grid always spans [-1, 1]
            y_dim=2,
            x_scale=x_semidim,
            y_scale=y_semidim,
            num_cells=num_grid_cells,
            visit_threshold=grid_visit_threshold,
            batch_size=batch_dim,
            device=device,
        )

        # ----------------------------------------------------- low‑level PID
        if use_velocity_controller:
            self.controller = VelocityController(
                self, world, pid_controller_params, mode="standard"
            )

        # ------------------------------------------ reward & termination buf
        zeros = torch.zeros(batch_dim, device=device)
        self.collision_rew = zeros.clone()
        self.covering_reward = zeros.clone()
        self.exploration_rew = zeros.clone()
        self.coverage_rew = zeros.clone()
        self.num_covered_targets = torch.zeros(
            batch_dim, dtype=torch.int, device=device
        )
        self.termination_rew = zeros.clone()
        self.termination_signal = zeros.clone()

        # -------------------------------------------------- state histories
        self._create_agent_state_histories(batch_dim)

    # ----------------------------------------------------------------- priv
    def _create_agent_state_histories(self, batch_dim: int) -> None:
        """Allocate circular buffers for position / velocity if requested."""
        if self.observe_pos_history:
            self.position_history = PositionHistory(
                batch_dim,
                self.pos_history_length,
                self.pos_dim,
                self.device,
            )
        if self.observe_vel_history:
            self.velocity_history = VelocityHistory(
                batch_dim,
                self.vel_history_length,
                self.vel_dim,
                self.device,
            )

