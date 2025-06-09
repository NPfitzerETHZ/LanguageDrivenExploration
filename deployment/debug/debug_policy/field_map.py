#!/usr/bin/env python3
# ╭──────────────────────────────────────────────────────────────╮
# │ Interactive velocity‑field visualiser with five sliders      │
# │  • θ₀   – initial heading                                    │
# │  • vₓ   – east‑velocity component (m s⁻¹)                    │
# │  • v_y  – north‑velocity component (m s⁻¹)                   │
# │  • Goal x – horizontal goal coordinate (m)                   │
# │  • Goal y – vertical   goal coordinate (m)                   │
# ╰──────────────────────────────────────────────────────────────╯
import math
import torch
import hydra
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from omegaconf import DictConfig
from tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type

# project‑level helpers
from trainers.benchmarl_setup_experiment import benchmarl_setup_experiment
from deployment.helper_utils import convert_xy_to_ne


# ────────────────────────────────────────────────────────────────
# Hydra entry point
# ────────────────────────────────────────────────────────────────
@hydra.main(
    version_base=None,
    config_path="../../../conf",
    config_name="deployment/single_agent_navigation",
)
def main(cfg: DictConfig) -> None:

    # ------------------------------------------------------------------ #
    # 1.  Load policy                                                    #
    # ------------------------------------------------------------------ #
    cfg.experiment.restore_file = (
        "/Users/nicolaspfitzer/ProrokLab/CustomScenarios/checkpoints/benchmarl/"
        "single_agent_navigation/single_agent_navigation.pt"
    )
    device = cfg.device
    experiment = benchmarl_setup_experiment(cfg=cfg)
    policy = experiment.policy.to(device).eval()

    # ------------------------------------------------------------------ #
    # 2.  Prepare grid (static)                                          #
    # ------------------------------------------------------------------ #
    xs, ys = cfg.grid_config.x_semidim, cfg.grid_config.y_semidim
    dx = 0.5
    x = torch.arange(-xs, xs + dx, dx, device=device)
    y = torch.arange(-ys, ys + dx, dx, device=device)
    X, Y = torch.meshgrid(x, y, indexing="xy")
    N = X.numel()

    # scale factors and constant observation components
    _scale = torch.tensor([xs, ys], device=device)  # (2,)
    pos = torch.stack((X.flatten(), Y.flatten()), 1).unsqueeze(1) / _scale  # (N,1,2)

    # ------------------------------------------------------------------ #
    # 3.  Helper: evaluate policy for given (θ, v_xy, goal)              #
    # ------------------------------------------------------------------ #
    def evaluate_policy(theta: float, goal_xy: torch.Tensor, vel_xy: torch.Tensor):
        """
        Evaluate policy for heading *theta* [rad], constant velocity vector *vel_xy*
        (east & north in m s⁻¹) and goal position *goal_xy* [x,y] (m).
        Returns east, north components and colour‑map array for quiver.
        """
        rot = torch.full((N, 1, 1), theta, device=device)  # (N,1,1)

        rel = (torch.stack((X.flatten(), Y.flatten()), 1) - goal_xy).unsqueeze(1) / _scale
        vel = vel_xy.view(1, 1, 2).expand(N, 1, 2)  # (N,1,2)
        obs = torch.cat((pos, rot, vel, rel), dim=-1)  # (N,1,Mtot)

        with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
            td_in = TensorDict({("agents", "observation"): obs}, batch_size=[N])
            act = policy(td_in)[("agents", "action")]  # (N,1,2)

        speed = act[:, 0, 0]  # (N,)
        omega = act[:, 0, 1]

        # world‑frame velocity predicted by policy (scaled back to metres per second)
        vx = speed * torch.cos(rot.squeeze()) * xs / cfg.task.params.x_semidim
        vy = speed * torch.sin(rot.squeeze()) * ys / cfg.task.params.y_semidim

        U = vx.view(X.shape).cpu().numpy()  # east (grid shape)
        V = vy.view(Y.shape).cpu().numpy()  # north
        W = omega.view(X.shape).cpu().numpy()  # angular rate ω

        # colour normalised to ±w_max → [0,1]
        w_max = np.max(np.abs(W)) + 1e-8
        C = (W + w_max) / (2 * w_max)

        N_arr, E_arr = convert_xy_to_ne(U, V)  # swap to NE
        return E_arr, N_arr, C

    # ------------------------------------------------------------------ #
    # 4.  Build interactive plot                                         #
    # ------------------------------------------------------------------ #
    θ0 = 0.0
    v0 = torch.tensor([0.0, 0.0], device=device)  # default velocity (v_x, v_y)
    goal0 = torch.tensor([0.0, 0.0], device=device)  # default goal at (0,0)

    E0, N0, C0 = evaluate_policy(θ0, goal0, v0)

    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(bottom=0.42)  # room for five sliders

    quiv = ax.quiver(
        X.cpu(),
        Y.cpu(),
        E0,
        N0,
        C0,
        cmap="seismic",
        scale_units="xy",
        scale=7,
        width=0.003,
        pivot="mid",
    )
    fig.colorbar(quiv, ax=ax, label="angular rate ω  [rad s⁻¹]")

    ax.set_xlim(-xs, xs)
    ax.set_ylim(-ys, ys)
    ax.set_aspect("equal")
    ax.axhline(0, lw=0.4, color="k")
    ax.axvline(0, lw=0.4, color="k")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    title = ax.set_title(
        "Velocity field  –  θ₀ = {:+.2f} rad, v=({:+.1f},{:+.1f}) m/s, goal (0, 0) m".format(
            θ0, v0[0], v0[1]
        )
    )

    # ── sliders ────────────────────────────────────────────────────────
    # v_x slider (east component)
    ax_vx = plt.axes([0.12, 0.34, 0.76, 0.04])
    svx = Slider(ax_vx, "v_x [m s⁻¹]", -2.0, 2.0, valinit=0.0, valstep=0.01, color="olive")

    # v_y slider (north component)
    ax_vy = plt.axes([0.12, 0.26, 0.76, 0.04])
    svy = Slider(ax_vy, "v_y [m s⁻¹]", -2.0, 2.0, valinit=0.0, valstep=0.01, color="olive")

    # θ₀ slider
    ax_theta = plt.axes([0.12, 0.18, 0.76, 0.04])
    sθ = Slider(ax_theta, "θ₀ [rad]", -math.pi, math.pi, valinit=θ0, valstep=0.02, color="gray")

    # goal‑x slider
    ax_gx = plt.axes([0.12, 0.10, 0.76, 0.04])
    sgx = Slider(ax_gx, "Goal x [m]", -xs, xs, valinit=0.0, color="steelblue")

    # goal‑y slider
    ax_gy = plt.axes([0.12, 0.02, 0.76, 0.04])
    sgy = Slider(ax_gy, "Goal y [m]", -ys, ys, valinit=0.0, color="steelblue")

    # optional red cross marking the goal
    goal_marker, = ax.plot(goal0[0].cpu(), goal0[1].cpu(), "xr", ms=8, mew=2, label="goal")

    def update(_):
        θ = sθ.val
        vx = float(svx.val)  # east component
        vy = float(svy.val)  # north component
        gx = float(sgx.val)
        gy = float(sgy.val)

        goal = torch.tensor([gx, gy], device=device)
        vel_xy = torch.tensor([vx, vy], device=device)

        E, N, C = evaluate_policy(θ, goal, vel_xy)
        quiv.set_UVC(E, N, C)
        goal_marker.set_data([gx], [gy])
        title.set_text(
            "Velocity field  –  θ₀ = {:+.2f} rad, v=({:+.1f},{:+.1f}) m/s, goal ({:+.1f}, {:+.1f}) m".format(
                θ, vx, vy, gx, gy
            )
        )
        fig.canvas.draw_idle()

    # update on any slider movement
    for slider in (sθ, svx, svy, sgx, sgy):
        slider.on_changed(update)

    plt.legend(loc="upper right", fontsize="small")
    plt.show()


# ────────────────────────────────────────────────────────────────
# Command‑line execution guard
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
