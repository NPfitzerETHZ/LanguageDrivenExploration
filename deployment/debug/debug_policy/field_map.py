#!/usr/bin/env python3
# ╭──────────────────────────────────────────────────────────────╮
# │ Interactive velocity-field visualiser with three sliders     │
# │  • θ₀  – initial heading                                     │
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

# project-level helpers
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
    device   = cfg.device
    experiment = benchmarl_setup_experiment(cfg=cfg)
    policy     = experiment.policy.to(device).eval()

    # ------------------------------------------------------------------ #
    # 2.  Prepare grid (static)                                          #
    # ------------------------------------------------------------------ #
    xs, ys = cfg.grid_config.x_semidim, cfg.grid_config.y_semidim
    dx     = 0.5
    x = torch.arange(-xs, xs + dx, dx, device=device)
    y = torch.arange(-ys, ys + dx, dx, device=device)
    X, Y  = torch.meshgrid(x, y, indexing="xy")
    N     = X.numel()

    # scale factors and constant observation components
    _scale = torch.tensor([xs, ys], device=device)                      # (2,)
    pos    = torch.stack((X.flatten(), Y.flatten()), 1).unsqueeze(1) / _scale   # (N,1,2)
    vel    = torch.zeros(N, 1, 2, device=device)                       # (N,1,2)

    # ------------------------------------------------------------------ #
    # 3.  Helper: evaluate policy for given (θ, goal)                    #
    # ------------------------------------------------------------------ #
    def evaluate_policy(theta: float, goal_xy: torch.Tensor):
        """
        Evaluate policy for heading theta [rad] and goal position [x,y] (m).
        Returns east, north components and colour map array for quiver.
        """
        rot = torch.full((N, 1, 1), theta, device=device)              # (N,1,1)

        rel = (torch.stack((X.flatten(), Y.flatten()), 1) - goal_xy).unsqueeze(1) / _scale
        obs = torch.cat((pos, rot, vel, rel), dim=-1)                  # (N,1,Mtot)

        with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
            td_in = TensorDict({("agents", "observation"): obs}, batch_size=[N])
            act   = policy(td_in)[("agents", "action")]                # (N,1,2)

        speed = act[:, 0, 0]                                           # (N,)
        omega = act[:, 0, 1]

        # world-frame velocity
        vx = speed * torch.cos(rot.squeeze()) * xs / cfg.task.params.x_semidim
        vy = speed * torch.sin(rot.squeeze()) * ys / cfg.task.params.y_semidim

        U = vx.view(X.shape).cpu().numpy()                             # east (grid shape)
        V = vy.view(Y.shape).cpu().numpy()                             # north
        W = omega.view(X.shape).cpu().numpy()                          # angular rate ω

        # colour normalised to ±w_max → [0,1]
        w_max = np.max(np.abs(W)) + 1e-8
        C = (W + w_max) / (2 * w_max)

        N_arr, E_arr = convert_xy_to_ne(U, V)                          # swap to NE
        return E_arr, N_arr, C

    # ------------------------------------------------------------------ #
    # 4.  Build interactive plot                                         #
    # ------------------------------------------------------------------ #
    θ0     = 0.0
    goal0  = torch.tensor([0.0, 0.0], device=device)                   # default goal at (0,0)
    E0, N0, C0 = evaluate_policy(θ0, goal0)

    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(bottom=0.30)                                   # room for sliders

    quiv = ax.quiver(X.cpu(), Y.cpu(), E0, N0, C0,
                     cmap="seismic", scale_units="xy", scale=7,
                     width=0.003, pivot="mid")
    fig.colorbar(quiv, ax=ax, label="angular rate ω  [rad s⁻¹]")

    ax.set_xlim(-xs, xs); ax.set_ylim(-ys, ys); ax.set_aspect("equal")
    ax.axhline(0, lw=0.4, color="k"); ax.axvline(0, lw=0.4, color="k")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    title = ax.set_title(f"Velocity field  –  θ₀ = {θ0:+.2f} rad, goal (0, 0) m")

    # ── sliders ────────────────────────────────────────────────────────
    # θ₀ slider
    ax_θ = plt.axes([0.12, 0.18, 0.76, 0.04])
    sθ = Slider(ax_θ, "θ₀ [rad]", -math.pi,  math.pi,
                valinit=θ0, valstep=0.02, color="gray")

    # goal-x slider
    ax_gx = plt.axes([0.12, 0.10, 0.76, 0.04])
    sgx = Slider(ax_gx, "Goal x [m]", -xs, xs,
             valinit=0.0, color="steelblue")

    # goal-y slider
    ax_gy = plt.axes([0.12, 0.02, 0.76, 0.04])
    sgy = Slider(ax_gy, "Goal y [m]", -ys, ys,
             valinit=0.0, color="steelblue")

    # optional red cross marking the goal
    goal_marker, = ax.plot(goal0[0].cpu(), goal0[1].cpu(), "xr", ms=8, mew=2, label="goal")

    def update(_):
        θ  = sθ.val
        gx = float(sgx.val)           # ← cast to plain Python float
        gy = float(sgy.val) 
        goal = torch.tensor([gx, gy], device=device)

        E, N, C = evaluate_policy(θ, goal)
        quiv.set_UVC(E, N, C)
        goal_marker.set_data([gx], [gy])
        title.set_text(
            f"Velocity field  –  θ₀ = {θ:+.2f} rad, goal ({gx:+.1f}, {gy:+.1f}) m"
        )
        fig.canvas.draw_idle()

    # update on any slider movement
    for slider in (sθ, sgx, sgy):
        slider.on_changed(update)

    plt.legend(loc="upper right", fontsize="small")
    plt.show()


# ────────────────────────────────────────────────────────────────
# Command-line execution guard
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
