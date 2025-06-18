import hydra
import os
import json
import re
import time
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd

import torch
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt

from trainers.benchmarl_setup_experiment import benchmarl_setup_experiment

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
from torchrl.envs.utils import ExplorationType, set_exploration_type
from benchmarl.experiment import Experiment

@torch.no_grad()
def _evaluation_loop(self):
    evaluation_start = time.time()
    with set_exploration_type(
        ExplorationType.DETERMINISTIC
        if self.config.evaluation_deterministic_actions
        else ExplorationType.RANDOM
    ):
        if self.task.has_render(self.test_env) and self.config.render:
            video_frames = []

            def callback(env, td):
                video_frames.append(
                    self.task.__class__.render_callback(self, env, td)
                )

        else:
            video_frames = None
            callback = None

        if self.test_env.batch_size == ():
            rollouts = []
            for eval_episode in range(self.config.evaluation_episodes):
                rollouts.append(
                    self.test_env.rollout(
                        max_steps=self.max_steps,
                        policy=self.policy,
                        callback=callback if eval_episode == 0 else None,
                        auto_cast_to_device=True,
                        break_when_any_done=True,
                    )
                )
        else:
            rollouts = self.test_env.rollout(
                max_steps=self.max_steps,
                policy=self.policy,
                callback=callback,
                auto_cast_to_device=True,
                break_when_any_done=False,
                break_when_all_done=True,
                # We are running vectorized evaluation we do not want it to stop when just one env is done
            )
            rollouts = list(rollouts.unbind(0))
    evaluation_time = time.time() - evaluation_start
    self.logger.log(
        {"timers/evaluation_time": evaluation_time}, step=self.n_iters_performed
    )
    self.logger.log_evaluation(
        rollouts,
        video_frames=video_frames,
        step=self.n_iters_performed,
        total_frames=self.total_frames,
    )
    # Callback
    self._on_evaluation_end(rollouts)


def _safe_filename(text: str, max_len: int = 32) -> str:
    """Return a filesystem-safe slug based on the first *max_len* characters."""
    slug = re.sub(r"[^A-Za-z0-9_-]", "_", text[:max_len])
    return slug or "instruction"


NUM_ROLLOUTS = 1000


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Interactive evaluation loop that encodes an instruction, runs the BenchMARL
    experiment, and saves a *single* image that contains **both** the occupancy
    heat-map and the team-spread time-series side-by-side.
    """

    # ---------------------------------------------------------------------
    # 1) Static settings that never change inside the loop
    # ---------------------------------------------------------------------
    restore_path = (
        "/Users/nicolaspfitzer/ProrokLab/CustomScenarios/checkpoints/benchmarl/"
        "gnn_multi_agent/checkpoint_agent_level_targets.pt"
    )

    cfg.experiment.restore_file = restore_path
    cfg.experiment.save_folder = Path(os.path.dirname(os.path.realpath(__file__))) / "experiments"
    cfg.experiment.loggers[0] = "csv"
    print(Path(os.path.dirname(os.path.realpath(__file__))) / "experiments")
    cfg.experiment.render = True
    cfg.experiment.evaluation_episodes = NUM_ROLLOUTS
    cfg.task.params.done_at_termination = False
    
    #Experiment._evaluation_loop = _evaluation_loop

    print("Loaded Hydra config:\n" + OmegaConf.to_yaml(cfg, resolve=True))

    # Pre-load the sentence-encoder once
    llm = SentenceTransformer("thenlper/gte-large", device="cpu")

    # Prepare deterministic directories relative to project root
    root_dir = get_original_cwd()
    data_dir = os.path.join(root_dir, "data")
    plots_dir = os.path.join(root_dir, "plots")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 2) Interactive evaluation loop
    # ------------------------------------------------------------------
    eval_id = 0  # incremental counter for file names
    print("\nEnter instructions to evaluate (blank / 'quit' to stop).\n")

    while True:
        new_sentence = input("Instruction > ").strip()
        if new_sentence.lower() in {"quit", "q", "exit"}:
            print("Exiting evaluation loop.")
            break

        # --------------------------------------------------------------
        # Encode sentence -> embedding (1D tensor)
        # --------------------------------------------------------------
        try:
            if new_sentence == "":
                embedding = torch.zeros(llm.get_sentence_embedding_dimension(), device="cpu")
                print("Using zero embedding for empty instruction.")
            else:
                embedding = torch.tensor(llm.encode([new_sentence]), device="cpu").squeeze(0)
        except Exception as e:
            print(f"Failed to encode instruction: {e}")
            continue

        # --------------------------------------------------------------
        # Serialize to JSON (overwrite each run)
        # --------------------------------------------------------------
        json_path = os.path.join(data_dir, "evaluation_instruction.json")
        payload = {
            #"grid": [0.0] * 100,
            "gemini_response": new_sentence,
            "embedding": embedding.tolist(),
        }
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(payload, jf)
        print(f"Saved instruction & embedding → {json_path}")

        # Tell the task where the freshly-written JSON lives
        cfg.task.params.data_json_path = json_path

        # --------------------------------------------------------------
        # Build experiment and run evaluation
        # --------------------------------------------------------------
        experiment = benchmarl_setup_experiment(cfg)
        experiment.evaluate()

        # --------------------------------------------------------------
        # Extract team spread (per environment *per timestep*)
        # --------------------------------------------------------------
        # Shape assumed: (num_envs, num_steps)
        team_spread_raw = experiment.test_env.base_env._env.scenario.team_spread
        # Mask zero entries
        non_zero_mask = team_spread_raw > 0  # shape: (1000, 250)

        # Replace zeros with NaN (so we can ignore them in the mean)
        team_spread_nan = team_spread_raw.masked_fill(~non_zero_mask, float('nan'))  # shape: (1000, 250)

        # Compute mean across envs per timestep, ignoring NaNs
        team_spread_ts = torch.nanmean(team_spread_nan, dim=0)  # shape: (250,)

        # Compute overall average (optional)
        avg_team_spread = team_spread_ts.nanmean().item()

        # --------------------------------------------------------------
        # Extract grid-visit heat-map
        # --------------------------------------------------------------
        grid_visits = experiment.test_env.base_env._env.scenario.occupancy_grid.grid_visits
        mean_visits = grid_visits.float().mean(dim=0)  # shape (G, G)

        # --------------------------------------------------------------
        # Prepare continuous heat-map
        # --------------------------------------------------------------
        G = mean_visits.shape[0]             # original grid size
        raw = mean_visits.cpu().numpy()      # to NumPy
        sigma = 1.0                          # blur strength in grid cells
        blur = gaussian_filter(raw, sigma=sigma)

        fine = 400                           # resolution of the interpolated grid
        xi = yi = np.linspace(0, 1, fine)
        interp = RegularGridInterpolator(
            (np.linspace(0, 1, G), np.linspace(0, 1, G)),
            blur,
            bounds_error=False,
            fill_value=None,
        )
        Xf, Yf = np.meshgrid(xi, yi, indexing="ij")
        Zf = interp(np.stack([Xf.ravel(), Yf.ravel()], axis=-1)).reshape(fine, fine)

        # --------------------------------------------------------------
        # Combined figure: heat-map (left) + time series (right)
        # --------------------------------------------------------------
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 1) Heat-map
        cont = axes[0].contourf(xi, yi, Zf, levels=100, cmap="viridis", antialiased=True)
        axes[0].set_xlabel("X (normalised)")
        axes[0].set_ylabel("Y (normalised)")
        axes[0].set_title(
            f"Average visits — mean spread = {avg_team_spread:.2f}")
        fig.colorbar(cont, ax=axes[0], label="Visit count")

        # 2) Team-spread time-series
        timesteps = np.arange(team_spread_ts.shape[0])
        axes[1].plot(timesteps, team_spread_ts.cpu().numpy())
        axes[1].set_xlabel("Timestep")
        axes[1].set_ylabel("Team spread")
        axes[1].set_title(
            f"Team spread over time\n(mean across {team_spread_raw.shape[0]} envs)")

        # Overall annotation (instruction as caption)
        fig.text(0.5, -0.04, f"\u201c{new_sentence}\u201d", ha="center", fontsize=9)
        fig.tight_layout(rect=[0, 0.03, 1, 1])  # leave space for caption

        # --------------------------------------------------------------
        # Save the single combined image
        # --------------------------------------------------------------
        eval_id += 1

        def bool_tag(name: str, value: bool) -> str:
            return name if value else f"no_{name}"

        config_tag = "__".join([
            bool_tag("gnn", cfg.task.params.use_gnn),
            bool_tag("conv2d", cfg.task.params.use_conv_2d),
            bool_tag("confidence", cfg.task.params.use_confidence_data),
        ])
        ckpt_file = os.path.basename(cfg.experiment.restore_file)
        ckpt_name = os.path.splitext(ckpt_file)[0]

        # Build nested directory: plots/{ckpt_name}/{flags}/
        config_dir = os.path.join(plots_dir, ckpt_name, config_tag)
        os.makedirs(config_dir, exist_ok=True)

        fname_comb = f"visits_spread_{eval_id:03d}_{_safe_filename(new_sentence)}.png"
        out_path_comb = os.path.join(config_dir, fname_comb)
        fig.savefig(out_path_comb, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Combined figure saved to → {out_path_comb}\n")


if __name__ == "__main__":
    main()
