from argparse import ArgumentParser, BooleanOptionalAction
from operator import add
from typing import Dict, Union

import hydra
import json
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

import numpy as np
import torch

from vmas.make_env import make_env
from vmas.simulator.environment.gym import GymWrapper
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import save_video
from sentence_transformers import SentenceTransformer

from pathlib import Path
import os

from trainers.benchmarl_setup_experiment import benchmarl_setup_experiment
from scenarios.sequential_tasks.multi_agent_sequential import MyLanguageScenario
from scenarios.sequential_tasks.scripts.rnn_model import EventRNN
class SequentialTaskEnv:
    """ Use this script to run the sequential task environment. """

    def __init__(self, experiment_config, rnn_path: str, save_render_path: str, render: bool = True, device: str = "cpu"):
 
        experiment = benchmarl_setup_experiment(cfg=experiment_config)
        self.policy = experiment.policy
        self.sequential_model = EventRNN.load_from_checkpoint(rnn_path, map_location=device).eval()
        
        self.scenario = experiment.test_env.base_env._env.scenario
        self.env: GymWrapper = make_env(scenario=self.scenario, render=render)
        self.env.render()
        self.n_agents = self.env.unwrapped.n_agents
        self.agents = self.env.unwrapped.agents
        self.reset = False
        
        # Models
        self.policy = experiment.policy
        self.sequential_model = EventRNN.load_from_checkpoint(rnn_path, map_location=device).eval()
        
        self.render = render
        self.save_render_path = save_render_path
        self.frame_list = []
        
    def run(self, num_episodes: int = 1, max_num_steps: int = 500): # Terminate after max 500 steps per episode
        """ Run the environment for a number of episodes. """
        
        total_rew = [0] * self.n_agents
        for episode in range(num_episodes):
            for step in range(max_num_steps+1):
                
                if self.reset:
                    
                    save_video(
                        self.frame_list,
                        os.path.join(self.save_render_path, f"episode_{episode}.mp4"),
                        fps=1 / self.env.unwrapped.world.dt,
                    )
                    self.env.reset()
                    self.reset = False
                    self.frame_list = []
                    total_rew = [0] * self.n_agents
                
                action = self.env.action_space.sample()
                obs, rew, done, info = self.env.step(action)
                total_rew = list(map(add, total_rew, rew))
                
                if done or step == max_num_steps:
                    self.reset = True
                    
                # obs = self.env.reset()
                # done = False
                # while not done:
                #     action = self.env.action_space.sample()  # Random action
                #     obs, reward, done, info = self.env.step(action)
                #     if self.render:
                #         self.env.render()
                # print(f"Episode {episode + 1} finished with reward: {reward}")

NUM_ROLLOUTS = 1

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    
    restore_path = (
        "/Users/nicolaspfitzer/ProrokLab/CustomScenarios/checkpoints/benchmarl/"
        "gnn_multi_agent/checkpoint_agent_level_targets.pt"
    )
    
    cfg.experiment.restore_file = restore_path# full merged config
    cfg.experiment.save_folder = Path(os.path.dirname(os.path.realpath(__file__))) / "experiments"
    cfg.experiment.render = True
    cfg.experiment.evaluation_episodes = NUM_ROLLOUTS
    cfg.task.params.done_at_termination = False
    
    print("Loaded Hydra config:\n" + OmegaConf.to_yaml(cfg, resolve=True))

    # Pre-load the sentence-encoder once
    llm = SentenceTransformer("thenlper/gte-large", device="cpu")
    
    # Prepare deterministic directories relative to project root
    root_dir = get_original_cwd()
    data_dir = os.path.join(root_dir, "data")
    
    # ------------------------------------------------------------------
    # 2) Interactive evaluation loop
    # ------------------------------------------------------------------
    eval_id = 0  # incremental counter for file names
    print("\nEnter instructions to run task (blank / 'quit' to stop).\n")
    
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
    
    sequential_task_env = SequentialTaskEnv(
        experiment_config=cfg.experiment,
        rnn_path=restore_path,
        save_render_path=os.path.join(cfg.experiment.save_folder, "render"),
        render=cfg.experiment.render,
        device="cpu"
    )
    
    sequential_task_env.run(num_episodes=NUM_ROLLOUTS)

if __name__ == "__main__":
    main()