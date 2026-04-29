"""
Optimize FiLM gamma/beta parameters for VLA Diffusion Policy using Nevergrad.
"""

import os
import argparse
import json
import numpy as np
import torch
import wandb
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
import nevergrad as ng
import time

from envs.robosuite_env import RoboSuiteWrapper
from models.vla_diffusion_policy import VLADiffusionPolicy
from utils.tokenizer import SimpleTokenizer

# Config
@dataclass
class OptimConfig:
    # model / env
    checkpoint:  str   = "checkpoints/fm_bottleneck_model_v2.pt"
    env_name:    str   = "Lift"
    robot:       str   = "Panda"
    controller:  str   = "OSC_POSE"
    camera_name: str   = "agentview"
    seed:        int   = 42
    instruction: str   = "Pick up the cube"
    device:      str   = "cpu"
    max_steps:   int   = 400
    resize_to:   int   = 84
    reward_shaping: bool = False

    # FiLM
    state_dim: int = 128

    # evaluation
    eval_episodes: int = 20

    # Nevergrad
    ng_budget: int = 5000

    # logging
    use_wandb:    bool = True
    project_name: str  = "FiLM_Optimization_Nevergrad"
    output_dir:   str  = "optim_results"

# Model and Env

def load_model_and_tokenizer(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = VLADiffusionPolicy(
        vocab_size  = max(ckpt["vocab"].values()) + 1,
        state_dim   = ckpt["state_dim"],
        action_dim  = ckpt["action_dim"],
        d_model     = ckpt["d_model"],
        diffusion_T = ckpt["diffusion_T"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, SimpleTokenizer(vocab=ckpt["vocab"])


def make_env(cfg: OptimConfig):
    return RoboSuiteWrapper(
        env_name=cfg.env_name,
        robots=cfg.robot,
        seed=cfg.seed,
        controller=cfg.controller,
        camera_name=cfg.camera_name,
        image_height=cfg.resize_to,
        image_width=cfg.resize_to,
        horizon=cfg.max_steps,
        reward_shaping=cfg.reward_shaping,
    )

# Single episode runner

def run_episode(model, env, text_ids, device, max_steps,
                gamma: torch.Tensor, beta: torch.Tensor) -> Tuple[bool, float]:
    try:
        img, state, _ = env.reset()
        gamma_t = gamma.unsqueeze(0).to(device)
        beta_t  = beta.unsqueeze(0).to(device)

        step = 0
        total_reward = 0.0
        success = False
        done = False

        max_r_reach = 0.0
        max_r_grasp = 0.0
        max_r_lift  = 0.0
        max_r_hover = 0.0

        def compute_phase_reward(info):
            reward = 0.0

            # ===== reach =====
            if "gripper_dist" in info:
                d = info["gripper_dist"]
                reward += np.exp(-5 * d)   # 0~1

            # ===== grasp =====
            if info.get("grasped", False):
                reward += 1.0

            # ===== lift =====
            if "cube_z" in info:
                lift = info["cube_z"] - 0.8
                reward += np.clip(lift * 10, 0, 1.0)

            # ===== success =====
            if info.get("success", False):
                reward += (max_steps - step) * 1.5

            return reward
        
        while not done and step < max_steps:
            img_t   = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).div(255.0).to(device)
            state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
            
            with torch.no_grad():
                action = model.act(img_t, text_ids, state_t, gamma_t, beta_t)
            
            img, state, _, done, info = env.step(action.squeeze(0).cpu().numpy())

            r_reach, r_grasp, r_lift, r_hover = env.env.staged_rewards()

            max_r_reach = max(max_r_reach, r_reach)
            max_r_grasp = max(max_r_grasp, r_grasp)
            max_r_lift  = max(max_r_lift,  r_lift)
            max_r_hover = max(max_r_hover, r_hover)

            # reward = compute_phase_reward(info)

            # total_reward += float(reward)
            step += 1
            
            if info.get("success", False):
                success = True
                done = True
                break

        if success:
            total_reward = 10.0
        else:
            w_reach = 1.5  / 0.1
            w_grasp = 1.0  / 0.35
            w_lift  = 1.5  / 0.5
            w_hover = 2.0  / 0.7

            total_reward = (
                max_r_reach * w_reach
              + max_r_grasp * w_grasp
              + max_r_lift  * w_lift
              + max_r_hover * w_hover
            )

        return success, total_reward
    
    except Exception as e:
        print(f"[ERROR] run_episode failed: {str(e)[:100]}")
        return False, 0.0


# Objective function
def evaluate(params: np.ndarray, cfg: OptimConfig,
             model, env, text_ids, device) -> Tuple[float, int]:
    """
    Evaluate parameters over multiple episodes.
    """
    d = cfg.state_dim
    # gamma = torch.tensor(params[:d], dtype=torch.float32)
    # beta  = torch.tensor(params[d:], dtype=torch.float32)

    params_t = torch.from_numpy(params).float().to(device)
    gamma = params_t[:d]
    beta  = params_t[d:]

    successes = 0
    rewards = []
    
    for _ in range(cfg.eval_episodes):
        success, reward = run_episode(model, env, text_ids, device, cfg.max_steps, gamma, beta)
        rewards.append(reward)
        if success:
            successes += 1
 
    mean_reward = np.mean(rewards)

    loss = -mean_reward

    return loss, successes


# Nevergrad Optimization

class ObjectiveFunction:
    def __init__(self, cfg: OptimConfig):
        self.cfg = cfg
        self.evaluations        = 0
        self.best_loss          = float("inf")
        self.best_params        = None
        self.best_success_count = 0
        self.history            = []
        self.success_params_list = []
 
    def record(self, loss: float, success_count: int, params: np.ndarray):
        self.evaluations += 1
 
        if success_count > 0:
            self.success_params_list.append({
                "params":        params.copy(),
                "success_count": success_count,
                "loss":          loss,
                "reward":        -loss,
            })
 
        if loss < self.best_loss:
            self.best_loss          = loss
            self.best_params        = params.copy()
            self.best_success_count = success_count
 
        if success_count >= 1:
            d = self.cfg.state_dim
            gamma_dict = {f"{i}:{i+1}": round(float(params[i]),   6) for i in range(d)}
            beta_dict  = {f"{i}:{i+1}": round(float(params[d+i]), 6) for i in range(d)}
            print(f"\nsuccess_count={success_count}/{self.cfg.eval_episodes}")
            print(f"  gamma = {gamma_dict}")
            print(f"  beta  = {beta_dict}\n")
 
        self.history.append({
            "evaluations":   self.evaluations,
            "loss":          float(loss),
            "success_count": int(success_count),
        })
 
        if self.cfg.use_wandb:
            wandb.log({
                "eval/loss":          float(loss),
                "eval/success_count": int(success_count),
                "evaluations":        self.evaluations,
            })


def run_optim(model, env, text_ids, device, cfg: OptimConfig) -> Tuple[np.ndarray, List[Dict], List[Dict]]:

    print("\n" + "=" * 70)
    print("  NEVERGRAD OPTIMIZATION STARTED")
    print(f"  Budget:     {cfg.ng_budget}")
    print(f"  Parameters: {cfg.state_dim * 2}")
    print("=" * 70)

    # Initial params
    x0 = np.concatenate([
        np.ones(cfg.state_dim,  dtype=np.float32),   # gamma
        np.zeros(cfg.state_dim, dtype=np.float32),   # beta
    ])

    param = ng.p.Array(init=x0).set_bounds(-2, 2)
    optimizer = ng.optimizers.NGOpt(parametrization=param, budget=cfg.ng_budget, num_workers=1)

    # print(type(optimizer.optim))     
    # print(optimizer.optim.name) 
    # if hasattr(optimizer.optim, 'optim'):
    #     print(type(optimizer.optim.optim)) 

    # Objective function wrapper
    objective = ObjectiveFunction(cfg)

    for evaluated in range(cfg.ng_budget):
        candidate = optimizer.ask()

        try:
            loss, success_count = evaluate(candidate.value, cfg, model, env, text_ids, device)
        except Exception as e:
            print(f"Error occurred while evaluating candidate: {e}")
            loss, success_count = 0.0, 0

        # Update optimizer with evaluated candidate
        noisy_loss = loss + np.random.normal(0, 1e-6)
        optimizer.tell(candidate, noisy_loss)

        objective.record(loss, success_count, candidate.value)

        print(
            f"[eval {evaluated + 1:5d}/{cfg.ng_budget}] "
            f"best_reward={-objective.best_loss:.4f}  "
            f"best_success={objective.best_success_count}/{cfg.eval_episodes}"
        )
 
        if cfg.use_wandb:
            wandb.log({
                "loss":          loss,
                "reward":        -loss,
                "success_count": success_count,
                "evaluations":   objective.evaluations,
            })

    recommendation = optimizer.provide_recommendation()
    best_params = recommendation.value

    if objective.best_params is None:
        objective.best_params = best_params

    print("\n" + "=" * 70)
    print(f"  Optimization finished!")
    print(f"  Best Reward:        {-objective.best_loss:.4f}")
    print(f"  Best Success Count: {objective.best_success_count}/{cfg.eval_episodes}")
    print(f"  Total Evaluations:  {objective.evaluations}")
    print(f"  Successful Sets:    {len(objective.success_params_list)}")
    print("=" * 70)

    return objective.best_params, objective


def report_results(best_params: np.ndarray, state_dim: int, objective: ObjectiveFunction):

    gamma = best_params[:state_dim]
    beta  = best_params[state_dim:]
    
    print(f"\n{'=' * 70}")
    print(" BEST PARAMETERS:")
    gamma_dict = {f"{i}:{i+1}": float(gamma[i]) for i in range(state_dim)}
    beta_dict  = {f"{i}:{i+1}": float(beta[i]) for i in range(state_dim)}
    print(f'  gamma = {gamma_dict}')
    print(f'  beta  = {beta_dict}')
    print(f"{'=' * 70}\n")
    
    # Print all successful parameter sets
    if objective.success_params_list:
        print(f"\n{'=' * 70}")
        print(f" ALL SUCCESSFUL PARAMETER SETS ({len(objective.success_params_list)} total)")
        print(f"{'=' * 70}\n")
        
        for idx, success_record in enumerate(objective.success_params_list, 1):
            params = success_record["params"]
            gamma_success = params[:state_dim]
            beta_success = params[state_dim:]
            
            gamma_dict_success = {f"{i}:{i+1}": round(float(gamma_success[i]), 4) 
                                 for i in range(state_dim)}
            beta_dict_success = {f"{i}:{i+1}": round(float(beta_success[i]), 4) 
                                for i in range(state_dim)}
            
            print(f"[Success #{idx}]")
            print(f"  gamma = {gamma_dict_success}")
            print(f"  beta  = {beta_dict_success}")
            print()
        
        print(f"{'=' * 70}\n")

def parse_args():
    parser = argparse.ArgumentParser(description="Optimize FiLM gamma/beta with CMA-ES")
    parser.add_argument("--checkpoint",        default="checkpoints/model.pt")
    parser.add_argument("--env-name",          default="Lift")
    parser.add_argument("--robot",             default="Panda")
    parser.add_argument("--controller",        default="OSC_POSE")
    parser.add_argument("--camera-name",       default="agentview")
    parser.add_argument("--resize-to",         type=int,   default=84)
    parser.add_argument("--seed",              type=int,   default=42)
    parser.add_argument("--device",            default="cpu")
    parser.add_argument("--instruction",       default="Pick up the cube")
    parser.add_argument("--max-steps",         type=int,   default=150)
    parser.add_argument("--eval-episodes",     type=int,   default=20)
    parser.add_argument("--ng-budget",         type=int,   default=2000)
    parser.add_argument("--reward-shaping",    action="store_true")
    parser.add_argument("--output-dir",        default="optim_results")
    parser.add_argument("--no-wandb",          action="store_true")
    return parser.parse_args()


# Main

def main():
    args = parse_args()
    
    # Seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Device setup
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    cfg = OptimConfig(
        checkpoint        = args.checkpoint,
        env_name          = args.env_name,
        robot             = args.robot,
        controller        = args.controller,
        camera_name       = args.camera_name,
        seed              = args.seed,
        device            = str(device),
        instruction       = args.instruction,
        max_steps         = args.max_steps,
        resize_to         = args.resize_to,
        reward_shaping    = args.reward_shaping,
        eval_episodes     = args.eval_episodes,
        ng_budget         = args.ng_budget,
        use_wandb         = not args.no_wandb,
    )

    if cfg.use_wandb:
        wandb.init(
            entity="kaitos_projects",
            project=cfg.project_name,
            config={
                "eval_episodes": cfg.eval_episodes,
                "ng_budget": cfg.ng_budget,
            }
        )

    model, tokenizer = load_model_and_tokenizer(cfg.checkpoint, device)
    text_ids = torch.tensor(
        tokenizer.encode(cfg.instruction), dtype=torch.long
    ).unsqueeze(0).to(device)

    env = make_env(cfg)

    try:
        # Run CMA-ES optimization
        best_params, objective = run_optim(model, env, text_ids, device, cfg)

        # Report and save results
        report_results(best_params, cfg.state_dim, objective)

    finally:
        env.close()
        if cfg.use_wandb:
            wandb.finish()


if __name__ == "__main__":
    main()