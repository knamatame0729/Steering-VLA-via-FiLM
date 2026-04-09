"""
Optimize FiLM gamma/beta parameters for VLA Diffusion Policy using ARS (Augmented Random Search).

ARS is a derivative-free optimization algorithm that:
1. Samples random directions in parameter space
2. Evaluates both positive and negative perturbations
3. Updates parameters in the direction of best-performing pairs
4. Is highly efficient in high-dimensional noisy settings

Perfect for discrete/noisy optimization (success/failure only).
"""

import os
import argparse
import numpy as np
import torch
import wandb
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
import time

from envs.metaworld_env import MetaWorldMT1Wrapper
from envs.ur10e_env import UR10ePickPlaceEnvV3
from models.vla_diffusion_policy import VLADiffusionPolicy
from utils.tokenizer import SimpleTokenizer

# Config
@dataclass
class OptimConfig:
    # model / env
    checkpoint:  str   = "checkpoints/fm_bottleneck_model.pt"
    env_name:    str   = "pick-place-v3"
    robot:       str   = "sawyer"
    seed:        int   = 42
    instruction: str   = "pick and place the object to the goal"
    device:      str   = "cpu"
    max_steps:   int   = 150

    # FiLM
    d_model: int = 16

    # evaluation
    eval_episodes: int = 20

    # ARS specific
    ars_iterations:    int   = 100    # Number of ARS iterations
    ars_num_directions: int  = 32     # Number of random directions per iteration
    ars_num_top_directions: int = 8   # Top directions to use for update
    ars_learning_rate:  float = 0.01  # Step size for parameter update
    ars_delta:         float = 2.0    # Perturbation magnitude
    ars_seed:          int   = 42

    # logging
    use_wandb:    bool = True
    project_name: str  = "FiLM_Optimization_ARS"
    output_dir:   str  = "optim_results"


# Model and Environment Setup

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
    if cfg.robot == "sawyer":
        return MetaWorldMT1Wrapper(
            env_name=cfg.env_name,
            seed=cfg.seed,
            render_mode="rgb_array",
            camera_name="corner2",
            #random_init=True,
        )
    return UR10ePickPlaceEnvV3(
        render_mode="rgb_array",
        camera_name="corner2",
        random_init=True,
    )


# 

def run_episode(model, env, text_ids, device, max_steps,
                gamma: torch.Tensor, beta: torch.Tensor) -> Tuple[float, bool]:
    """Run single episode and return (reward, success)."""
    img, state, _ = env.reset()
    gamma_t = gamma.unsqueeze(0).to(device)
    beta_t  = beta.unsqueeze(0).to(device)

    ep_reward, step, success, done = 0.0, 0, False, False
    while not done and step < max_steps:
        img_t   = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).div(255.0).to(device)
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)

        with torch.no_grad():
            action = model.act(img_t, text_ids, state_t, gamma_t, beta_t)

        img, state, reward, done, info = env.step(action.squeeze(0).cpu().numpy())
        # ep_reward += reward
        step += 1

        if info.get("success", False):
            success = True
            break

    return ep_reward, success, reward


# Objective function

def evaluate(params: np.ndarray, model, env, text_ids, device, cfg: OptimConfig,
             success_params_list: List[np.ndarray] = None) -> Tuple[float, int]:
    
    d = cfg.d_model
    gamma = torch.tensor(params[:d], dtype=torch.float32)
    beta  = torch.tensor(params[d:], dtype=torch.float32)

    successes = []
    for episode_idx in range(cfg.eval_episodes):
        try:
            r, s, _ = run_episode(model, env, text_ids, device, cfg.max_steps, gamma, beta)
            successes.append(float(s))
            
            # Track successful parameters
            if s and success_params_list is not None:
                success_params_list.append({
                    'params': params.copy(),
                    'gamma': gamma.numpy().copy(),
                    'beta': beta.numpy().copy(),
                    'episode_idx': episode_idx,
                })
        except Exception as e:
            print(f"    [WARNING] Episode {episode_idx} failed: {str(e)[:100]}")
            successes.append(0.0)

    success_count = int(np.sum(successes))
    loss = -r

    return loss, success_count


class ObjectiveFunction:
    """
    Objective function wrapper for ARS optimization.
    Tracks evaluation count and best parameters.
    Logs all successful parameter configurations.
    """
    def __init__(self, model, env, text_ids, device, cfg: OptimConfig):
        self.model = model
        self.env = env
        self.text_ids = text_ids
        self.device = device
        self.cfg = cfg
        self.evaluations = 0
        self.best_loss = float("inf")
        self.best_params = None
        self.best_success_count = 0
        self.history = []
        self.success_params_list = []  # Track all successful parameter sets

    def __call__(self, params: np.ndarray) -> Tuple[float, int]:
        """Return (loss, success_count) for ARS given params"""
        loss, success_count = evaluate(params, self.model, self.env, 
                                      self.text_ids, self.device, self.cfg,
                                      self.success_params_list)
        self.evaluations += 1

        # Track best params
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_params = params.copy()
            self.best_success_count = success_count

        # Append to history for logging
        self.history.append({
            'evaluations': self.evaluations,
            'loss': loss,
            'success_count': success_count,
        })

        return loss, success_count

# ARS Optimization

class AugmentedRandomSearch:
    """
    Augmented Random Search (ARS) for derivative-free optimization.
    
    Algorithm:
    1. Sample num_directions random directions in parameter space
    2. For each direction, evaluate both (+delta) and (-delta) perturbations
    3. Sort direction pairs by (loss_plus, loss_minus) performance
    4. Take top_num directions and apply positive parameter update
    5. Update parameters: params += learning_rate * sum(top_directions)
    
    Advantages for discrete/noisy optimization:
    - Only needs ranking information (robust to noise)
    - Efficient direction-based search
    - Natural parallelization across directions
    """
    
    def __init__(self, dim: int, learning_rate: float = 0.01, delta: float = 2.0, seed: int = 42):
        np.random.seed(seed)
        self.dim = dim
        self.learning_rate = learning_rate
        self.delta = delta
        
        # Initialize parameters: gamma=1, beta=0
        self.params = np.concatenate([
            np.ones(dim, dtype=np.float32),
            np.zeros(dim, dtype=np.float32),
        ])
        
        self.iteration = 0
    
    def sample_directions(self, num_directions: int) -> np.ndarray:
        """
        Sample random directions from standard normal.
        Shape: (num_directions, dim)
        """
        return np.random.normal(0, 1, (num_directions, self.dim * 2))
    
    def evaluate_direction_pair(self, direction: np.ndarray, 
                               objective_fn, delta: float) -> Tuple[float, float, float, float]:
        """
        Evaluate both +delta and -delta perturbations along direction.
        Returns: (loss_plus, loss_minus, success_plus, success_minus)
        """
        # Normalize direction to unit vector
        direction_normalized = direction / (np.linalg.norm(direction) + 1e-8)
        
        # Positive perturbation
        params_plus = self.params + delta * direction_normalized
        loss_plus, success_plus = objective_fn(params_plus)
        
        # Negative perturbation
        params_minus = self.params - delta * direction_normalized
        loss_minus, success_minus = objective_fn(params_minus)
        
        return loss_plus, loss_minus, success_plus, success_minus
    
    def update(self, directions: np.ndarray, direction_losses: List[Tuple[float, float]],
               num_top: int):
        """
        Update parameters using top performing directions.
        
        Strategy: Rank direction pairs by max(loss_plus, loss_minus)
        (best worst-case performance), then average top directions.
        """
        # Score each direction pair by the better of the two
        scores = []
        for i, (loss_plus, loss_minus) in enumerate(direction_losses):
            # Use the better (lower) loss
            better_loss = min(loss_plus, loss_minus)
            # Also consider the direction that achieved it
            if loss_plus < loss_minus:
                direction_sign = 1.0  # Use positive direction
            else:
                direction_sign = -1.0  # Use negative direction
            
            scores.append((better_loss, i, direction_sign))
        
        # Sort by loss (ascending) and take top_num
        scores.sort(key=lambda x: x[0])
        top_scores = scores[:num_top]
        
        # Compute update as average of top directions (with signs)
        update_vector = np.zeros_like(self.params)
        for better_loss, direction_idx, sign in top_scores:
            direction = directions[direction_idx]
            direction_normalized = direction / (np.linalg.norm(direction) + 1e-8)
            update_vector += sign * direction_normalized
        
        update_vector /= num_top
        
        # Apply update with learning rate
        self.params += self.learning_rate * update_vector
        
        self.iteration += 1
        
        # Return statistics
        best_loss = min(score[0] for score in scores)
        return best_loss


def run_ars(model, env, text_ids, device, cfg: OptimConfig) -> Tuple[np.ndarray, List[Dict]]:
    """
    ARS Optimization Loop
    Returns: (best_params, success_params_list)
    """
    print("\n" + "=" * 70)
    print("  AUGMENTED RANDOM SEARCH (ARS) OPTIMIZATION")
    print(f"  Iterations: {cfg.ars_iterations}")
    print(f"  Directions per iteration: {cfg.ars_num_directions}")
    print(f"  Top directions used: {cfg.ars_num_top_directions}")
    print(f"  Learning rate: {cfg.ars_learning_rate:.6f}")
    print(f"  Delta (perturbation): {cfg.ars_delta:.6f}")
    print(f"  Total Parameters: {cfg.d_model * 2}")
    print("=" * 70)

    # Initialize ARS
    ars = AugmentedRandomSearch(
        dim=cfg.d_model,
        learning_rate=cfg.ars_learning_rate,
        delta=cfg.ars_delta,
        seed=cfg.ars_seed
    )

    # Objective function wrapper
    objective = ObjectiveFunction(model, env, text_ids, device, cfg)

    # Main loop
    for iteration in range(cfg.ars_iterations):
        iter_start = time.time()
        
        # Sample random directions
        directions = ars.sample_directions(cfg.ars_num_directions)
        
        # Evaluate each direction pair
        direction_losses = []
        for direction_idx, direction in enumerate(directions):
            loss_plus, loss_minus, success_plus, success_minus = \
                ars.evaluate_direction_pair(direction, objective, cfg.ars_delta)
            direction_losses.append((loss_plus, loss_minus))
            
            # Print progress every 8 evaluations
            evals_in_iter = (direction_idx + 1) * 2  # 2 evals per direction
            if evals_in_iter % 16 == 0:
                print(f"  [iter {iteration + 1}] Evaluated {evals_in_iter}/{cfg.ars_num_directions * 2}", end='\r')
        
        # Update parameters using top directions
        best_loss = ars.update(directions, direction_losses, cfg.ars_num_top_directions)
        
        # Logging
        iter_time = time.time() - iter_start
        best_loss_overall = objective.best_loss
        
        print(
            f"[iter {iteration + 1:3d}/{cfg.ars_iterations}] "
            f"loss={best_loss:.4f} "
            f"success={-best_loss:.0f}/{cfg.eval_episodes} "
            f"lr={cfg.ars_learning_rate:.4f} "
            f"(evals={objective.evaluations}, {iter_time:.1f}s)"
        )

        if cfg.use_wandb:
            wandb.log({
                "iteration": iteration + 1,
                "loss_best_in_iter": best_loss,
                "loss_overall_best": best_loss_overall,
                "success_count": -best_loss,
                "success_count_best": objective.best_success_count,
                "evaluations": objective.evaluations,
                "iter_time": iter_time,
                "mean_param": ars.params.mean(),
                "std_param": ars.params.std(),
            })

    # Optimization finished
    print("\n" + "=" * 70)
    print(f"  Optimization finished!")
    print(f"  Best Loss: {objective.best_loss:.4f}")
    print(f"  Best Success Count: {objective.best_success_count}/{cfg.eval_episodes}")
    print(f"  Total Evaluations: {objective.evaluations}")
    print(f"  Total Iterations: {cfg.ars_iterations}")
    print(f"  Final Mean Gamma: {ars.params[:cfg.d_model].mean():.4f}")
    print(f"  Final Mean Beta: {ars.params[cfg.d_model:].mean():.4f}")
    print(f"  Total Successful Parameter Sets: {len(objective.success_params_list)}")
    print("=" * 70)

    return objective.best_params, objective.success_params_list


def report_results(best_params: np.ndarray, d_model: int, success_params_list: List[Dict]):
    """Print optimized parameters and all successful configurations"""
    gamma = best_params[:d_model]
    beta  = best_params[d_model:]
    
    print(f"\n{'=' * 70}")
    print(" BEST PARAMETERS:")
    gamma_dict = {f"{i}": round(float(gamma[i]), 4) for i in range(d_model)}
    beta_dict  = {f"{i}": round(float(beta[i]),  4) for i in range(d_model)}
    print(f'  GAMMA = {gamma_dict}')
    print(f'  BETA  = {beta_dict}')
    print(f"{'=' * 70}\n")
    
    # Print all successful parameter sets
    if success_params_list:
        print(f"\n{'=' * 70}")
        print(f" ALL SUCCESSFUL PARAMETER SETS ({len(success_params_list)} total)")
        print(f"{'=' * 70}\n")
        
        for idx, success_record in enumerate(success_params_list, 1):
            gamma_success = success_record['gamma']
            beta_success = success_record['beta']
            
            gamma_dict_success = {f"{i}": round(float(gamma_success[i]), 4) 
                                 for i in range(d_model)}
            beta_dict_success = {f"{i}": round(float(beta_success[i]), 4) 
                                for i in range(d_model)}
            
            print(f"[Success #{idx}]")
            print(f"  GAMMA = {gamma_dict_success}")
            print(f"  BETA  = {beta_dict_success}")
            print()
        
        print(f"{'=' * 70}\n")
    else:
        print(f"\nWARNING: No successful episodes found!\n")

# CLI

def parse_args():
    p = argparse.ArgumentParser(description="Optimize FiLM gamma/beta with ARS")
    p.add_argument("--checkpoint",      default="checkpoints/fm_bottleneck_model.pt")
    p.add_argument("--env-name",        default="pick-place-v3")
    p.add_argument("--robot",           default="sawyer", choices=["sawyer", "ur10e"])
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--device",          default="cpu")
    p.add_argument("--instruction",     default="pick and place the object to the goal")
    p.add_argument("--max-steps",       type=int,   default=150)
    p.add_argument("--eval-episodes",   type=int,   default=20)
    p.add_argument("--ars-iterations",  type=int,   default=100)
    p.add_argument("--ars-num-directions", type=int, default=32)
    p.add_argument("--ars-num-top-directions", type=int, default=8)
    p.add_argument("--ars-learning-rate", type=float, default=0.01)
    p.add_argument("--ars-delta",       type=float, default=2.0)
    p.add_argument("--no-wandb",        action="store_true")
    return p.parse_args()


# Main
def main():
    args = parse_args()
    
    # Seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cfg = OptimConfig(
        checkpoint      = args.checkpoint,
        env_name        = args.env_name,
        robot           = args.robot,
        seed            = args.seed,
        device          = str(device),
        instruction     = args.instruction,
        max_steps       = args.max_steps,
        eval_episodes   = args.eval_episodes,
        ars_iterations  = args.ars_iterations,
        ars_num_directions = args.ars_num_directions,
        ars_num_top_directions = args.ars_num_top_directions,
        ars_learning_rate = args.ars_learning_rate,
        ars_delta       = args.ars_delta,
        use_wandb       = not args.no_wandb,
    )

    if cfg.use_wandb:
        wandb.init(
            entity="kaitos_projects",
            project=cfg.project_name,
            job_type="film_optimization_ars",
            config={
                **vars(cfg),
                "total_params": cfg.d_model * 2,
                "optimizer": "ARS (Augmented Random Search)",
                "evals_per_iteration": cfg.ars_num_directions * 2,
            },
            tags=["film_optimization", "ars", cfg.env_name],
        )

    model, tokenizer = load_model_and_tokenizer(cfg.checkpoint, device)
    text_ids = torch.tensor(
        tokenizer.encode(cfg.instruction), dtype=torch.long
    ).unsqueeze(0).to(device)

    env = make_env(cfg)

    print(f"\nOptimizing {cfg.d_model * 2} params (gamma×{cfg.d_model} + beta×{cfg.d_model})")
    print(f"Objective : maximize success rate over {cfg.eval_episodes} episodes/eval")
    print(f"Optimizer : ARS (Augmented Random Search)")
    print(f"Directions: {cfg.ars_num_directions} per iteration, "
          f"top {cfg.ars_num_top_directions} used for update")
    print(f"Learning rate: {cfg.ars_learning_rate}")
    print(f"Evaluations per iteration: {cfg.ars_num_directions * 2}")

    # Run ARS optimization
    best_params, success_params_list = run_ars(model, env, text_ids, device, cfg)
    
    # Print final results (including all successful parameters)
    report_results(best_params, cfg.d_model, success_params_list)

    env.close()
    if cfg.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()