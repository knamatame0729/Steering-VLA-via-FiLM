"""Test VLA Diffusion Policy on Meta-World MT1 with Manual FiLM Parameters"""

import os
import argparse
import numpy as np
import tempfile
import torch
import imageio.v2 as imageio
import wandb
from typing import Tuple, List, Dict, Optional

from envs.robosuite_env import RoboSuiteWrapper
from models.vla_diffusion_policy import VLADiffusionPolicy
from utils.tokenizer import SimpleTokenizer

from .llm_film_generator import LLMFiLMGenerator


def parse_args():
    parser = argparse.ArgumentParser(description="Test VLA Diffusion Policy on Meta-World MT1")

    parser.add_argument("--checkpoint", type=str, default="checkpoints/can_model_v2.pt")
    parser.add_argument("--robot", type=str, default="sawyer")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--instruction", type=str, default="pick and place the object to the goal")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--video-dir", type=str, default="videos")
    parser.add_argument("--optimum-reward", type=float, default=0.0)
    parser.add_argument("--llm-model", type=str, default="google/gemma-3-4b-it")
    parser.add_argument("--prompt-id", type=int, default=1)
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--eval-episodes",  type=int, default=20)
    parser.add_argument("--env-name", type=str, default="CanPickAndPlace")
    parser.add_argument("--controller", type=str, default="OSC_POSE")
    parser.add_argument("--camera-name", type=str, default="agentview")
    parser.add_argument("--resize-to", type=int, default=84)
    parser.add_argument("--reward-shaping", action="store_true")
    parser.add_argument("--render", action="store_true")

    return parser.parse_args()


def load_model_and_tokenizer(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)

    vocab = ckpt["vocab"]
    state_dim = ckpt["state_dim"]
    action_dim = ckpt["action_dim"]
    d_model = ckpt["d_model"]
    diffusion_T = ckpt["diffusion_T"]

    vocab_size = max(vocab.values()) + 1

    model = VLADiffusionPolicy(
        vocab_size=vocab_size,
        state_dim=state_dim,
        action_dim=action_dim,
        d_model=d_model,
        diffusion_T=diffusion_T,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tokenizer = SimpleTokenizer(vocab=vocab)

    return model, tokenizer

def make_env(cfg):
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
        render=cfg.render,
    )

def run_episode_with_film(model, env, text_ids, device, max_steps,
                gamma: torch.Tensor, beta: torch.Tensor, episode_num: int, save_video: bool) -> Tuple[bool, float, List]:

    try:

        img, state, info = env.reset()
        gamma_t = gamma.unsqueeze(0).to(device)
        beta_t = beta.unsqueeze(0).to(device)

        step = 0
        reward = 0.0
        frames = [img.copy()]
        success = False
        done = False

        max_r_reach = 0.0
        max_r_grasp = 0.0
        max_r_lift  = 0.0
        max_r_hover = 0.0

        while not done and step < max_steps:
            img_t = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0 # (1, 3, H, W)
            state_t = torch.from_numpy(state).float().unsqueeze(0)

            # Move to device
            img_t = img_t.to(device)
            state_t = state_t.to(device)

            with torch.no_grad():
                action = model.act(img_t, text_ids, state_t, gamma_t, beta_t)

            img, state, reward, done, info = env.step(action.squeeze(0).cpu().numpy())

            r_reach, r_grasp, r_lift, r_hover = env.env.staged_rewards()

            max_r_reach = max(max_r_reach, r_reach)
            max_r_grasp = max(max_r_grasp, r_grasp)
            max_r_lift  = max(max_r_lift,  r_lift)
            max_r_hover = max(max_r_hover, r_hover)

            step += 1

            if save_video:
                frames.append(img.copy())

            # Check for success
            if info.get('success', False):
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

        
        return success, -total_reward + 10.0, frames
    
    except Exception as e:
        print(f"[ERROR] run_episode failed: {str(e)[:100]}")
        return False, 0.0, []

def run_episode(args, model, env, text_ids, device, film_generator, episode_num):

    gamma, beta, reasoning = film_generator.generate_film_params(
        instruction=args.instruction,
        episode_num=episode_num,
        total_episodes=args.episodes,
        device=device,
        prompt_id=args.prompt_id
    )

    eval_rewards = []
    eval_successes = []
    all_frames = []

    # Run Episode with FiLM parameters
    for eval_ep in range(args.eval_episodes):
        success, ep_reward, frames = run_episode_with_film(
            model, env, text_ids, device, max_steps=args.max_steps, gamma=gamma, beta=beta, 
            save_video=(args.save_video and eval_ep == 0), episode_num=episode_num
        )
        eval_rewards.append(ep_reward)
        eval_successes.append(success)
        if args.save_video and eval_ep == 0:
            all_frames = frames

    mean_reward = float(np.mean(eval_rewards))
    success_count = int(np.sum(eval_successes))

    # Add episode result to history for future episodes
    film_generator.add_episode_result(
        episode_num=episode_num,
        gamma=gamma.cpu().numpy(),
        beta=beta.cpu().numpy(),
        total_reward=mean_reward,
        success=success_count
    )

    # Log LLM reasoning to W&B
    wandb.log({f"eval/reasoning": wandb.Html(f"<pre>{reasoning}</pre>")}, step=episode_num + 1)

    # Save and Log video
    if args.save_video and all_frames:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
            with imageio.get_writer(tmp.name, fps=args.video_fps) as writer:
                for frame in all_frames:
                    frame_rot = np.rot90(frame, 2)
                    writer.append_data(frame_rot)

            wandb.log({
                f"eval/video": wandb.Video(tmp.name, format="mp4")
            }, step=episode_num + 1)
    
    return mean_reward, success_count


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    cfg = argparse.Namespace(
        env_name=args.env_name,
        robot=args.robot,
        seed=args.seed,
        controller=args.controller,
        camera_name=args.camera_name,
        resize_to=args.resize_to,
        max_steps=args.max_steps,
        reward_shaping=args.reward_shaping,
        render=args.render,
    )

    # Initialize W&B for evaluation
    wandb.init(
        project="LLM_FiLM_Optimization",
        config={
            "checkpoint": args.checkpoint,
            "env_name": args.env_name,
            "seed": args.seed,
            "episodes": args.episodes,
            "max_steps": args.max_steps,
            "instruction": args.instruction,
            "optimum_reward": args.optimum_reward,
        },
    )

    model, tokenizer = load_model_and_tokenizer(args.checkpoint, device)

    # encode instruction
    text_tokens = tokenizer.encode(args.instruction)
    text_ids = torch.tensor(text_tokens, dtype=torch.long).unsqueeze(0).to(device)

    env = make_env(cfg)

    # Initialize LLM-based FiLM generator
    print(f"Initializing LLM FiLM generator with model: {args.llm_model}")
    film_generator = LLMFiLMGenerator(
        bottleneck_dim=16,
        model_name=args.llm_model,
        device=args.device,
        optimum_reward=args.optimum_reward,
    )

    # Run evaluation episodes
    all_rewards = []
    all_successes = []

    # Run evaluation episodes
    for ep in range(args.episodes):
        mean_reward, success_count = run_episode(
            args, model=model, env=env, text_ids=text_ids, device=device, 
            film_generator=film_generator, episode_num=ep,
        )
        
        all_rewards.append(mean_reward)
        all_successes.append(success_count)

        # Log episode results to W&B
        wandb.log({
            "eval/reward": mean_reward,
            "eval/episode": ep + 1,
            "eval/success": success_count,
        }, step=ep + 1)
        
    env.close()
    wandb.finish()


if __name__ == "__main__":
    main()