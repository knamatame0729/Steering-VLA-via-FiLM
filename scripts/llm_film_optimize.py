"""Test VLA Diffusion Policy on Meta-World MT1 with Manual FiLM Parameters"""

import os
import argparse
import numpy as np
import tempfile
import torch
import imageio.v2 as imageio
import wandb
from typing import Tuple, List, Dict, Optional

from envs.metaworld_env import MetaWorldMT1Wrapper
from models.vla_diffusion_policy import VLADiffusionPolicy
from utils.tokenizer import SimpleTokenizer

from .llm_film_generator import LLMFiLMGenerator


def parse_args():
    parser = argparse.ArgumentParser(description="Test VLA Diffusion Policy on Meta-World MT1")

    parser.add_argument("--checkpoint", type=str, default="checkpoints/fm_bottleneck_model.pt")
    parser.add_argument("--env-name", type=str, default="pick-place-v3")
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


def run_episode_with_film(model, env, text_ids, device, max_steps,
                gamma: torch.Tensor, beta: torch.Tensor, episode_num: int, save_video: bool) -> Tuple[bool, float]:

    try:

        img, state, info = env.reset()
        gamma_t = gamma.unsqueeze(0).to(device)
        beta_t = beta.unsqueeze(0).to(device)

        step = 0
        reward = 0.0
        frames = [img.copy()]
        success = False
        done = False

        max_tcp_to_obj_reward = 0.0
        max_object_grasped = 0.0
        max_lift_reward = 0.0
        max_move_reward = 0.0
        max_in_place = 0.0
        max_in_place_and_obj_grasped = 0.0

        # print(f"\n{'='*60}")
        # print(f"Episode {episode_num + 1}: Running with FiLM params")
        # print(f"  gamma: {gamma.cpu().numpy()}")
        # print(f"  beta:  {beta.cpu().numpy()}")
        # print(f"{'='*60}\n")

        while not done and step < max_steps:
            img_t = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0 # (1, 3, H, W)
            state_t = torch.from_numpy(state).float().unsqueeze(0)

            # Move to device
            img_t = img_t.to(device)
            state_t = state_t.to(device)

            with torch.no_grad():
                action = model.act(img_t, text_ids, state_t, gamma_t, beta_t)

            img, state, reward, done, info = env.step(action.squeeze(0).cpu().numpy())
            step += 1

            max_tcp_to_obj_reward = max(max_tcp_to_obj_reward, info.get("tcp_to_obj_reward", 0.0))
            max_object_grasped = max(max_object_grasped, info.get("grasp_reward", 0.0))
            max_lift_reward = max(max_lift_reward, info.get("lift_reward", 0.0))
            max_move_reward = max(max_move_reward, info.get("move_reward", 0.0))
            max_in_place = max(max_in_place, info.get("in_place", 0.0))
            max_in_place_and_obj_grasped = max(max_in_place_and_obj_grasped, info.get("in_place_and_object_grasped", 0.0))

            if save_video:
                frames.append(img.copy())

            # Check for success
            if info.get('success', False):
                success = True
                done = True
                break

        if success:
            reward = 10.0

        else:
            reward = (
            max_tcp_to_obj_reward        * 2.0 +
            max_object_grasped           * 1.0 +
            max_lift_reward              * 2.0 +
            max_move_reward              * 1.0 +
            max_in_place                 * 0 +
            max_in_place_and_obj_grasped * 0
        )
        
        return success, -reward + 10.0, frames
    
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

    # environment
    if args.robot == "sawyer":
        env = MetaWorldMT1Wrapper(
            env_name=args.env_name,
            seed=args.seed,
            render_mode="rgb_array",
            camera_name="corner2",
            random_init=True,
        )
    elif args.robot == "ur10e":
        env = UR10ePickPlaceEnvV3(
            render_mode="rgb_array",
            camera_name="corner",
            seed=args.seed,
            random_init=False,
        )


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