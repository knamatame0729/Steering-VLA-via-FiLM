"""Evaluate a trained VLA checkpoint on robosuite tasks."""

from __future__ import annotations

import argparse
import os

import imageio.v2 as imageio
import numpy as np
import torch

from envs.robosuite_env import RoboSuiteWrapper
from models.vla_diffusion_policy import VLADiffusionPolicy
from utils.tokenizer import SimpleTokenizer

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate VLA on robosuite")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--env-name", type=str, default="Lift")
    parser.add_argument("--robot", type=str, default="Panda")
    parser.add_argument("--controller", type=str, default="OSC_POSE")
    parser.add_argument("--camera-name", type=str, default="agentview")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--instruction", type=str, default="Pick up the red cube")
    parser.add_argument("--resize-to", type=int, default=84)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--video-dir", type=str, default="videos")
    parser.add_argument("--video-macro-block-size", type=int, default=1)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--reward-shaping", action="store_true")
    return parser.parse_args()


def load_model_and_tokenizer(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)

    vocab = ckpt["vocab"]
    state_dim = ckpt["state_dim"]
    action_dim = ckpt["action_dim"]
    d_model = ckpt["d_model"]
    diffusion_t = ckpt["diffusion_T"]

    vocab_size = max(vocab.values()) + 1
    model = VLADiffusionPolicy(
        vocab_size=vocab_size,
        state_dim=state_dim,
        action_dim=action_dim,
        d_model=d_model,
        diffusion_T=diffusion_t,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tokenizer = SimpleTokenizer(vocab=vocab)
    return model, tokenizer


def preprocess_image(img: np.ndarray, resize_to: int) -> np.ndarray:
    import cv2
    if img.shape[0] != resize_to or img.shape[1] != resize_to:
        img = cv2.resize(img, (resize_to, resize_to), interpolation=cv2.INTER_LINEAR)
    if img.dtype != np.uint8:
        img = (img.clip(0, 1) * 255).astype(np.uint8) if img.max() <= 1.0 \
            else img.clip(0, 255).astype(np.uint8)
    return img


def run_episode(model, env, text_ids, device, max_steps, resize_to):
    img, state, _ = env.reset()

    done = False
    step = 0
    total_reward = 0.0
    success = False
    frames = [img.copy()]

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

    while not done and step < max_steps and not success:
        img_proc = preprocess_image(img, resize_to)
        img_t = torch.from_numpy(img_proc).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        state_t = torch.from_numpy(state).float().unsqueeze(0)

        img_t = img_t.to(device)
        state_t = state_t.to(device)

        with torch.no_grad():
            action = model.act(img_t, text_ids, state_t)

        action_np = action.squeeze(0).cpu().numpy()
        img, state, reward, done, info = env.step(action_np)

        # reward = compute_phase_reward(info)

        total_reward += reward
        success = success or bool(info.get("success", False))
        frames.append(img.copy())
        step += 1

    return total_reward, step, success, frames


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model_and_tokenizer(args.checkpoint, device)

    token_ids = tokenizer.encode(args.instruction)
    text_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(device)

    env = RoboSuiteWrapper(
        env_name=args.env_name,
        robots=args.robot,
        seed=args.seed,
        controller=args.controller,
        camera_name=args.camera_name,
        image_height=args.resize_to,
        image_width=args.resize_to,
        horizon=args.max_steps,
        render=args.render,
        reward_shaping=args.reward_shaping,
    )

    if args.save_video:
        os.makedirs(args.video_dir, exist_ok=True)

    episode_rewards = []
    episode_success = []

    for ep in range(args.episodes):
        reward, steps, success, frames = run_episode(
            model=model,
            env=env,
            text_ids=text_ids,
            device=device,
            max_steps=args.max_steps,
            resize_to=args.resize_to,
        )
        episode_rewards.append(reward)
        episode_success.append(int(success))

        print(
            f"Episode {ep + 1}/{args.episodes} | reward={reward:.4f} | "
            f"steps={steps} | success={int(success)}"
        )

        if args.save_video:
            video_path = os.path.join(args.video_dir, f"robosuite_eval_ep{ep + 1}.mp4")
            with imageio.get_writer(
                video_path,
                fps=20,
                macro_block_size=args.video_macro_block_size,
            ) as writer:
                for frame in frames:
                    safe_frame = np.require(frame, dtype=np.uint8, requirements=["C", "A"])
                    writer.append_data(safe_frame)

    mean_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
    success_rate = float(np.mean(episode_success)) if episode_success else 0.0

    print(f"Mean reward: {mean_reward:.4f}")
    print(f"Success rate: {success_rate:.4f}")

    env.close()


if __name__ == "__main__":
    main()
