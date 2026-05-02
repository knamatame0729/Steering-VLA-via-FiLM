"""Test VLA Diffusion Policy on robosuite with manual FiLM parameters."""

import argparse
import numpy as np
import torch
import imageio.v2 as imageio
import wandb
import copy

from envs.robosuite_env import RoboSuiteWrapper
from models.vla_diffusion_policy import VLADiffusionPolicy
from utils.tokenizer import SimpleTokenizer
from .logger import FiLMExperimentLogger

# Sawyer wider range good FiLM params
# base_override = {
#     "gamma": {'0:1': 0.917339, '1:2': 0.962252, '2:3': 1.010938, '3:4': 1.232241, '4:5': 1.064154, '5:6': 0.632821, '6:7': 1.341464, '7:8': 1.106445, '8:9': 0.557592, '9:10': 1.01909, '10:11': 0.660356, '11:12': 0.983418, '12:13': 0.773278, '13:14': 1.319914, '14:15': 0.818635, '15:16': 1.124007},
#     "beta": {'0:1': 0.146912, '1:2': -0.00281, '2:3': -0.055409, '3:4': -0.146731, '4:5': -0.185106, '5:6': -0.066671, '6:7': 0.193678, '7:8': -0.596643, '8:9': 0.17281, '9:10': 0.291882, '10:11': 0.262151, '11:12': 0.372038, '12:13': 0.03398, '13:14': 0.036068, '14:15': 0.331704, '15:16': -0.36434}
# }

# CMA-ES Best Parameters
# base_override = {
#         "gamma": {'0:1': 1.0036, '1:2': 0.8894, '2:3': 0.9556, '3:4': 0.9628, '4:5': 0.9176, '5:6': 0.9778, '6:7': 1.1042, '7:8': 1.0148, '8:9': 1.0244, '9:10': 1.099, '10:11': 1.1944, '11:12': 0.9939, '12:13': 0.9917, '13:14': 0.9579, '14:15': 1.1068, '15:16': 1.1728},
#         "beta": {'0:1': 0.0953, '1:2': 0.0604, '2:3': 0.082, '3:4': 0.0132, '4:5': -0.0537, '5:6': -0.0812, '6:7': -0.1486, '7:8': 0.1255, '8:9': 0.036, '9:10': -0.0352, '10:11': 0.1463, '11:12': -0.1022, '12:13': 0.005, '13:14': 0.0829, '14:15': 0.1154, '15:16': 0.0211}
#     }


# CMA-ES Robosuite
# base_override = {
#     "gamma": {'0:1': 1.010116, '1:2': 1.081593, '2:3': 0.963348, '3:4': 1.090903, '4:5': 1.038758, '5:6': 1.027652, '6:7': 1.011701, '7:8': 0.987574, '8:9': 1.060027, '9:10': 1.007018, '10:11': 0.901647, '11:12': 0.944141, '12:13': 0.990708, '13:14': 1.0155, '14:15': 0.997171, '15:16': 1.060949},
#     "beta": {'0:1': -0.097555, '1:2': 0.007179, '2:3': -0.090899, '3:4': 0.037986, '4:5': -0.004723, '5:6': 0.02098, '6:7': -0.043194, '7:8': 0.063973, '8:9': 0.052086, '9:10': 0.029177, '10:11': -0.006476, '11:12': 0.028993, '12:13': -0.035345, '13:14': 0.042778, '14:15': 0.082474, '15:16': 0.053531}
# }

# CMA-ES Robosuite Sawyer
base_override = {
    "gamma": {'0:1': 0.9887045531259142, '1:2': 0.8638239563765379, '2:3': 1.2846456526915289, '3:4': 1.4119723672213613, '4:5': 1.1325799631744147, '5:6': 1.1273397332200847, '6:7': 1.4338174352414264, '7:8': 0.9482503758236042, '8:9': 1.1429599472535021, '9:10': 1.5845725422320436, '10:11': 0.40185026114344913, '11:12': 1.1989657921091084, '12:13': 1.32758105679682, '13:14': 1.4795456846931783, '14:15': 1.2657945704356395, '15:16': 0.454004694266782},
    "beta": {'0:1': -0.0015019347561489047, '1:2': -0.6023327674335669, '2:3': 0.015362766652881598, '3:4': -0.22700152339138965, '4:5': -0.5314224120428316, '5:6': 0.13828734242513332, '6:7': 0.24184001697600727, '7:8': 0.14549648680328622, '8:9': -0.3654152140981087, '9:10': 0.03915875287309238, '10:11': -0.14060181687163328, '11:12': -0.12485619777980436, '12:13': 0.27932450080854937, '13:14': 0.08606328468194936, '14:15': -0.14350313951707722, '15:16': -0.3141984600883442}
}

# CMA-ES Button Press
# base_override = {
#     "gamma": {'0:1': 0.814441, '1:2': 1.482282, '2:3': 0.646283, '3:4': 1.265097, '4:5': 0.739964, '5:6': 1.013566, '6:7': 1.645152, '7:8': 0.86746, '8:9': 1.251768, '9:10': 0.927698, '10:11': 0.381518, '11:12': 1.144052, '12:13': 0.15139, '13:14': 0.601108, '14:15': 1.817781, '15:16': 1.182714},
#     "beta": {'0:1': 0.307467, '1:2': -0.019153, '2:3': 0.690268, '3:4': -0.19925, '4:5': 0.236203, '5:6': -0.232972, '6:7': 0.823434, '7:8': -0.439126, '8:9': -0.289881, '9:10': 0.30246, '10:11': 0.737407, '11:12': 0.673626, '12:13': 0.138845, '13:14': -0.100873, '14:15': 0.28158, '15:16': -0.392525}
# }

# base_override = {"gamma": {},            "beta": {}}

FILM_CONFIG = {
    "default_gamma": 1.0,
    "default_beta": 0.0,
    "episode_overrides": {
        # 0:  {"gamma": {},            "beta": {}},  # Baseline
        **{i: copy.deepcopy(base_override) for i in range(0, 101)}
    },
}

def parse_args():
    parser = argparse.ArgumentParser(description="Test VLA Diffusion Policy on robosuite")

    parser.add_argument("--checkpoint", type=str, default="checkpoints/can_model_v2.pt")
    parser.add_argument("--env-name", type=str, default="PickPlaceCan")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--instruction", type=str, default="Pick up the can and place it")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--video-dir", type=str, default="videos")
    parser.add_argument("--video-macro-block-size", type=int, default=1)
    parser.add_argument("--controller", type=str, default="OSC_POSE")
    parser.add_argument("--camera-name", type=str, default="agentview")
    parser.add_argument("--model-image-size", type=int, default=84)
    parser.add_argument("--capture-image-size", type=int, default=256)
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--video-crf", type=int, default=18)
    parser.add_argument("--robot", type=str, default="Panda", choices=["Sawyer", "Panda"])
    parser.add_argument("--reward-shaping", action="store_true")
    parser.add_argument("--render", action="store_true")

    return parser.parse_args()


def load_model_and_tokenizer(checkpoint_path: str, device: torch.device):
    """Load the trained VLA model and tokenizer."""
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

def _parse_slice(s: str) -> slice:
    parts = s.split(":")
    if len(parts) == 1:
        i = int(parts[0])
        return slice(i, i + 1)
    return slice(*[int(p) if p else None for p in parts])


def _build_array(d_model: int, default: float, spec) -> np.ndarray:
    arr = np.full(d_model, default, dtype=np.float32)
    if isinstance(spec, (int, float)): 
        arr[:] = float(spec)
    elif isinstance(spec, dict):  
        for key, val in spec.items():
            arr[_parse_slice(key)] = float(val)
    return arr

def get_film_params(d_model, episode_num):

    cfg = FILM_CONFIG.copy()
    override = cfg["episode_overrides"].get(episode_num, {})

    gamma_arr = _build_array(d_model, cfg["default_gamma"], override.get("gamma", cfg["default_gamma"]))
    beta_arr  = _build_array(d_model, cfg["default_beta"],  override.get("beta",  cfg["default_beta"]))

    g_changed = np.where(gamma_arr != cfg["default_gamma"])[0]
    b_changed = np.where(beta_arr  != cfg["default_beta"])[0]
    return torch.tensor(gamma_arr), torch.tensor(beta_arr)


def preprocess_image_for_model(img: np.ndarray, resize_to: int) -> np.ndarray:
    import cv2
    if img.shape[0] != resize_to or img.shape[1] != resize_to:
        img = cv2.resize(img, (resize_to, resize_to), interpolation=cv2.INTER_AREA)
    if img.dtype != np.uint8:
        img = (img.clip(0, 1) * 255).astype(np.uint8) if img.max() <= 1.0 else img.clip(0, 255).astype(np.uint8)
    return img


def run_episode_with_modulation(model, env, text_ids, device, max_steps, gamma, beta, model_image_size, save_video=False, episode_num=0):
    """
    Run a single episode using the full diffusion model with gamma/beta modulation.
    """
    img, state, info = env.reset()
    obj_init_pos = info.get("obj_init_pos") if isinstance(info, dict) else None

    # print(f"  [Episode {episode_num}] obj_init_pos: {unwrapped.obj_init_pos}, _target_pos: {unwrapped._target_pos}")
    step = 0
    ep_reward = 0.0
    frames = [img.copy()]
    last_action = None
    success = False
    pos = []
    actions_list = []

    # Move gamma and beta to device
    gamma = gamma.to(device).unsqueeze(0)  # (1, d_model)
    beta = beta.to(device).unsqueeze(0)    # (1, d_model)

    done = False
    while not done and step < max_steps:
        img_model = preprocess_image_for_model(img, model_image_size)
        img_t = torch.from_numpy(img_model).permute(2, 0, 1).float().unsqueeze(0) / 255.0 # (1, 3, H, W)
        state_t = torch.from_numpy(state).float().unsqueeze(0)

        # Move to device
        img_t = img_t.to(device)
        state_t = state_t.to(device)

        # Inference action with diffusion
        with torch.no_grad():
            diffusion_action = model.act(img_t, text_ids, state_t, gamma, beta)  # (1, action_dim)

        # print(f" Step {step}:")
        # print(f" Modulated action: {diffusion_action.squeeze(0).cpu().numpy()}")
        
        last_action = diffusion_action.clone()
        action_np = diffusion_action.squeeze(0).cpu().numpy()
        actions_list.append(action_np.copy())

        img, state, reward, done, info = env.step(action_np)
        # print(f" State: {state[:3]}")
        ep_reward = reward
        step += 1
        frames.append(img.copy())

        # Append x, y, z position of the end-effector for visualization
        pos.append(state[:3].copy())

        pos_array = np.array(pos)
        actions_array = np.array(actions_list)

        # Check for success
        if info.get('success', False):
            success = True
            done = True

    actions_array = np.array(actions_list) if actions_list else np.array([])
    # Return episode results
    return ep_reward, step, frames, last_action, img, state, success, pos_array, actions_array, obj_init_pos


def run_modulated_episode(args, model, env, text_ids, device, episode_num):
    """
    Run one episode with gamma/beta modulation applied to every diffusion action.
    """
    print(f"Episode {episode_num+1}/{args.episodes}")
    
    # Get FiLM parameters
    gamma, beta = get_film_params(16, episode_num)
    gamma = gamma.to(device)
    beta = beta.to(device)

    # Run Episode with Modulation
    ep_reward, step, frames, last_action, final_img, final_state, success, pos_array, actions_array, obj_init_pos = run_episode_with_modulation(
        model, env, text_ids, device, args.max_steps, gamma, beta, args.model_image_size, args.save_video, episode_num
    )
    
    print(f"Episode {episode_num+1} is Done")
    # Use FiLMExperimentLogger for comprehensive logging
    # episode_data = {
    #     "positions": pos_array,
    #     "actions": actions_array,
    #     "gamma": gamma.cpu().numpy(),
    #     "beta": beta.cpu().numpy(),
    #     "success": success,
    #     "reward": ep_reward,
    #     "steps": step,
    #     "layer_target": FILM_CONFIG["layer_target"],
    # }
    # logger.log_episode(episode_data, episode_num)

    obj_metrics = {}
    if obj_init_pos is not None and len(obj_init_pos) >= 2:
        obj_x, obj_y = float(obj_init_pos[0]), float(obj_init_pos[1])
        obj_init_str = f"({obj_x:.4f}, {obj_y:.4f})"
        obj_metrics = {
            "eval/obj_init_x": obj_x,
            "eval/obj_init_y": obj_y,
            "eval/obj_init_pos": wandb.Html(f"<pre>{obj_init_str}</pre>"),
        }

    # Log video
    if args.save_video:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
            with imageio.get_writer(
                tmp.name,
                fps=args.video_fps,
                macro_block_size=args.video_macro_block_size,
                codec="libx264",
                ffmpeg_params=["-crf", str(args.video_crf), "-preset", "slow", "-pix_fmt", "yuv420p"],
            ) as writer:
                for f in frames:
                    f_rot = np.rot90(f, 2)
                    # Keep frame memory contiguous and aligned for ffmpeg swscale.
                    safe_frame = np.require(f_rot, dtype=np.uint8, requirements=["C", "A"])
                    writer.append_data(safe_frame)

            wandb.log({
                "Episode": episode_num,
                "eval/video": wandb.Video(tmp.name, format="mp4"),
                "eval/reward": ep_reward,
                "eval/success": int(success),
                **obj_metrics,
            }, step=episode_num)
    else:
        wandb.log({"eval/reward": ep_reward,
                   "eval/success": int(success),
                   **obj_metrics,
                   }, step=episode_num)

    
    return ep_reward, step, success


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Initialize W&B for evaluation
    wandb.init(
        entity="kaitos_projects",
        project="Manual_FiLM_VLA_Testing",
        config={
            "env_name": args.env_name,
            "episodes": args.episodes,
            "max_steps": args.max_steps,
            "model_image_size": args.model_image_size,
            "capture_image_size": args.capture_image_size,
            "video_fps": args.video_fps,
            "video_crf": args.video_crf,
        },
    )

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.checkpoint, device)

    # encode instruction
    text_tokens = tokenizer.encode(args.instruction)
    text_ids = torch.tensor(text_tokens, dtype=torch.long).unsqueeze(0).to(device)

    # environment
    if args.robot == "Sawyer":
        env = RoboSuiteWrapper(
            env_name=args.env_name,
            robots="Sawyer",
            seed=args.seed,
            controller=args.controller,
            camera_name=args.camera_name,
            image_height=args.capture_image_size,
            image_width=args.capture_image_size,
            horizon=args.max_steps,
            render=args.render,
        )
    elif args.robot == "Panda":
        env = RoboSuiteWrapper(
            env_name=args.env_name,
            robots="Panda",
            seed=args.seed,
            controller=args.controller,
            camera_name=args.camera_name,
            image_height=args.capture_image_size,
            image_width=args.capture_image_size,
            horizon=args.max_steps,
            render=args.render,
        )

    # Initialize the FiLM experiment logger
    # logger = FiLMExperimentLogger(project_name="Manual_FiLM_VLA_Testing")

    results = []

    #env.reset_episode_count()

    # Run evaluation episodes
    for ep in range(args.episodes):
        
        reward, steps, success = run_modulated_episode(args, model, env, text_ids, device, ep)
        results.append((success))

    n_success = sum(results)
    wandb.log({"eval/baseline": n_success})
    print(f"Final Results: {n_success}/{args.episodes} successful episodes.")

    env.close()
    wandb.finish()


if __name__ == "__main__":
    main()