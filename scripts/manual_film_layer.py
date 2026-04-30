"""Test VLA Diffusion Policy on robosuite with manual FiLM parameters."""

import os
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

# Nevergrad Robosuite can pick and place Panda
base_override = {
    "gamma": {'0:1': 0.444391, '1:2': 1.408402, '2:3': 1.486834, '3:4': 1.017262, '4:5': 1.882958, '5:6': 1.374099, '6:7': 0.015547, '7:8': 0.968614, '8:9': 1.383011, '9:10': 1.543726, '10:11': 0.939651, '11:12': 1.723332, '12:13': 1.176755, '13:14': 0.791018, '14:15': 1.795022, '15:16': 1.783102, '16:17': 0.294015, '17:18': 1.558944, '18:19': 1.200336, '19:20': 0.24327, '20:21': 1.685317, '21:22': 1.862088, '22:23': 0.812899, '23:24': 0.671482, '24:25': 1.591432, '25:26': 0.738139, '26:27': 0.552359, '27:28': 1.853145, '28:29': 0.60839, '29:30': 1.020484, '30:31': 0.995159, '31:32': 1.985576, '32:33': 0.080408, '33:34': 1.325785, '34:35': 1.747553, '35:36': 1.013214, '36:37': 0.559608, '37:38': 1.061264, '38:39': -0.629949, '39:40': 1.80327, '40:41': 0.569027, '41:42': 1.456275, '42:43': 1.035713, '43:44': 1.321767, '44:45': 1.334959, '45:46': 0.094788, '46:47': 1.535331, '47:48': 1.533486, '48:49': 0.882652, '49:50': 1.84445, '50:51': 1.975378, '51:52': 1.027607, '52:53': -0.0803, '53:54': 1.435709, '54:55': 1.176788, '55:56': 1.395586, '56:57': 1.327553, '57:58': 0.346231, '58:59': 0.769946, '59:60': 1.208695, '60:61': 0.334214, '61:62': 0.999077, '62:63': 1.002261, '63:64': 1.623109, '64:65': 1.325572, '65:66': -0.194985, '66:67': 0.102898, '67:68': 1.707458, '68:69': 1.212945, '69:70': 0.998722, '70:71': 1.728165, '71:72': 0.577078, '72:73': 1.556615, '73:74': 0.869714, '74:75': 1.09442, '75:76': 1.029847, '76:77': 1.790363, '77:78': 1.186317, '78:79': 1.9812, '79:80': 0.80656, '80:81': 0.614803, '81:82': 1.439466, '82:83': 1.803501, '83:84': 1.890823, '84:85': 1.965927, '85:86': 0.662261, '86:87': 0.606171, '87:88': 1.531408, '88:89': -0.046477, '89:90': 1.080651, '90:91': 0.929253, '91:92': 0.460951, '92:93': 1.151005, '93:94': 1.485827, '94:95': 0.338297, '95:96': 0.428071, '96:97': 0.850596, '97:98': 1.241888, '98:99': 1.647494, '99:100': 0.70885, '100:101': 0.353435, '101:102': 1.652853, '102:103': 0.809789, '103:104': 0.817148, '104:105': 0.928834, '105:106': -0.398734, '106:107': 1.09812, '107:108': 1.58822, '108:109': 1.073236, '109:110': 0.703175, '110:111': 0.924661, '111:112': 0.496162, '112:113': 1.292388, '113:114': 1.756879, '114:115': 1.412517, '115:116': 1.507705, '116:117': 1.886168, '117:118': 0.108938, '118:119': 0.594789, '119:120': 1.526845, '120:121': 1.11717, '121:122': 1.254616, '122:123': 0.675119, '123:124': 1.440101, '124:125': 1.105471, '125:126': 0.929888, '126:127': 0.585609, '127:128': 0.684056},
    "beta": {'0:1': 1.387744, '1:2': -0.365724, '2:3': -0.20335, '3:4': -0.819139, '4:5': -0.275832, '5:6': -0.747271, '6:7': 0.710091, '7:8': -0.364501, '8:9': -0.567022, '9:10': 0.156126, '10:11': 0.461148, '11:12': -0.763712, '12:13': -0.34814, '13:14': -0.938396, '14:15': -1.577682, '15:16': -0.414001, '16:17': -0.241521, '17:18': -0.460656, '18:19': -0.510164, '19:20': 0.514375, '20:21': -0.148597, '21:22': 0.417693, '22:23': 0.394736, '23:24': 0.225342, '24:25': -0.209907, '25:26': -0.109203, '26:27': 0.118919, '27:28': 0.799201, '28:29': -0.43485, '29:30': -0.484212, '30:31': 0.652652, '31:32': 0.227932, '32:33': 0.655069, '33:34': -1.408086, '34:35': 0.092811, '35:36': 0.237556, '36:37': -0.747308, '37:38': -0.044892, '38:39': 0.338987, '39:40': 0.043432, '40:41': 0.495793, '41:42': -0.01004, '42:43': -0.290987, '43:44': -1.634331, '44:45': -0.255775, '45:46': -0.639104, '46:47': 0.166459, '47:48': 0.018593, '48:49': 0.471878, '49:50': 0.6424, '50:51': 0.366804, '51:52': 0.879923, '52:53': 0.754724, '53:54': 0.419399, '54:55': 0.215512, '55:56': 0.470981, '56:57': 0.148377, '57:58': -0.612774, '58:59': 0.039093, '59:60': -0.252054, '60:61': -0.603436, '61:62': 0.074317, '62:63': -0.465048, '63:64': 0.594348, '64:65': -0.263217, '65:66': -0.562857, '66:67': 0.238452, '67:68': -0.755338, '68:69': 0.355124, '69:70': 0.66477, '70:71': 0.426109, '71:72': 0.077684, '72:73': -0.353887, '73:74': 0.642622, '74:75': 0.691148, '75:76': 0.763478, '76:77': -1.081213, '77:78': 0.164929, '78:79': 0.014243, '79:80': 1.124171, '80:81': 0.469817, '81:82': 0.353388, '82:83': -0.484755, '83:84': 0.138763, '84:85': -1.179072, '85:86': -0.470581, '86:87': -0.482402, '87:88': -0.27275, '88:89': -0.143869, '89:90': -0.517568, '90:91': -0.065252, '91:92': 0.32455, '92:93': -0.12702, '93:94': 0.078367, '94:95': 0.583306, '95:96': 0.304892, '96:97': 0.079941, '97:98': 0.02897, '98:99': 0.404915, '99:100': -0.460919, '100:101': 0.420792, '101:102': 0.26671, '102:103': 0.522484, '103:104': -0.886971, '104:105': 0.178829, '105:106': 0.234833, '106:107': 0.406903, '107:108': -0.143076, '108:109': -0.249203, '109:110': -0.267936, '110:111': -0.391876, '111:112': -0.926441, '112:113': -0.699703, '113:114': -1.050514, '114:115': -0.609895, '115:116': -0.639009, '116:117': -1.463392, '117:118': 0.564896, '118:119': 0.202541, '119:120': -0.32086, '120:121': -0.813635, '121:122': 0.627845, '122:123': 0.125501, '123:124': -0.054512, '124:125': -0.044311, '125:126': 0.121093, '126:127': -0.666668, '127:128': 0.60495}
}


# Baseline 
# base_override = {"gamma": {},            "beta": {}}

FILM_CONFIG = {
    "default_gamma": 1.0,
    "default_beta": 0.0,
    "episode_overrides": {
        **{i: copy.deepcopy(base_override) for i in range(0, 101)}
    },
}

def parse_args():
    parser = argparse.ArgumentParser(description="Test VLA with FiLM parameters")

    parser.add_argument("--checkpoint", type=str, default="checkpoints/model.pt")
    parser.add_argument("--env-name", type=str, default="Lift")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--instruction", type=str, default="Pick up the cube")
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
    gamma, beta = get_film_params(128, episode_num)
    gamma = gamma.to(device)
    beta = beta.to(device)

    # Run Episode with Modulation
    ep_reward, step, frames, last_action, final_img, final_state, success, pos_array, actions_array, obj_init_pos = run_episode_with_modulation(
        model, env, text_ids, device, args.max_steps, gamma, beta, args.model_image_size, args.save_video, episode_num
    )
    
    print(f"Episode {episode_num+1} is Done")

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
    env = RoboSuiteWrapper(
        env_name=args.env_name,
        robots=args.robot,
        seed=args.seed,
        controller=args.controller,
        camera_name=args.camera_name,
        image_height=args.capture_image_size,
        image_width=args.capture_image_size,
        horizon=args.max_steps,
        render=args.render,
    )

    results = []

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