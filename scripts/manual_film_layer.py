"""Test VLA Diffusion Policy on robosuite with manual FiLM parameters."""

import os
import argparse
import numpy as np
import torch
import imageio.v2 as imageio
import wandb
import io
import copy

from envs.robosuite_env import RoboSuiteWrapper
from models.vla_diffusion_policy import VLADiffusionPolicy
from utils.tokenizer import SimpleTokenizer
from .logger import FiLMExperimentLogger

base_override = {
    "gamma": {'0:1': 0.974761, '1:2': -0.166018, '2:3': 0.657034, '3:4': 1.366024, '4:5': 1.037103, '5:6': 1.656685, '6:7': 0.700308, '7:8': 0.644144, '8:9': 1.531631, '9:10': 0.074814, '10:11': 1.604239, '11:12': 1.371114, '12:13': 1.931459, '13:14': 0.505926, '14:15': 1.029417, '15:16': 0.594804, '16:17': 1.531708, '17:18': -0.076529, '18:19': 0.919027, '19:20': 1.356935, '20:21': 1.48073, '21:22': 0.830708, '22:23': 1.247039, '23:24': 0.146841, '24:25': 1.258725, '25:26': 1.269758, '26:27': 1.209357, '27:28': 0.210152, '28:29': 0.807469, '29:30': 1.485614, '30:31': 1.059823, '31:32': 0.231076, '32:33': 0.821081, '33:34': 1.260756, '34:35': 0.798095, '35:36': 0.479097, '36:37': 0.716202, '37:38': 1.272115, '38:39': 1.14571, '39:40': 0.136808, '40:41': 1.199289, '41:42': 0.624391, '42:43': 0.279269, '43:44': 0.945541, '44:45': 0.814434, '45:46': 1.376238, '46:47': 1.799775, '47:48': 1.787688, '48:49': 0.974839, '49:50': 1.115872, '50:51': 0.476754, '51:52': 0.727073, '52:53': 1.943496, '53:54': 1.336998, '54:55': 0.860813, '55:56': 1.093039, '56:57': 0.937404, '57:58': 1.783875, '58:59': 1.076877, '59:60': 1.306193, '60:61': 1.201288, '61:62': 0.307505, '62:63': 0.494619, '63:64': 0.673181, '64:65': 0.940918, '65:66': 1.220214, '66:67': 0.649294, '67:68': 0.821254, '68:69': 1.725446, '69:70': 0.142546, '70:71': 0.959539, '71:72': 0.92595, '72:73': 0.755442, '73:74': 1.247666, '74:75': 0.71231, '75:76': 0.703301, '76:77': 1.965668, '77:78': 1.098225, '78:79': 1.008843, '79:80': 1.662981, '80:81': 1.06513, '81:82': 0.678735, '82:83': 1.285091, '83:84': -0.206878, '84:85': 1.228965, '85:86': 1.15409, '86:87': 1.196708, '87:88': 0.768941, '88:89': 0.689299, '89:90': 0.785333, '90:91': 0.385776, '91:92': 0.944251, '92:93': 0.560887, '93:94': 0.819794, '94:95': 0.58337, '95:96': 0.967473, '96:97': 1.412781, '97:98': 1.149807, '98:99': 0.460747, '99:100': 1.095842, '100:101': 0.869428, '101:102': 0.18868, '102:103': 0.41477, '103:104': 1.684207, '104:105': 0.64667, '105:106': 1.059862, '106:107': 0.344904, '107:108': 1.730066, '108:109': 1.924806, '109:110': 0.915392, '110:111': 1.198666, '111:112': 0.910513, '112:113': 1.117453, '113:114': 0.829882, '114:115': 0.956187, '115:116': 1.435622, '116:117': 1.083084, '117:118': 0.961868, '118:119': 1.250799, '119:120': 1.222432, '120:121': 0.630104, '121:122': 0.482158, '122:123': 0.850242, '123:124': 0.717736, '124:125': 0.420021, '125:126': 0.648591, '126:127': 1.238053, '127:128': 1.229606},
    "beta": {'0:1': -0.125082, '1:2': 0.362287, '2:3': 0.34132, '3:4': -0.048079, '4:5': -0.125614, '5:6': 0.293364, '6:7': 0.246199, '7:8': -0.19502, '8:9': -0.34815, '9:10': -0.436344, '10:11': -0.770779, '11:12': 0.893375, '12:13': -0.102608, '13:14': -0.560349, '14:15': 0.030554, '15:16': 0.059386, '16:17': 0.030828, '17:18': 0.201679, '18:19': 0.459675, '19:20': -0.179454, '20:21': -0.013642, '21:22': 0.169016, '22:23': -0.037085, '23:24': 0.442031, '24:25': 0.123721, '25:26': 0.318397, '26:27': 0.189171, '27:28': 0.328747, '28:29': 0.703988, '29:30': 0.002763, '30:31': 0.35675, '31:32': 0.021355, '32:33': 1.061722, '33:34': 0.095365, '34:35': 0.638995, '35:36': 0.333622, '36:37': 0.93357, '37:38': -0.307407, '38:39': 0.183905, '39:40': -0.294013, '40:41': -0.15476, '41:42': -0.000459, '42:43': 0.345402, '43:44': 0.159805, '44:45': -0.076108, '45:46': 0.061383, '46:47': 0.331526, '47:48': -0.244393, '48:49': -0.764657, '49:50': -0.190886, '50:51': 0.174843, '51:52': 0.119436, '52:53': 0.114013, '53:54': -0.271326, '54:55': -0.005604, '55:56': 0.620635, '56:57': 0.111474, '57:58': 0.262727, '58:59': 0.55049, '59:60': 0.373674, '60:61': -0.280953, '61:62': 0.079454, '62:63': 0.006606, '63:64': -0.170597, '64:65': 0.126034, '65:66': -0.464818, '66:67': -0.234129, '67:68': 0.390457, '68:69': 0.351376, '69:70': 0.137724, '70:71': -0.961666, '71:72': 0.280909, '72:73': -0.228974, '73:74': -0.366543, '74:75': 0.229706, '75:76': 0.077604, '76:77': 0.406542, '77:78': 0.646644, '78:79': 0.085366, '79:80': 0.935449, '80:81': 0.198316, '81:82': -0.168176, '82:83': -0.547482, '83:84': -0.632582, '84:85': 0.383485, '85:86': -0.35209, '86:87': -0.217635, '87:88': 0.174412, '88:89': -0.292916, '89:90': 0.030641, '90:91': 0.567358, '91:92': 0.250661, '92:93': 0.265177, '93:94': 0.108757, '94:95': 0.140057, '95:96': -0.22073, '96:97': -0.323295, '97:98': -0.002011, '98:99': 0.389874, '99:100': 0.483086, '100:101': -0.006741, '101:102': -0.531472, '102:103': 0.241708, '103:104': -0.275169, '104:105': -0.62495, '105:106': 0.158742, '106:107': -0.223986, '107:108': -0.603968, '108:109': -0.358489, '109:110': 0.103971, '110:111': 0.344581, '111:112': -0.160749, '112:113': 0.006777, '113:114': 0.20837, '114:115': 0.52077, '115:116': 0.0251, '116:117': -0.214819, '117:118': -0.926055, '118:119': 0.09543, '119:120': -0.645288, '120:121': 0.323818, '121:122': -0.285087, '122:123': -0.274485, '123:124': 0.027508, '124:125': -0.006697, '125:126': 0.041951, '126:127': -0.385808, '127:128': -0.288403}
}

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

    parser.add_argument("--checkpoint", type=str, default="checkpoints/model.pt")
    parser.add_argument("--env-name", type=str, default="Lift")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=150)
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
    gamma, beta = get_film_params(128, episode_num)
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
        project="Manual_FiLM_VLA_Testing_CNN",
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