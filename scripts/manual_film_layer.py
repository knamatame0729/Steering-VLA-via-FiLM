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


# CMA-ES Robosuite cube pick up
# base_override = {
#     "gamma": {'0:1': 1.010116, '1:2': 1.081593, '2:3': 0.963348, '3:4': 1.090903, '4:5': 1.038758, '5:6': 1.027652, '6:7': 1.011701, '7:8': 0.987574, '8:9': 1.060027, '9:10': 1.007018, '10:11': 0.901647, '11:12': 0.944141, '12:13': 0.990708, '13:14': 1.0155, '14:15': 0.997171, '15:16': 1.060949},
#     "beta": {'0:1': -0.097555, '1:2': 0.007179, '2:3': -0.090899, '3:4': 0.037986, '4:5': -0.004723, '5:6': 0.02098, '6:7': -0.043194, '7:8': 0.063973, '8:9': 0.052086, '9:10': 0.029177, '10:11': -0.006476, '11:12': 0.028993, '12:13': -0.035345, '13:14': 0.042778, '14:15': 0.082474, '15:16': 0.053531}
# }

# CMA-ES Robosuite Sawyer
# base_override = {
#     "gamma": {'0:1': 1.050425076057512, '1:2': 0.6495178806021719, '2:3': 1.151852718010805, '3:4': 1.9266932199831186, '4:5': 0.48570232790961887, '5:6': 1.1771438576039182, '6:7': 0.9616357714088363, '7:8': 1.409517885960958, '8:9': 1.816547666611413, '9:10': 1.2319048456908352, '10:11': 0.2434153125982882, '11:12': 1.4571344826671981, '12:13': 1.0909898448275468, '13:14': 1.5626547288834718, '14:15': 1.9074143388294207, '15:16': 0.7314061731992814},
#     "beta": {'0:1': -0.1661938380152983, '1:2': 0.39533398325966224, '2:3': -1.6162004949876931, '3:4': 0.29496454103637, '4:5': 0.09168976758133944, '5:6': -0.6660833578683718, '6:7': 0.5652163989564604, '7:8': 0.9269425234143532, '8:9': 0.3227178097659556, '9:10': -0.18634543776703766, '10:11': 0.10121336933064344, '11:12': 0.42625971829136367, '12:13': -0.8546715394307984, '13:14': 0.6109644469083851, '14:15': 0.3894591975938798, '15:16': 3.350397385273873e-05}
# }

# base_override = {
#     "gamma": {'0:1': 1.0854422922163836, '1:2': -1.1666607570904608, '2:3': -0.14128417427209605, '3:4': 1.0709071602010773, '4:5': 1.0562370328831143, '5:6': 0.6195288906322446, '6:7': 1.6020043011762266, '7:8': 0.8102203999577735, '8:9': 0.592074855716811, '9:10': 1.0417592883875675, '10:11': 1.7522755462534951, '11:12': 1.4212619200116854, '12:13': 1.0380225563330259, '13:14': 1.9041983991244864, '14:15': 0.4149163846488284, '15:16': 0.5297952573156472},
#     "beta": {'0:1': -0.14231543406998973, '1:2': 0.9016957954061697, '2:3': -1.3344869575843608, '3:4': -0.3225731802137936, '4:5': 0.5846562370423151, '5:6': 1.2184869286410676, '6:7': 0.7141084382537771, '7:8': 0.6267705221395774, '8:9': -0.9270974405774178, '9:10': -0.0969397448140588, '10:11': 0.03212387245714738, '11:12': -0.3117284410482479, '12:13': -0.6902972544111748, '13:14': -0.41411237012903707, '14:15': 0.721577804874857, '15:16': -0.904801535059234}
# }

# Nevergrad Robosuite can pick and place Panda
base_override = {
    "gamma": {'0:1': 1.985504, '1:2': 1.010491, '2:3': 1.002835, '3:4': 0.998978, '4:5': 1.991995, '5:6': 1.997247, '6:7': 0.991372, '7:8': 1.012341, '8:9': 1.999457, '9:10': 1.010292, '10:11': 0.990623, '11:12': 0.985442, '12:13': 1.000729, '13:14': 0.994974, '14:15': 0.989501, '15:16': 0.997372, '16:17': 0.992853, '17:18': 1.006154, '18:19': 0.996955, '19:20': 1.003189, '20:21': 0.996801, '21:22': 0.998382, '22:23': 0.990931, '23:24': 1.004519, '24:25': 0.998187, '25:26': 0.996087, '26:27': 0.999151, '27:28': 1.00059, '28:29': 0.997028, '29:30': 0.993615, '30:31': 1.008639, '31:32': 1.999226, '32:33': 1.004321, '33:34': 0.990598, '34:35': 1.010147, '35:36': 0.987918, '36:37': 1.008199, '37:38': 1.002458, '38:39': 0.995116, '39:40': 0.999, '40:41': 1.014386, '41:42': 1.019379, '42:43': 1.011426, '43:44': 1.09281, '44:45': 0.964621, '45:46': 0.978411, '46:47': 1.001793, '47:48': 0.999529, '48:49': 0.980406, '49:50': 1.990166, '50:51': 0.999257, '51:52': 0.990541, '52:53': 1.004127, '53:54': 0.991752, '54:55': 1.015979, '55:56': 0.989897, '56:57': 1.000764, '57:58': 1.006905, '58:59': 0.982624, '59:60': 1.004257, '60:61': 1.006257, '61:62': 1.011399, '62:63': 1.019377, '63:64': 1.007688, '64:65': 1.010706, '65:66': 1.006487, '66:67': 0.991202, '67:68': 1.006066, '68:69': 0.999903, '69:70': 0.996335, '70:71': 1.009952, '71:72': 1.005746, '72:73': 0.986399, '73:74': 0.916244, '74:75': 0.980521, '75:76': 1.007629, '76:77': 0.994013, '77:78': 1.013146, '78:79': 0.999893, '79:80': 0.991892, '80:81': 0.981813, '81:82': 1.007145, '82:83': 1.011958, '83:84': 1.022765, '84:85': 1.015943, '85:86': 1.004294, '86:87': 1.007759, '87:88': 0.996807, '88:89': 1.011903, '89:90': 1.007506, '90:91': 1.002707, '91:92': 1.993489, '92:93': 0.999721, '93:94': 1.005511, '94:95': 1.004018, '95:96': 1.01495, '96:97': 1.004298, '97:98': 0.985817, '98:99': 0.992028, '99:100': 1.01046, '100:101': 1.011738, '101:102': 1.004547, '102:103': 0.995024, '103:104': 1.011519, '104:105': 0.997903, '105:106': 0.981109, '106:107': 1.008826, '107:108': 1.041855, '108:109': 1.002821, '109:110': 1.005785, '110:111': 1.031286, '111:112': 0.999429, '112:113': 1.001052, '113:114': 0.999737, '114:115': 1.014822, '115:116': 1.000204, '116:117': 0.991888, '117:118': 1.012069, '118:119': 1.009569, '119:120': 0.993508, '120:121': 0.984844, '121:122': 1.00372, '122:123': 0.986344, '123:124': 1.008584, '124:125': 1.000459, '125:126': 1.000567, '126:127': 0.984994, '127:128': 1.007024},
    "beta": {'0:1': -0.016574, '1:2': 0.000455, '2:3': -0.012527, '3:4': -0.489317, '4:5': 0.052145, '5:6': 0.000194, '6:7': 0.011766, '7:8': -0.000268, '8:9': 6.2e-05, '9:10': 0.013694, '10:11': 0.003715, '11:12': 0.001057, '12:13': -0.004717, '13:14': -0.001572, '14:15': -0.006484, '15:16': 0.016936, '16:17': -0.000318, '17:18': 0.013792, '18:19': 0.012619, '19:20': -9.6e-05, '20:21': -0.000109, '21:22': 0.002619, '22:23': 0.012228, '23:24': 0.005829, '24:25': -0.00889, '25:26': -0.00696, '26:27': 0.000634, '27:28': 0.012662, '28:29': 0.000677, '29:30': 0.004237, '30:31': -0.002447, '31:32': -0.003329, '32:33': 0.011469, '33:34': -0.002114, '34:35': -0.007128, '35:36': 0.009067, '36:37': 0.016068, '37:38': 0.020579, '38:39': 0.016946, '39:40': -0.003872, '40:41': -0.009247, '41:42': 0.011606, '42:43': -0.001973, '43:44': 0.015631, '44:45': -0.000331, '45:46': 0.003081, '46:47': 0.192907, '47:48': 0.012879, '48:49': 0.015675, '49:50': 0.010864, '50:51': -0.005012, '51:52': 0.001323, '52:53': -0.012422, '53:54': -0.007471, '54:55': 0.012888, '55:56': 0.001621, '56:57': -0.002128, '57:58': -0.010018, '58:59': 0.00494, '59:60': 0.008042, '60:61': -0.167272, '61:62': 0.002131, '62:63': 0.000243, '63:64': -0.001285, '64:65': 0.005504, '65:66': 0.01309, '66:67': -0.002666, '67:68': -0.011121, '68:69': 1.00186, '69:70': -0.000528, '70:71': 0.007969, '71:72': -0.001791, '72:73': 0.006533, '73:74': -0.005692, '74:75': -0.002512, '75:76': -0.002741, '76:77': 0.007197, '77:78': -0.00562, '78:79': -0.012625, '79:80': -0.001199, '80:81': 0.00152, '81:82': 0.010112, '82:83': -0.013133, '83:84': -0.011864, '84:85': -0.003613, '85:86': -0.003094, '86:87': -0.000666, '87:88': -0.004354, '88:89': -0.000364, '89:90': -0.00105, '90:91': -0.015895, '91:92': -0.004962, '92:93': 0.010823, '93:94': 0.005674, '94:95': -0.001335, '95:96': 0.003074, '96:97': 0.04209, '97:98': 0.010988, '98:99': 0.007822, '99:100': 0.001661, '100:101': 0.013758, '101:102': 0.007285, '102:103': 0.009659, '103:104': 0.003642, '104:105': -0.004084, '105:106': -0.000154, '106:107': 0.000254, '107:108': -0.016604, '108:109': -0.009162, '109:110': -0.014131, '110:111': 0.002624, '111:112': -0.009192, '112:113': -0.014583, '113:114': -0.012048, '114:115': -0.003001, '115:116': 0.008402, '116:117': -0.011043, '117:118': -0.013621, '118:119': -0.001829, '119:120': -0.02275, '120:121': -0.004238, '121:122': 0.009429, '122:123': 0.003416, '123:124': 0.003238, '124:125': -0.003938, '125:126': 0.00749, '126:127': 0.00635, '127:128': -0.001253}
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