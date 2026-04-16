"""Train VLA on a robomimic HDF5 dataset (image, state, action, text instruction)"""

import argparse
import os
import json

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from models.vla_diffusion_policy import VLADiffusionPolicy
from utils.tokenizer import SimpleTokenizer

import wandb

DEFAULT_STATE_KEYS = [
    "robot0_eef_pos",  
    "robot0_eef_quat",   
    "robot0_gripper_qpos", 
    "object",               
]


class RoboMimicHDF5Dataset(Dataset):

    def __init__(
        self,
        dataset_path: str,
        camera_obs_key: str = "agentview_image",
        state_obs_keys: list = None,
        instruction: str = "pick up the cube",
        filter_key: str = None,
        max_demos: int = None,
    ):
        if state_obs_keys is None:
            state_obs_keys = DEFAULT_STATE_KEYS

        images, states, actions, texts = self._load(
            dataset_path, camera_obs_key, state_obs_keys,
            instruction, filter_key, max_demos
        )

        self.images = np.concatenate(images, axis=0)           # (N, H, W, 3)
        self.states = np.concatenate(states, axis=0).astype(np.float32)  # (N, state_dim)
        self.actions = np.concatenate(actions, axis=0).astype(np.float32)  # (N, action_dim)

        tokenizer = SimpleTokenizer(vocab=None)
        tokenizer.build_from_texts(texts)
        text_ids_list = [tokenizer.encode(t) for t in texts]
        max_len = max(len(t) for t in text_ids_list)
        self.text_ids = np.zeros((len(text_ids_list), max_len), dtype=np.int64)
        for i, token_ids in enumerate(text_ids_list):
            self.text_ids[i, :len(token_ids)] = np.asarray(token_ids, dtype=np.int64)
        self.vocab = tokenizer.vocab

        print(f"Dataset loaded: {len(self.images)} transitions")
        print(f"  image shape : {self.images.shape}")
        print(f"  state shape : {self.states.shape}")
        print(f"  action shape: {self.actions.shape}")
        print(f"  vocab size  : {max(self.vocab.values()) + 1}")

    @staticmethod
    def _load(dataset_path, camera_obs_key, state_obs_keys,
              instruction, filter_key, max_demos):
        images, states, actions, texts = [], [], [], []

        with h5py.File(dataset_path, "r") as f:
            if filter_key is not None:
                print(f"Using filter key: {filter_key}")
                demos = sorted([
                    elem.decode("utf-8")
                    for elem in np.array(f[f"mask/{filter_key}"])
                ])
            else:
                demos = sorted(list(f["data"].keys()))

            demos = sorted(demos, key=lambda x: int(x[5:]))

            if max_demos is not None:
                demos = demos[:max_demos]
                print(f"Using {len(demos)} demos (max_demos={max_demos})")

            for demo_key in demos:
                demo = f["data"][demo_key]
                obs = demo["obs"]

                # image
                if camera_obs_key not in obs:
                    raise KeyError(
                        f"camera_obs_key '{camera_obs_key}' not found in demo '{demo_key}'. "
                        f"Available: {list(obs.keys())}"
                    )
                image_arr = np.asarray(obs[camera_obs_key])  # (T, H, W, 3)

                # state
                missing = [k for k in state_obs_keys if k not in obs]
                if missing:
                    raise KeyError(f"State keys {missing} not found in demo '{demo_key}'.")
                state_arr = np.concatenate(
                    [np.asarray(obs[k], dtype=np.float32) for k in state_obs_keys],
                    axis=-1
                )  # (T, state_dim)

                # action
                action_arr = np.asarray(demo["actions"], dtype=np.float32)  # (T, action_dim)

                T = min(image_arr.shape[0], state_arr.shape[0], action_arr.shape[0])
                if T <= 0:
                    continue

                images.append(image_arr[:T])
                states.append(state_arr[:T])
                actions.append(action_arr[:T])

                lang = demo.attrs.get("language_instruction", instruction)
                if isinstance(lang, bytes):
                    lang = lang.decode("utf-8")
                texts.extend([str(lang)] * T)

        return images, states, actions, texts

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        img = self.images[idx]  # (H, W, 3), uint8

        # uint8に変換
        if img.dtype != np.uint8:
            img = (img.clip(0, 1) * 255).astype(np.uint8) if img.max() <= 1.0 \
                else img.clip(0, 255).astype(np.uint8)

        # (H, W, 3) -> (3, H, W), [0, 1]
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        state = torch.from_numpy(self.states[idx]).float()
        action = torch.from_numpy(self.actions[idx]).float()
        text_ids = torch.from_numpy(self.text_ids[idx]).long()
        return img, state, action, text_ids


def parse_args():
    parser = argparse.ArgumentParser(description="Train VLA on robomimic HDF5 dataset")
    parser.add_argument("--dataset-path", type=str,
                        default="/home/kaito/robomimic/datasets/lift/ph/image.hdf5")
    parser.add_argument("--camera-obs-key", type=str, default="agentview_image")
    parser.add_argument("--state-obs-keys", type=str, default="")
    parser.add_argument("--instruction", type=str, default="pick up the cube")
    parser.add_argument("--filter-key", type=str, default="train")
    parser.add_argument("--max-demos", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--diffusion-T", type=int, default=16)
    parser.add_argument("--save-path", type=str, default="checkpoints/model.pt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--wandb-project", type=str, default="mini-vla")
    return parser.parse_args()


def main():
    args = parse_args()

    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    state_obs_keys = [k.strip() for k in args.state_obs_keys.split(",") if k.strip()] or None
    max_demos = args.max_demos if args.max_demos > 0 else None
    filter_key = args.filter_key if args.filter_key.lower() != "none" else None

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = RoboMimicHDF5Dataset(
        dataset_path=args.dataset_path,
        camera_obs_key=args.camera_obs_key,
        state_obs_keys=state_obs_keys,
        instruction=args.instruction,
        filter_key=filter_key,
        max_demos=max_demos,
    )

    vocab_size = max(dataset.vocab.values()) + 1
    state_dim = dataset.states.shape[1]
    action_dim = dataset.actions.shape[1]

    model = VLADiffusionPolicy(
        vocab_size=vocab_size,
        state_dim=state_dim,
        action_dim=action_dim,
        d_model=args.d_model,
        diffusion_T=args.diffusion_T,
    ).to(device)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    wandb.init(
        project=args.wandb_project,
        config={
            "dataset_path": args.dataset_path,
            "filter_key": filter_key,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "d_model": args.d_model,
            "diffusion_T": args.diffusion_T,
            "optimizer": "Adam",
            "vocab_size": vocab_size,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "dataset_size": len(dataset),
            "num_batches_per_epoch": len(loader),
        }
    )

    avg_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        epoch_losses = []
        epoch_loss_dict = {f"loss_t{t}": [] for t in range(args.diffusion_T)}

        for batch_idx, (img, state, action, text_ids) in enumerate(loader):
            img = img.to(device)
            state = state.to(device)
            action = action.to(device)
            text_ids = text_ids.to(device)

            loss, loss_dict = model.loss(img, text_ids, state, action)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * img.size(0)
            epoch_losses.append(loss.item())

            for t_key, t_loss in loss_dict.items():
                epoch_loss_dict[t_key].append(t_loss)

            log_dict = {
                "train/batch_loss": loss.item(),
                "train/epoch": epoch + 1,
                "train/global_step": epoch * len(loader) + batch_idx,
            }
            for t_key, t_loss in loss_dict.items():
                log_dict[f"train/{t_key}"] = t_loss
            wandb.log(log_dict)

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}/{args.epochs}  loss={avg_loss:.6f}")

        epoch_log_dict = {
            "train/epoch_loss": avg_loss,
            "train/epoch_loss_std": float(np.std(epoch_losses)),
        }
        for t_key in epoch_loss_dict:
            if epoch_loss_dict[t_key]:
                epoch_log_dict[f"epoch_{t_key}"] = np.mean(epoch_loss_dict[t_key])
        wandb.log(epoch_log_dict)

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "final_loss": avg_loss,
        "vocab": dataset.vocab,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "d_model": args.d_model,
        "diffusion_T": args.diffusion_T,
    }, args.save_path)
    print(f"Saved final checkpoint: {args.save_path}")

    wandb.log({"train/final_loss": avg_loss, "train/total_epochs": args.epochs})
    wandb.finish()


if __name__ == "__main__":
    main()