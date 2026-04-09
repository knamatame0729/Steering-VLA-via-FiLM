"""Train VLA on dataset of image, state, action, and text instruction"""

import argparse
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from models.vla_diffusion_policy import VLADiffusionPolicy

import wandb

class TrainingDataset(Dataset):
    """Dataset for VLA
    
    Loads multi-modal data:
    - images: robot camera observations
    - states: joint positions, gripper state, etc.
    - actions: robot actions (delta end-effector pose, gripper command)
    - text_ids: tokenized language instructions
    """
    def __init__(self, path, resize_to=64):
        data = np.load(path, allow_pickle=True)
        self.images = data["images"]             # (N, H, W, 3)
        self.states = data["states"]             # (N, state_dim)
        self.actions = data["actions"]           # (N, action_dim)
        self.text_ids = data["text_ids"]         # (N, T_text)
        self.vocab = data["vocab"].item() if data["vocab"].shape == () else data["vocab"]
        self.resize_to = resize_to

        try:
            import cv2
            self.cv2 = cv2
        except ImportError:
            self.cv2 = None

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        # Load and preprocess image: resize and normalize to [0, 1]
        img = self.images[idx]  # (H, W, 3), uint8
        if self.cv2 is not None and (img.shape[0] != self.resize_to or img.shape[1] != self.resize_to):
            img = self.cv2.resize(img, (self.resize_to, self.resize_to))

        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # (3, H, W)
        state = torch.from_numpy(self.states[idx]).float()
        action = torch.from_numpy(self.actions[idx]).float()
        text_ids = torch.from_numpy(self.text_ids[idx]).long()
        return img, state, action, text_ids


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str,
                        default="data/dataset.npz")
    parser.add_argument("--resize-to", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--diffusion-T", type=int, default=16)
    parser.add_argument("--save-path", type=str,
                        default="checkpoints/model.pt")
    parser.add_argument("--device", type=str, default="cuda",
                        help="'cuda' or 'cpu'")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    # Setup device and load dataset
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = TrainingDataset(args.dataset_path, resize_to=args.resize_to)
    
    # Infer model dimensions from dataset
    vocab_size = max(dataset.vocab.values()) + 1
    state_dim = dataset.states.shape[1]
    action_dim = dataset.actions.shape[1]

    # Initialize VLA model with diffusion policy head
    # Architecture: Image Encoder + Text Encoder + State Encoder + Fusion MLP + Diffusion Head
    model = VLADiffusionPolicy(
        vocab_size=vocab_size,
        state_dim=state_dim,
        action_dim=action_dim,
        d_model=args.d_model,
        diffusion_T=args.diffusion_T
    ).to(device)

    # Setup training pipeline
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    num_epochs = args.epochs

    # Initialize Weights & Biases for experiment tracking
    wandb.init(
        project="mini-vla",
        config={
            "dataset_path": args.dataset_path,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "d_model": args.d_model,
            "diffusion_T": args.diffusion_T,
            "resize_to": args.resize_to,
            "optimizer": "Adam",
            "vocab_size": vocab_size,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "dataset_size": len(dataset),
            "num_batches_per_epoch": len(loader),
        }
    )

    # Initialize training tracking variables
    best_loss = float('inf')
    loss_history = []

    # Main training loop
    for epoch in range(num_epochs):

        model.train()
        total_loss = 0.0
        epoch_losses = []

        # Track loss at each diffusion timestep (t=0 to T-1)
        epoch_loss_dict = {f"loss_t{t}": [] for t in range(args.diffusion_T)}

        for batch_idx, (img, state, action, text_ids) in enumerate(loader):
            # Move batch to device
            img = img.to(device)
            state = state.to(device)
            action = action.to(device)
            text_ids = text_ids.to(device)

            # Forward pass: compute diffusion loss (denoising objective)
            # loss_dict contains per-timestep losses for each diffusion step
            loss, loss_dict = model.loss(img, text_ids, state, action)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * img.size(0)
            epoch_losses.append(loss.item())

            # Collect per-timestep losses
            for t_key, t_loss in loss_dict.items():
                epoch_loss_dict[t_key].append(t_loss)

            # Log batch metrics
            log_dict = {
                "train/batch_loss": loss.item(),
                "train/epoch": epoch + 1,
                "train/batch": batch_idx,
                "train/global_step": epoch * len(loader) + batch_idx,
            }

            # Log per-timestep losses for current batch
            for t_key, t_loss in loss_dict.items():
                log_dict[f"train/{t_key}"] = t_loss

            wandb.log(log_dict)

        # Compute epoch statistics
        avg_loss = total_loss / len(dataset)
        loss_history.append(avg_loss)
        
        # Compute statistics for epoch loss distribution
        epoch_losses_array = np.array(epoch_losses)
        
        # Average per-timestep losses for the epoch (useful for debugging diffusion training)
        avg_per_t_losses = {}
        for t_key in epoch_loss_dict:
            if epoch_loss_dict[t_key]:
                avg_per_t_losses[f"epoch_{t_key}"] = np.mean(epoch_loss_dict[t_key])

        print(f"Epoch {epoch+1}/{num_epochs}  loss={avg_loss:.4f}")

        # Log epoch metrics
        epoch_log_dict = {
            "train/epoch_loss": avg_loss,
            "train/epoch_loss_std": float(np.std(epoch_losses)),
        }
        
        # Add per-timestep averages
        epoch_log_dict.update(avg_per_t_losses)

        wandb.log(epoch_log_dict)

        # Save checkpoint after each epoch for easy resuming and model selection
        checkpoint_path = args.save_path.replace('.pt', f'_epoch{epoch+1}.pt')
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch + 1,
                "loss": avg_loss,
                # Save model config for inference
                "vocab": dataset.vocab,
                "state_dim": state_dim,
                "action_dim": action_dim,
                "d_model": args.d_model,
                "diffusion_T": args.diffusion_T,
            },
            checkpoint_path,
        )

    # Save final model
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "final_loss": avg_loss,
            "vocab": dataset.vocab,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "d_model": args.d_model,
            "diffusion_T": args.diffusion_T,
        },
        args.save_path,
    )
    print("Saved checkpoint:", args.save_path)

    # Log final metrics
    wandb.log({
        "train/final_loss": avg_loss,
        "train/total_epochs": num_epochs,
    })

    # Close W&B run
    wandb.finish()


if __name__ == "__main__":
    main()
