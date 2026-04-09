"""
FiLMExperimentLogger
"""

import numpy as np
import matplotlib.pyplot as plt
import wandb


def _make_bar_colors(gamma: np.ndarray) -> list:
    colors = []
    for g in gamma:
        g = float(g)
        if g > 1.05:
            intensity = min((g - 1.0) / 3.0, 1.0)
            gb = round(1.0 - intensity * 0.8, 4)
            colors.append((1.0, gb, gb))
        elif g < 0.95:
            intensity = min(1.0 - g, 1.0)
            rb = round(1.0 - intensity * 0.8, 4)
            colors.append((rb, rb, 1.0))
        else:
            colors.append((0.88, 0.88, 0.88))
    return colors


class FiLMExperimentLogger:
    def __init__(self, project_name: str = "Manual_FiLM_VLA_Evaluation"):
        self.project_name      = project_name
        self.all_episodes_data = []
        self.baseline_actions  = None

    # ------------------------------------------------------------------
    # 1. Baseline vs Modulated
    # ------------------------------------------------------------------
    def visualize_baseline_vs_modulated(self, actions, gamma, episode_num):
        action_dim   = actions.shape[1]
        baseline     = self.baseline_actions
        changed_dims = np.where(np.abs(gamma - 1.0) > 1e-5)[0].tolist()
        min_steps    = min(len(actions), len(baseline))

        fig, axes = plt.subplots(action_dim, 1, figsize=(14, 2.8 * action_dim), sharex=True)
        if action_dim == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            ax.plot(baseline[:min_steps, i], color="gray", linewidth=1.5,
                    linestyle="--", label="baseline", alpha=0.8)
            ax.plot(actions[:min_steps, i],  color=f"C{i}", linewidth=2.0,
                    label="modulated")
            ax.axhline(0, color="black", linewidth=0.8, linestyle=":", alpha=0.4)
            if i == 0:
                ax.set_ylabel(f"dx", fontsize=10)
            elif i == 1:
                ax.set_ylabel(f"dy", fontsize=10)
            elif i == 2:
                ax.set_ylabel(f"dz", fontsize=10)
            elif i == 3:
                ax.set_ylabel(f"gripper", fontsize=10)
            ax.legend(fontsize=8, loc="upper right")
            ax.grid(True, alpha=0.25)

        axes[-1].set_xlabel("Step", fontsize=10)
        fig.suptitle(
            f"Episode {episode_num}"
            f"({len(changed_dims)} / {len(gamma)} neurons)",
            fontsize=12, fontweight="bold"
        )
        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # 2. 2D trajectory
    # ------------------------------------------------------------------
    def visualize_trajectory_2d(self, positions, episode_num):
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.plot(positions[:, 0], positions[:, 1],
                marker="o", markersize=2, alpha=0.6, linewidth=2, color="steelblue")
        ax.scatter(*positions[0, :2],  color="green", s=200, marker="o", label="Start", zorder=5)
        ax.scatter(*positions[-1, :2], color="red",   s=200, marker="X", label="End",   zorder=5)
        ax.set_title(f"Episode {episode_num}  |  2D Trajectory", fontsize=13, fontweight="bold")
        ax.set_xlabel("X"); ax.set_ylabel("Y")
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # 3. 3D trajectory
    # ------------------------------------------------------------------
    def visualize_trajectory_3d(self, positions, episode_num):
        fig = plt.figure(figsize=(9, 7))
        ax  = fig.add_subplot(111, projection="3d")
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
                marker="o", markersize=2, alpha=0.6, linewidth=2)
        ax.scatter(*positions[0],  color="green", s=200, marker="o", label="Start", zorder=5)
        ax.scatter(*positions[-1], color="red",   s=200, marker="X", label="End",   zorder=5)
        ax.set_title(f"Episode {episode_num}  |  3D Trajectory", fontsize=13, fontweight="bold")
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.legend()
        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------
    def log_episode(self, episode_data: dict, episode_num: int):
        positions = episode_data["positions"]
        actions   = episode_data["actions"]
        gamma     = np.array(episode_data["gamma"]).flatten()

        print(f"\n[Logger] Episode {episode_num} logging...")
        log_dict = {}

        if episode_num == 0:
            self.baseline_actions = actions.copy()
            print("  [Logger] ep0 saved as baseline.")

        if len(actions) > 0 and episode_num > 0 and self.baseline_actions is not None:
            # 1. Baseline vs Modulated
            fig = self.visualize_baseline_vs_modulated(actions, gamma, episode_num)
            log_dict[f"film/baseline_vs_modulated"] = wandb.Image(fig)
            plt.close(fig)

        if len(positions) > 0:
            fig = self.visualize_trajectory_2d(positions, episode_num)
            log_dict[f"traj/2d"] = wandb.Image(fig)
            plt.close(fig)

            if positions.shape[1] >= 3:
                fig = self.visualize_trajectory_3d(positions, episode_num)
                log_dict[f"traj/3d"] = wandb.Image(fig)
                plt.close(fig)

        wandb.log(log_dict, step=episode_num)
        self.all_episodes_data.append(episode_data)
        print(f"[Logger] Episode {episode_num} done.")

    def log_final_summary(self):
        print("\n[Logger] Generating cross-episode delta comparison...")
        fig = self.visualize_cross_episode_delta()
        if fig:
            wandb.log({"film/cross_episode_delta": wandb.Image(fig)})
            plt.close(fig)
        print("[Logger] Done.")