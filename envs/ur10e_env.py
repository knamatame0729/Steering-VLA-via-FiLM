"""
UR10e robot environment wrappers for Metaworld tasks.
"""
import os
import sys

import mujoco
import numpy as np

# Add Metaworld to path if not already present
_metaworld_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Metaworld")
if _metaworld_path not in sys.path:
    sys.path.insert(0, _metaworld_path)

# Import the UR10e-specific environments
from metaworld.envs.ur10e_pick_place_v3 import UR10ePickPlaceEnvV3 as _UR10ePickPlaceEnvV3


class UR10eEnvWrapper:
    """
    Wraps a UR10e environment
    """
    def __init__(self, env, seed=42, random_init=False):
        self.env = env
        self.render_mode = env.render_mode
        self.random_init = random_init
        self._saved_init_state = None
        
        # Create a seeded random generator for reproducible random init positions
        self.rng = np.random.default_rng(seed)
        
        # Mark task as set to bypass the assert_task_is_set decorator
        self.env._set_task_called = True

        if hasattr(self.env, "_freeze_rand_vec"):
            self.env._freeze_rand_vec = False
        if hasattr(self.env, "_last_rand_vec"):
            self.env._last_rand_vec = None
        
        # Get dimensions from a test reset
        obs, _ = self.env.reset(seed=seed)

        # Keep a deterministic initial state for non-random evaluation episodes.
        if not self.random_init:
            self._save_current_state(getattr(self.env, "unwrapped", self.env))

        self.state_dim = self._extract_state(obs).shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.obs_shape = self._get_image().shape

    def _save_current_state(self, unwrapped):
        goal_site_id = mujoco.mj_name2id(unwrapped.model, mujoco.mjtObj.mjOBJ_SITE, "goal")
        self._saved_init_state = {
            "qpos": unwrapped.data.qpos.copy(),
            "qvel": unwrapped.data.qvel.copy(),
            "goal": unwrapped.goal.copy(),
            "obj_init_pos": unwrapped.obj_init_pos.copy(),
            "_target_pos": unwrapped._target_pos.copy() if hasattr(unwrapped, "_target_pos") else None,
            "goal_site_pos": unwrapped.model.site_pos[goal_site_id].copy(),
        }

    def _restore_saved_state(self, unwrapped):
        if self._saved_init_state is None:
            return

        unwrapped.obj_init_pos = self._saved_init_state["obj_init_pos"].copy()
        unwrapped.goal = self._saved_init_state["goal"].copy()
        unwrapped.data.qpos[:] = self._saved_init_state["qpos"]
        unwrapped.data.qvel[:] = self._saved_init_state["qvel"]

        if self._saved_init_state.get("_target_pos") is not None:
            unwrapped._target_pos = self._saved_init_state["_target_pos"].copy()

        if self._saved_init_state.get("goal_site_pos") is not None:
            goal_site_id = mujoco.mj_name2id(unwrapped.model, mujoco.mjtObj.mjOBJ_SITE, "goal")
            unwrapped.model.site_pos[goal_site_id] = self._saved_init_state["goal_site_pos"].copy()

        mujoco.mj_forward(unwrapped.model, unwrapped.data)

    def _extract_state(self, obs):
        """Extract state from observation."""
        if isinstance(obs, dict):
            if "observation" in obs:
                state = obs["observation"]
            elif "robot_state" in obs or "object_state" in obs:
                state_parts = []
                if "robot_state" in obs:
                    state_parts.append(obs["robot_state"])
                if "object_state" in obs:
                    state_parts.append(obs["object_state"])
                state = np.concatenate(state_parts, axis=-1)
            else:
                raise KeyError(
                    f"No suitable state keys in observation dict. "
                    f"Available keys: {list(obs.keys())}"
                )
        else:
            state = obs
        return np.asarray(state, dtype=np.float32)

    def _get_image(self):
        """Render and return RGB array."""
        img = self.env.render()
        if img is None:
            raise RuntimeError("render() returned None from rgb_array environment")
        return img.astype(np.uint8)

    def reset(self, seed=None):
        """Reset environment."""
        unwrapped = getattr(self.env, 'unwrapped', self.env)
        
        if seed is not None:
            unwrapped.np_random = np.random.RandomState(seed)
        
        if self.random_init:
            unwrapped._freeze_rand_vec = False
            unwrapped._last_rand_vec = None 
        
        obs, info = self.env.reset(seed=seed)

        # For deterministic eval, force the same object/goal every episode.
        if not self.random_init:
            self._restore_saved_state(unwrapped)
            obs = unwrapped._get_obs()
        
        if self.random_init:
            # Sample random object position within valid UR10e pick-place ranges
            obj_pos = np.array([
                self.rng.uniform(-0.1, 0.1),   # x
                self.rng.uniform(0.6, 0.7),    # y
                0.02                           # z 
            ])
            # Sample random goal position
            goal_pos = np.array([
                self.rng.uniform(-0.1, 0.1),   # x
                self.rng.uniform(0.8, 0.9),    # y
                self.rng.uniform(0.05, 0.3)    # z 
            ])

            # Ensure minimum XY distance of 0.15 between object and goal
            while np.linalg.norm(obj_pos[:2] - goal_pos[:2]) < 0.15:
                goal_pos[0] = self.rng.uniform(-0.1, 0.1)
                goal_pos[1] = self.rng.uniform(0.8, 0.9)

            # Apply randomized positions
            unwrapped.obj_init_pos = obj_pos.copy()
            unwrapped._target_pos = goal_pos.copy()
            unwrapped.goal = goal_pos.copy()
            
            # Set object position directly using correct qpos indices
            qpos = unwrapped.data.qpos.flatten().copy()
            qvel = unwrapped.data.qvel.flatten().copy()
            qpos[14:17] = obj_pos.copy()
            qvel[14:20] = 0
            unwrapped.set_state(qpos, qvel)
            
            # Set goal site position
            unwrapped.model.site("goal").pos = goal_pos.copy()

            # Forward simulation to apply changes
            mujoco.mj_forward(unwrapped.model, unwrapped.data)

            # Re-read observation with updated positions
            obs = unwrapped._get_obs()
            
            print(f"[UR10e Reset] Object pos: {obj_pos}, Goal pos: {goal_pos}")
        
        state = self._extract_state(obs)
        image = self._get_image()
        return image, state, info

    def step(self, action):
        """Take step in environment."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        state = self._extract_state(obs)
        image = self._get_image()
        return image, state, reward, done, info

    def close(self):
        """Close the environment."""
        self.env.close()

    def __getattr__(self, name):
        """Delegate attribute access to the underlying environment."""
        return getattr(self.env, name)


class UR10ePickPlaceEnvV3(UR10eEnvWrapper):
    """Pick and Place environment using UR10e robot with wrapper interface."""
    
    def __init__(self, render_mode="rgb_array", camera_name="topview", seed=42, random_init=False, **kwargs):
        env = _UR10ePickPlaceEnvV3(
            render_mode=render_mode,
            camera_name=camera_name,
            **kwargs
        )
        super().__init__(env, seed=seed, random_init=random_init)


__all__ = ["UR10ePickPlaceEnvV3", "UR10eEnvWrapper"]