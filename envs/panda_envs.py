"""
Panda robot environment wrappers for Metaworld tasks.
Re-exports the Panda-specific environments from Metaworld.
"""
import os
import sys

import numpy as np

# Add Metaworld to path if not already present
_metaworld_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Metaworld")
if _metaworld_path not in sys.path:
    sys.path.insert(0, _metaworld_path)

# Import the Panda-specific environments (not Sawyer-based)
from metaworld.envs.panda_push_v3 import PandaPushEnvV3 as _PandaPushEnvV3
from metaworld.envs.panda_pick_place_v3 import PandaPickPlaceEnvV3 as _PandaPickPlaceEnvV3


class PandaEnvWrapper:
    """
    Wraps a Panda environment into a simple interface matching MetaWorldMT1Wrapper:
    - reset() -> (image, state, info)
    - step(action) -> (image, state, reward, done, info)
    """
    def __init__(self, env):
        self.env = env
        self.render_mode = env.render_mode
        
        # Mark task as set to bypass the assert_task_is_set decorator
        self.env._set_task_called = True
        
        # Get dimensions from a test reset
        obs, _ = self.env.reset()
        self.state_dim = self._extract_state(obs).shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.obs_shape = self._get_image().shape

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
        obs, info = self.env.reset(seed=seed)
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


class PandaPushEnvV3(PandaEnvWrapper):
    """Push environment using Franka Panda robot with wrapper interface."""
    
    def __init__(self, render_mode="rgb_array", camera_name="corner2", **kwargs):
        env = _PandaPushEnvV3(
            render_mode=render_mode,
            camera_name=camera_name,
            **kwargs
        )
        super().__init__(env)


class PandaPickPlaceEnvV3(PandaEnvWrapper):
    """Pick and Place environment using Franka Panda robot with wrapper interface."""
    
    def __init__(self, render_mode="rgb_array", camera_name="corner2", **kwargs):
        env = _PandaPickPlaceEnvV3(
            render_mode=render_mode,
            camera_name=camera_name,
            **kwargs
        )
        super().__init__(env)


__all__ = ["PandaPushEnvV3", "PandaPickPlaceEnvV3", "PandaEnvWrapper"]
