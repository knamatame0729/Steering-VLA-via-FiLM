"""
Random object position
"""

import gymnasium as gym
import numpy as np
import mujoco
import metaworld

class MetaWorldMT1Wrapper:
    """
    Wraps a Metaworld MT1 environment into a simple interface:
    - reset() -> (image, state)
    - step(action) -> (image, state, reward, done, info)
    """
    def __init__(self, env_name='push-v3', seed=42, render_mode='rgb_array', camera_name='topview', random_init=False):
        self.env = gym.make(
            'Meta-World/MT1',
            env_name=env_name,
            seed=seed,
            render_mode=render_mode,
            camera_name=camera_name
        )
        self.render_mode = render_mode
        self.random_init = random_init
        self.init_seed = seed 
        
        # Create a seeded random generator for reproducible random init positions
        self.ep_rng = None
        self.ep_count = 0

        self._saved_init_state = None
        obs, _ = self.env.reset(seed=seed)
        self._save_init_state()

        self.state_dim = self._extract_state(obs).shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.obs_shape = self._get_image().shape

    def _extract_state(self, obs):
        """
        Adapt this to your env's observation structure.
        Examples:
          - obs might be a dict with keys ["robot_state", "object_state"].
          - or it might already be a flat vector.
        """
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
        img = self.env.render()
        img = img.astype(np.uint8)
        return img

    def reset_episode_count(self):
        self.ep_count = 0

    def _save_init_state(self):
        unwrapped = self.env.unwrapped
        goal_site_id = mujoco.mj_name2id(unwrapped.model, mujoco.mjtObj.mjOBJ_SITE, 'goal')
        self._saved_init_state = {
            'qpos': unwrapped.data.qpos.copy(),
            'qvel': unwrapped.data.qvel.copy(),
            'goal': unwrapped.goal.copy(),
            'obj_init_pos': unwrapped.obj_init_pos.copy(),
            '_target_pos': unwrapped._target_pos.copy() if hasattr(unwrapped, '_target_pos') else None,
            'goal_site_pos': unwrapped.model.site_pos[goal_site_id].copy(),
        }

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)

        if self.random_init:
            ep_seed = self.init_seed + self.ep_count
            self.ep_rng = np.random.default_rng(ep_seed)
            self.ep_count += 1

            unwrapped = self.env.unwrapped
            
            obj_pos = np.array([
                self.ep_rng.uniform(-0.1, 0.1),   # x
                self.ep_rng.uniform(0.6, 0.7),    # y
                0.02                           # z
            ])

            #print(f"Object position {obj_pos}")

            # # Sample random goal position
            # goal_pos = np.array([
            #     self.rng.uniform(-0.1, 0.1),   # x
            #     self.rng.uniform(0.8, 0.9),     # y
            #     self.rng.uniform(0.05, 0.3)     # z (aerial)
            # ])
            # # Ensure minimum XY distance of 0.15 between object and goal
            # while np.linalg.norm(obj_pos[:2] - goal_pos[:2]) < 0.15:
            #     goal_pos[0] = self.rng.uniform(-0.1, 0.1)
            #     goal_pos[1] = self.rng.uniform(0.8, 0.9)

            goal_pos = np.array([0.0, 0.85, 0.2])

            # Apply randomized positions
            unwrapped.obj_init_pos = obj_pos.copy()
            unwrapped._target_pos = goal_pos.copy()
            unwrapped.goal = goal_pos.copy()
            unwrapped._set_obj_xyz(obj_pos)
            unwrapped.model.site("goal").pos = goal_pos.copy()

            # Forward simulation to apply changes
            mujoco.mj_forward(unwrapped.model, unwrapped.data)

            # Re-read observation with updated positions
            obs = unwrapped._get_obs()

        else:
            unwrapped = self.env.unwrapped

            obj_pos = np.array([
                0.24,   # x
                0.55,    # y
                0.02                           # z
            ])
            goal_pos = np.array([0.0, 0.85, 0.2])

            unwrapped.obj_init_pos = obj_pos.copy()
            unwrapped.goal = goal_pos.copy()
            unwrapped._target_pos = goal_pos.copy()

            unwrapped.data.qpos[:] = self._saved_init_state['qpos']
            unwrapped.data.qvel[:] = self._saved_init_state['qvel']

            unwrapped._set_obj_xyz(obj_pos)

            goal_site_id = mujoco.mj_name2id(
                unwrapped.model, 
                mujoco.mjtObj.mjOBJ_SITE, 
                'goal'
            )
            unwrapped.model.site_pos[goal_site_id] = goal_pos.copy()

            mujoco.mj_forward(unwrapped.model, unwrapped.data)
            obs = unwrapped._get_obs()

        state = self._extract_state(obs)
        image = self._get_image()
        return image, state, info

    def step(self, action):
        obs, reward, truncate, terminate, info = self.env.step(action)
        done = truncate or terminate
        state = self._extract_state(obs)
        image = self._get_image()
        return image, state, reward, done, info

    def close(self):
        self.env.close()



#     ###########

# For another environment set-up
import gymnasium as gym
import numpy as np
import metaworld

class MetaWorldMT1Wrapper:
    """
    Wraps a Metaworld MT1 environment into a simple interface:
    - reset() -> (image, state)
    - step(action) -> (image, state, reward, done, info)
    """
    def __init__(self, env_name='push-v3', seed=42, render_mode='rgb_array', camera_name='topview'):
        self.env = gym.make(
            'Meta-World/MT1',
            env_name=env_name,
            seed=seed,
            render_mode=render_mode,
            camera_name=camera_name
        )
        self.render_mode = render_mode

        obs, _ = self.env.reset()
        self.state_dim = self._extract_state(obs).shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.obs_shape = self._get_image().shape

    def _extract_state(self, obs):
        """
        Adapt this to your env's observation structure.
        Examples:
          - obs might be a dict with keys ["robot_state", "object_state"].
          - or it might already be a flat vector.
        """
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
        img = self.env.render()
        img = img.astype(np.uint8)
        return img

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        state = self._extract_state(obs)
        image = self._get_image()
        return image, state, info

    def step(self, action):
        obs, reward, truncate, terminate, info = self.env.step(action)
        done = truncate or terminate
        state = self._extract_state(obs)
        image = self._get_image()
        return image, state, reward, done, info

    def close(self):
        self.env.close()