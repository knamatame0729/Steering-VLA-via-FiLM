"""RoboSuite environment wrapper with the same interface as the Meta-World wrapper."""

from __future__ import annotations

import random
from typing import Dict, Iterable, Optional, Sequence, Union

import numpy as np

try:
    import robosuite as suite
except ImportError:
    suite = None


def _load_controller_config(controller: str, robots: Union[Sequence[str], str]) -> dict:
    from robosuite.controllers import load_composite_controller_config, load_part_controller_config
    from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config

    robot = robots[0] if isinstance(robots, (list, tuple)) else robots
    part_config = load_part_controller_config(default_controller=controller)
    return refactor_composite_controller_config(
        controller_config=part_config,
        robot_type=robot,
        arms=["right"],
    )

DEFAULT_STATE_KEYS = [
    "robot0_eef_pos",      # 3
    "robot0_eef_quat",     # 4
    "robot0_gripper_qpos", # 2
    "object",              # 10
] 

STATE_KEY_ALIASES = {
    "object": ["object-state"],
    "object-state": ["object"],
}

class RoboSuiteWrapper:
    """
    Wrap a robosuite env into a simple interface:
    - reset() -> (image, state, info)
    - step(action) -> (image, state, reward, done, info)
    """

    def __init__(
        self,
        env_name: str = "PickPlaceCan",
        robots: Union[Sequence[str], str] = "Panda",
        seed: int = 42,
        deterministic_reset_order: bool = True,
        controller: str = "OSC_POSE",
        camera_name: str = "agentview",
        image_height: int = 84,
        image_width: int = 84,
        control_freq: int = 20,
        horizon: int = 400,
        reward_shaping: bool = False,
        ignore_done: bool = False,
        state_keys: list = None,
        render: bool = False,
    ):
        if suite is None:
            raise ImportError(
                "robosuite is required for RoboSuiteWrapper. "
                "Install robosuite and robomimic first."
            )

        controller_config = _load_controller_config(controller=controller, robots=robots)
        self.env = suite.make(
            env_name=env_name,
            robots=robots,
            controller_configs=controller_config,
            has_renderer=render,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            use_object_obs=True,
            camera_names=camera_name,
            camera_heights=image_height,
            camera_widths=image_width,
            control_freq=control_freq,
            horizon=horizon,
            reward_shaping=reward_shaping,
            ignore_done=ignore_done,
        )

        self.seed = seed
        self.init_seed = seed
        self._deterministic_reset_order = deterministic_reset_order
        self._episode_count = 0
        self._render = render
        self.camera_name = camera_name
        self.image_key = f"{camera_name}_image"
        self._rng = np.random.default_rng(seed)

        self._apply_seed(self.init_seed)
        obs = self.env.reset()
        self.state_keys = state_keys or DEFAULT_STATE_KEYS
        self.state_dim = self._extract_state(obs).shape[0]
        self.action_dim = self._infer_action_dim()
        self.obs_shape = self._get_image(obs).shape

        self._episode_count = 0

    def reset_episode_count(self):
        self._episode_count = 0

    def _apply_seed(self, seed: int):
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)

        np.random.seed(self.seed)
        random.seed(self.seed)

        self.env.seed = self.seed
        self.env.rng = np.random.default_rng(self.seed)

        placement_initializer = getattr(self.env, "placement_initializer", None)
        if placement_initializer is not None and hasattr(placement_initializer, "rng"):
            placement_initializer.rng = self.env.rng

    def _infer_action_dim(self) -> int:
        action_spec = getattr(self.env, "action_spec", None)
        if action_spec is not None:
            low, _ = action_spec
            return int(np.asarray(low).shape[0])
        return int(getattr(self.env, "action_dim"))

    def _extract_state(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        state_parts = []
        used_obs_keys = set()
        for key in self.state_keys:
            candidate_keys = [key] + STATE_KEY_ALIASES.get(key, [])
            selected_key = next((k for k in candidate_keys if k in obs and k not in used_obs_keys), None)
            if selected_key is not None:
                state_parts.append(np.asarray(obs[selected_key], dtype=np.float32).ravel())
                used_obs_keys.add(selected_key)

        if not state_parts:
            raise KeyError(f"Could not infer state keys from observation keys: {list(obs.keys())}")

        return np.concatenate(state_parts, axis=0)

    def _get_image(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        if self.image_key not in obs:
            available = [k for k in obs.keys() if k.endswith("_image")]
            if not available:
                raise KeyError(f"No image key found in observation dict keys: {list(obs.keys())}")
            img = np.asarray(obs[available[0]])
        else:
            img = np.asarray(obs[self.image_key])

        img = np.ascontiguousarray(img)

        if img.dtype != np.uint8:
            if np.max(img) <= 1.0:
                img = (img * 255.0).clip(0, 255).astype(np.uint8)
            else:
                img = img.clip(0, 255).astype(np.uint8)
        return img

    def _success(self) -> bool:
        check_success = getattr(self.env, "_check_success", None)
        if callable(check_success):
            return bool(check_success())
        return False

    def _extract_object_pose(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        obj_info = {}

        obj_state = obs.get("object-state") if isinstance(obs, dict) else None
        if isinstance(obj_state, np.ndarray) and obj_state.size >= 7:
            obj_info["obj_init_pos"] = np.asarray(obj_state[:3], dtype=np.float32).copy()
            obj_info["obj_init_quat"] = np.asarray(obj_state[3:7], dtype=np.float32).copy()
            return obj_info

        sim = getattr(self.env, "sim", None)
        body_id = getattr(self.env, "cube_body_id", None)
        if sim is not None and body_id is not None:
            obj_info["obj_init_pos"] = np.asarray(sim.data.body_xpos[body_id], dtype=np.float32).copy()
            obj_info["obj_init_quat"] = np.asarray(sim.data.body_xquat[body_id], dtype=np.float32).copy()

        return obj_info

    def reset(self, seed: Optional[int] = None):
        applied_seed = None
        if seed is not None:
            applied_seed = int(seed)
        elif self._deterministic_reset_order:
            applied_seed = int(self.init_seed + self._episode_count)
            self._episode_count += 1

        if applied_seed is not None:
            self._apply_seed(applied_seed)

        obs = self.env.reset()
        state = self._extract_state(obs)
        image = self._get_image(obs)
        info = {"success": self._success()}
        if applied_seed is not None:
            info["reset_seed"] = applied_seed
        info.update(self._extract_object_pose(obs))
        return image, state, info

    def step(self, action: np.ndarray):
        obs, reward, done, info = self.env.step(action)
        if self._render:
            self.env.render()
        state = self._extract_state(obs)
        image = self._get_image(obs)
        info = dict(info) if info is not None else {}
        info["success"] = info.get("success", self._success())
        
        return image, state, float(reward), bool(done), info

    def close(self):
        self.env.close()
