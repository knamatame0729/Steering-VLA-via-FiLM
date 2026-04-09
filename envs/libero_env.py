"""LIBERO environment wrapper for VLA training.

LIBERO is a benchmark for lifelong robot learning with language-conditioned tasks.
Reference: https://github.com/Lifelong-Robot-Learning/LIBERO

Installation:
    git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
    cd LIBERO
    pip install -e .
"""

import numpy as np

try:
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    LIBERO_AVAILABLE = True
except ImportError:
    LIBERO_AVAILABLE = False
    print("Warning: LIBERO not installed. Install with:")
    print("  git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git")
    print("  cd LIBERO && pip install -e .")


class LIBEROEnvWrapper:
    """
    Wraps a LIBERO environment into a simple interface compatible with VLA training:
    - reset() -> (image, state, info)
    - step(action) -> (image, state, reward, done, info)
    
    LIBERO provides 4 task suites:
    - libero_spatial: 10 tasks with spatial reasoning
    - libero_object: 10 tasks with object manipulation
    - libero_goal: 10 tasks with goal-directed behavior
    - libero_100: 100 diverse long-horizon tasks
    """
    
    # Available task suites
    TASK_SUITES = ['libero_spatial', 'libero_object', 'libero_goal', 'libero_100']
    
    def __init__(self, task_suite: str = 'libero_spatial', task_id: int = 0, image_size: int = 128, seed: int = 42, camera_name: str = 'agentview',):
        """
        Initialize LIBERO environment.
        
        Args:
            task_suite: One of 'libero_spatial', 'libero_object', 'libero_goal', 'libero_100'
            task_id: Task index within the suite (0-9 for most suites, 0-99 for libero_100)
            image_size: Image observation size (default 128x128)
            seed: Random seed
            camera_name: Camera view ('agentview', 'robot0_eye_in_hand', etc.)
        """
        if not LIBERO_AVAILABLE:
            raise ImportError("LIBERO is not installed. See installation instructions above.")
        
        assert task_suite in self.TASK_SUITES, f"task_suite must be one of {self.TASK_SUITES}"
        
        self.task_suite = task_suite
        self.task_id = task_id
        self.image_size = image_size
        self.seed = seed
        self.camera_name = camera_name
        
        # Get task suite and task description
        self.benchmark_dict = benchmark.get_benchmark_dict()
        self.task_suite_obj = self.benchmark_dict[task_suite]()
        self.task = self.task_suite_obj.get_task(task_id)
        self.task_name = self.task.name
        self.task_description = self.task.language  # Natural language instruction
        
        # Get BDDL file for the task
        self.task_bddl_file = self.task_suite_obj.get_task_bddl_file_path(task_id)
        
        # Create environment
        self.env = OffScreenRenderEnv(
            bddl_file=self.task_bddl_file,
            camera_heights=image_size,
            camera_widths=image_size,
        )
        
        # Get environment dimensions
        self.env.seed(seed)
        obs = self.env.reset()

        # Check keys in observation
        print(obs.keys())
        for key, value in obs.items():
            print(f"{key}: shape={value.shape}, dtype={value.dtype}")
        
        self.state_dim = self._extract_state(obs).shape[0]
        self.action_dim = self.env.action_spec[0].shape[0]  # (low, high)
        self.obs_shape = (image_size, image_size, 3)
        
        print(f"LIBERO Environment initialized:")
        print(f"  Task suite: {task_suite}")
        print(f"  Task: {self.task_name}")
        print(f"  Instruction: {self.task_description}")
        print(f"  State dim: {self.state_dim}")
        print(f"  Action dim: {self.action_dim}")
    
    def _extract_state(self, obs: dict) -> np.ndarray:
        """
        Extract robot state from LIBERO observation dict.
        
        LIBERO obs contains:
        - 'robot0_eef_pos': End-effector position (3,)
        - 'robot0_eef_quat': End-effector quaternion (4,)
        - 'robot0_gripper_qpos': Gripper joint positions (2,)
        - 'robot0_joint_pos': Joint positions (7,)
        - plus object states depending on the task
        """
        state_parts = []
        
        # Robot state
        if 'robot0_eef_pos' in obs:
            state_parts.append(obs['robot0_eef_pos'])
        if 'robot0_eef_quat' in obs:
            state_parts.append(obs['robot0_eef_quat'])
        if 'robot0_gripper_qpos' in obs:
            state_parts.append(obs['robot0_gripper_qpos'])
        if 'robot0_joint_pos' in obs:
            state_parts.append(obs['robot0_joint_pos'])
        
        # Fallback: use flat observation if available
        if len(state_parts) == 0:
            if 'robot0_proprio-state' in obs:
                state_parts.append(obs['robot0_proprio-state'])
            else:
                # Try to find any non-image observation
                for key, value in obs.items():
                    if isinstance(value, np.ndarray) and value.ndim == 1:
                        state_parts.append(value)
        
        if len(state_parts) == 0:
            raise KeyError(f"Could not extract state. Available keys: {list(obs.keys())}")
        
        return np.concatenate(state_parts, axis=-1).astype(np.float32)
    
    def _get_image(self, obs: dict) -> np.ndarray:
        """
        Extract image observation from LIBERO observation dict.
        
        LIBERO obs contains camera images like:
        - 'agentview_image': (H, W, 3)
        - 'robot0_eye_in_hand_image': (H, W, 3)
        """
        image_key = f"{self.camera_name}_image"
        if image_key in obs:
            img = obs[image_key]
        elif 'agentview_image' in obs:
            img = obs['agentview_image']
        else:
            # Find any image key
            for key in obs:
                if 'image' in key and isinstance(obs[key], np.ndarray) and obs[key].ndim == 3:
                    img = obs[key]
                    break
            else:
                raise KeyError(f"No image found in obs. Keys: {list(obs.keys())}")
        
        # Ensure uint8 format
        if img.dtype != np.uint8:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        
        return img
    
    def get_instruction(self) -> str:
        """Return the natural language instruction for the current task."""
        return self.task_description
    
    def reset(self, seed: int = None) -> tuple:
        """
        Reset the environment.
        
        Returns:
            image: (H, W, 3) uint8 image observation
            state: (state_dim,) float32 robot state
            info: dict with task info
        """
        if seed is not None:
            self.env.seed(seed)
        
        obs = self.env.reset()
        
        image = self._get_image(obs)
        state = self._extract_state(obs)
        info = {
            'task_name': self.task_name,
            'instruction': self.task_description,
        }
        
        return image, state, info
    
    def step(self, action: np.ndarray) -> tuple:
        """
        Take an environment step.
        
        Args:
            action: (action_dim,) robot action
            
        Returns:
            image: (H, W, 3) uint8 image observation
            state: (state_dim,) float32 robot state
            reward: float reward
            done: bool episode termination
            info: dict with 'success' key
        """
        obs, reward, done, info = self.env.step(action)
        
        image = self._get_image(obs)
        state = self._extract_state(obs)
        
        return image, state, reward, done, info
    
    def close(self):
        """Close the environment."""
        self.env.close()
    
    @classmethod
    def list_tasks(cls, task_suite: str = 'libero_spatial') -> list:
        """
        List all available tasks in a suite.
        
        Args:
            task_suite: One of 'libero_spatial', 'libero_object', 'libero_goal', 'libero_100'
            
        Returns:
            List of (task_id, task_name, instruction) tuples
        """
        if not LIBERO_AVAILABLE:
            raise ImportError("LIBERO is not installed.")
        
        benchmark_dict = benchmark.get_benchmark_dict()
        suite = benchmark_dict[task_suite]()
        
        tasks = []
        for i in range(suite.n_tasks):
            task = suite.get_task(i)
            tasks.append((i, task.name, task.language))
        
        return tasks


def get_libero_demo_dataset(
    task_suite: str = 'libero_spatial',
    task_id: int = 0,
    num_demos: int = 50,
):
    """
    Load pre-collected demonstration data from LIBERO.
    
    LIBERO provides expert demonstrations for each task.
    
    Args:
        task_suite: Task suite name
        task_id: Task index
        num_demos: Number of demonstrations to load
        
    Returns:
        dict with 'images', 'states', 'actions', 'texts' keys
    """
    if not LIBERO_AVAILABLE:
        raise ImportError("LIBERO is not installed.")
    
    from libero.libero import get_libero_path
    import h5py
    import os
    
    benchmark_dict = benchmark.get_benchmark_dict()
    suite = benchmark_dict[task_suite]()
    task = suite.get_task(task_id)
    
    # Get demo file path
    demo_path = os.path.join(
        get_libero_path("datasets"),
        task_suite,
        f"{task.name}_demo.hdf5"
    )
    
    if not os.path.exists(demo_path):
        raise FileNotFoundError(
            f"Demo file not found: {demo_path}\n"
            "Download LIBERO datasets first: python -m libero.download_datasets"
        )
    
    images = []
    states = []
    actions = []
    texts = []
    
    with h5py.File(demo_path, 'r') as f:
        demos = list(f['data'].keys())[:num_demos]
        
        for demo_key in demos:
            demo = f['data'][demo_key]
            
            # Get observations
            obs = demo['obs']
            if 'agentview_image' in obs:
                demo_images = obs['agentview_image'][:]
            else:
                # Find image key
                for key in obs.keys():
                    if 'image' in key:
                        demo_images = obs[key][:]
                        break
            
            # Get states (robot proprioception)
            if 'robot0_proprio-state' in obs:
                demo_states = obs['robot0_proprio-state'][:]
            else:
                # Concatenate available state info
                state_parts = []
                for key in ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos']:
                    if key in obs:
                        state_parts.append(obs[key][:])
                demo_states = np.concatenate(state_parts, axis=-1)
            
            # Get actions
            demo_actions = demo['actions'][:]
            
            # Add to lists
            for i in range(len(demo_actions)):
                images.append(demo_images[i])
                states.append(demo_states[i])
                actions.append(demo_actions[i])
                texts.append(task.language)
    
    return {
        'images': np.stack(images, axis=0),
        'states': np.stack(states, axis=0).astype(np.float32),
        'actions': np.stack(actions, axis=0).astype(np.float32),
        'texts': texts,
        'instruction': task.language,
        'task_name': task.name,
    }


if __name__ == "__main__":
    # Test the wrapper
    if LIBERO_AVAILABLE:
        print("Listing LIBERO spatial tasks:")
        tasks = LIBEROEnvWrapper.list_tasks('libero_spatial')
        for task_id, task_name, instruction in tasks:
            print(f"  [{task_id}] {task_name}: {instruction}")
        
        print("\nTesting environment wrapper...")
        env = LIBEROEnvWrapper(task_suite='libero_spatial', task_id=0)
        image, state, info = env.reset()
        print(f"Image shape: {image.shape}, dtype: {image.dtype}")
        print(f"State shape: {state.shape}")
        print(f"Instruction: {info['instruction']}")
        
        # Take random action
        action = np.random.uniform(-1, 1, env.action_dim)
        image, state, reward, done, info = env.step(action)
        print(f"Step - reward: {reward}, done: {done}")
        
        env.close()
    else:
        print("LIBERO not available. Install it to test.")
