"""
Image-based BlockPush env runner for SODA training evaluation.

Follows the same interface as PushTImageRunner. Key differences:
  - Uses BlockPushMultimodal(image_size=...) for RGB observations
  - Renames 'rgb' → 'image' and converts (H,W,3) uint8 → (3,H,W) float32
  - Success metric: fraction of episodes where both blocks reach their targets
"""
import sys, os, types, math, pathlib, collections
import numpy as np
import torch
import tqdm
import dill
import wandb
import wandb.sdk.data_types.video as wv

# ── tf_agents shim (needed to import oracle module chain) ────────────────────
def _install_tf_agents_shim():
    if "tf_agents" in sys.modules:
        return
    ta = types.ModuleType("tf_agents")
    for sub in ["policies", "policies.py_policy", "trajectories",
                "trajectories.policy_step", "trajectories.time_step",
                "typing", "typing.types"]:
        sys.modules[f"tf_agents.{sub}"] = types.ModuleType(f"tf_agents.{sub}")

    class PyPolicy:
        def __init__(self, *a, **kw): pass
        def get_initial_state(self, n): return None

    class PolicyStep:
        def __init__(self, action=None): self.action = action

    class TimeStep:
        def __init__(self, step_type=0, observation=None):
            self.step_type = step_type
            self.observation = observation or {}
        def is_first(self): return self.step_type == 0

    sys.modules["tf_agents.policies.py_policy"].PyPolicy = PyPolicy
    sys.modules["tf_agents.trajectories.policy_step"].PolicyStep = PolicyStep
    sys.modules["tf_agents.trajectories.time_step"].TimeStep = TimeStep
    sys.modules["tf_agents"] = ta

_install_tf_agents_shim()

# ── URDF path patching ────────────────────────────────────────────────────────
def _patch_urdf_paths():
    import pybullet_data
    _here = os.path.dirname(os.path.abspath(__file__))
    _root = os.path.dirname(os.path.dirname(_here))
    _assets = os.path.join(_root, "third_party", "diffusion_policy",
                           "diffusion_policy", "env", "block_pushing", "assets")

    import diffusion_policy.env.block_pushing.block_pushing as _bp
    import diffusion_policy.env.block_pushing.block_pushing_multimodal as _bpm
    _bp.BLOCK_URDF_PATH     = os.path.join(_assets, "block.urdf")
    _bp.WORKSPACE_URDF_PATH = os.path.join(_assets, "workspace.urdf")
    _bp.ZONE_URDF_PATH      = os.path.join(_assets, "zone.urdf")
    _bp.INSERT_URDF_PATH    = os.path.join(_assets, "insert.urdf")
    _bp.PLANE_URDF_PATH     = os.path.join(pybullet_data.getDataPath(), "plane.urdf")
    _bpm.BLOCK2_URDF_PATH   = os.path.join(_assets, "block2.urdf")
    _bpm.ZONE2_URDF_PATH    = os.path.join(_assets, "zone2.urdf")

_patch_urdf_paths()

import gym
import gym.spaces as spaces
from diffusion_policy.env.block_pushing.block_pushing_multimodal import BlockPushMultimodal
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner


class _BlockPushImageObsWrapper(gym.ObservationWrapper):
    """Renames 'rgb' → 'image' and converts (H,W,3) uint8 → (3,H,W) float32."""

    def __init__(self, env):
        super().__init__(env)
        old_spaces = dict(env.observation_space.spaces)
        img_shape = old_spaces.pop("rgb").shape       # (H, W, 3)
        img_shape_chw = (img_shape[2], img_shape[0], img_shape[1])  # (3, H, W)
        old_spaces["image"] = spaces.Box(0.0, 1.0, shape=img_shape_chw, dtype=np.float32)
        self.observation_space = spaces.Dict(old_spaces)

    def observation(self, obs):
        obs = dict(obs)
        rgb = obs.pop("rgb").astype(np.float32) / 255.0
        obs["image"] = np.moveaxis(rgb, -1, 0)   # (H,W,3) → (3,H,W)
        return obs


class BlockPushImageRunner(BaseImageRunner):
    def __init__(
        self,
        output_dir,
        n_train=6,
        n_train_vis=2,
        train_start_seed=0,
        n_test=22,
        n_test_vis=4,
        test_start_seed=100000,
        max_steps=350,
        n_obs_steps=2,
        n_action_steps=8,
        fps=10,
        crf=22,
        render_size=96,
        tqdm_interval_sec=5.0,
        n_envs=None,
    ):
        super().__init__(output_dir)

        if n_envs is None:
            n_envs = n_train + n_test

        steps_per_render = max(10 // fps, 1)

        def env_fn():
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    _BlockPushImageObsWrapper(
                        BlockPushMultimodal(
                            image_size=(render_size, render_size),
                            control_frequency=10.0,
                        )
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps, codec="h264", input_pix_fmt="rgb24",
                        crf=crf, thread_type="FRAME", thread_count=1,
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render,
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
            )

        env_seeds, env_prefixs, env_init_fn_dills = [], [], []

        for i in range(n_train):
            seed = train_start_seed + i
            enable_render = i < n_train_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    fn = pathlib.Path(output_dir) / "media" / (wv.util.generate_id() + ".mp4")
                    fn.parent.mkdir(parents=False, exist_ok=True)
                    env.env.file_path = str(fn)
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append("train/")
            env_init_fn_dills.append(dill.dumps(init_fn))

        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    fn = pathlib.Path(output_dir) / "media" / (wv.util.generate_id() + ".mp4")
                    fn.parent.mkdir(parents=False, exist_ok=True)
                    env.env.file_path = str(fn)
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append("test/")
            env_init_fn_dills.append(dill.dumps(init_fn))

        self.env = AsyncVectorEnv([env_fn] * n_envs)
        self.env_fns = [env_fn] * n_envs
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

    def run(self, policy: BaseImagePolicy):
        device = policy.device
        dtype = policy.dtype
        env = self.env

        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_n = end - start
            this_slice = slice(start, end)

            init_fns = self.env_init_fn_dills[this_slice]
            if len(init_fns) < n_envs:
                init_fns = init_fns + [self.env_init_fn_dills[0]] * (n_envs - len(init_fns))

            env.call_each("run_dill_function", args_list=[(x,) for x in init_fns])

            obs = env.reset()
            policy.reset()

            done = False
            pbar = tqdm.tqdm(
                total=self.max_steps,
                desc=f"BlockPushImageRunner chunk {chunk_idx+1}/{n_chunks}",
                leave=False, mininterval=self.tqdm_interval_sec,
            )

            episode_rewards = [[] for _ in range(n_envs)]

            while not done:
                np_obs_dict = dict(obs)
                obs_dict = dict_apply(np_obs_dict,
                    lambda x: torch.from_numpy(x).to(device=device))

                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)

                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to("cpu").numpy())
                action = np_action_dict["action"]

                obs, reward, done, info = env.step(action)
                done = np.all(done[:this_n])

                for i in range(this_n):
                    episode_rewards[i].append(reward[i])

                pbar.update(self.n_action_steps)

            pbar.close()

            # Collect results
            for i in range(this_n):
                all_rewards[start + i] = episode_rewards[i]

            # Collect video paths
            video_paths = env.call("get_video_path")
            for i in range(this_n):
                all_video_paths[start + i] = video_paths[i]

        # Compute metrics
        log_data = {}
        prefix_reward_map = collections.defaultdict(list)

        for i, (prefix, rewards) in enumerate(zip(self.env_prefixs, all_rewards)):
            if rewards is None:
                continue
            # reward=1 means both blocks at target (from BlockPushEventManager)
            max_reward = float(np.max(rewards)) if rewards else 0.0
            prefix_reward_map[prefix].append(max_reward)

        for prefix, rewards in prefix_reward_map.items():
            mean_reward = np.mean(rewards)
            log_data[f"{prefix}mean_score"] = mean_reward

        # Log videos
        for prefix, video_path in zip(self.env_prefixs, all_video_paths):
            if video_path is not None and pathlib.Path(video_path).exists():
                log_data[f"{prefix}sim_video"] = wandb.Video(video_path)

        return log_data
