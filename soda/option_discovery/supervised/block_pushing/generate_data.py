"""
Generate expert block pushing demonstrations with oracle labels.

Saves N episodes to data/raw/block_pushing/ as individual .npz files.
Each file contains: images (T,H,W,3), state (T,obs_dim), actions (T,2),
option_id (T,) with labels 0-3.

Option mapping:
  0 = reach_first  (navigating to first block)
  1 = push_first   (pushing/orienting first block)
  2 = reach_second (navigating to second block)
  3 = push_second  (pushing/orienting second block)

Run from repo root:
  conda activate soda
  python soda/option_discovery/supervised/block_pushing/generate_data.py --n-episodes 200
"""
import sys, os, types, argparse
import numpy as np
from tqdm import tqdm

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, os.path.join(ROOT, "third_party", "diffusion_policy"))

# ── tf_agents shim ────────────────────────────────────────────────────────────
def _install_tf_agents_shim():
    ta = types.ModuleType("tf_agents")
    policies_mod   = types.ModuleType("tf_agents.policies")
    py_policy_mod  = types.ModuleType("tf_agents.policies.py_policy")
    traj_mod       = types.ModuleType("tf_agents.trajectories")
    ps_mod         = types.ModuleType("tf_agents.trajectories.policy_step")
    ts_mod         = types.ModuleType("tf_agents.trajectories.time_step")
    typing_mod     = types.ModuleType("tf_agents.typing")
    types_submod   = types.ModuleType("tf_agents.typing.types")

    class PyPolicy:
        def __init__(self, *a, **kw): pass
        def get_initial_state(self, n): return None
        def action(self, time_step, policy_state):
            return self._action(time_step, policy_state)

    class PolicyStep:
        def __init__(self, action=None): self.action = action

    class TimeStep:
        FIRST = 0
        def __init__(self, step_type=0, observation=None):
            self.step_type = step_type
            self.observation = observation or {}
        def is_first(self): return self.step_type == 0

    py_policy_mod.PyPolicy   = PyPolicy
    ps_mod.PolicyStep        = PolicyStep
    ts_mod.TimeStep          = TimeStep

    ta.policies = policies_mod
    ta.trajectories = traj_mod
    ta.typing = typing_mod
    policies_mod.py_policy = py_policy_mod
    traj_mod.policy_step = ps_mod
    traj_mod.time_step   = ts_mod
    typing_mod.types     = types_submod

    for name, mod in [
        ("tf_agents", ta),
        ("tf_agents.policies", policies_mod),
        ("tf_agents.policies.py_policy", py_policy_mod),
        ("tf_agents.trajectories", traj_mod),
        ("tf_agents.trajectories.policy_step", ps_mod),
        ("tf_agents.trajectories.time_step", ts_mod),
        ("tf_agents.typing", typing_mod),
        ("tf_agents.typing.types", types_submod),
    ]:
        sys.modules[name] = mod

_install_tf_agents_shim()

# ── imports ───────────────────────────────────────────────────────────────────
from diffusion_policy.env.block_pushing.block_pushing_multimodal import BlockPushMultimodal
from diffusion_policy.env.block_pushing.oracles.multimodal_push_oracle import MultimodalOrientedPushOracle

class _GymEnvWrapper:
    """Thin wrapper that adds tf_agents stubs so the oracle __init__ doesn't crash."""
    def __init__(self, env):
        self._env = env
    def __getattr__(self, name):
        return getattr(self._env, name)
    def time_step_spec(self): return None
    def action_spec(self):    return None
import diffusion_policy.env.block_pushing.block_pushing as _bp
import diffusion_policy.env.block_pushing.block_pushing_multimodal as _bpm
import pybullet_data

# ── patch URDF paths ──────────────────────────────────────────────────────────
_ASSETS = os.path.join(ROOT, "third_party", "diffusion_policy",
                       "diffusion_policy", "env", "block_pushing", "assets")
_bp.BLOCK_URDF_PATH     = os.path.join(_ASSETS, "block.urdf")
_bp.WORKSPACE_URDF_PATH = os.path.join(_ASSETS, "workspace.urdf")
_bp.ZONE_URDF_PATH      = os.path.join(_ASSETS, "zone.urdf")
_bp.INSERT_URDF_PATH    = os.path.join(_ASSETS, "insert.urdf")
_bp.PLANE_URDF_PATH     = os.path.join(pybullet_data.getDataPath(), "plane.urdf")
_bpm.BLOCK2_URDF_PATH   = os.path.join(_ASSETS, "block2.urdf")
_bpm.ZONE2_URDF_PATH    = os.path.join(_ASSETS, "zone2.urdf")

# ── oracle phase → option id ──────────────────────────────────────────────────
REACH_PHASES = {"move_to_pre_block", "move_to_block",
                "return_to_first_preblock", "return_to_origin"}
PUSH_PHASES  = {"push_block", "orient_block_left", "orient_block_right"}

OPTION_NAMES = ["reach_first", "push_first", "reach_second", "push_second"]

def oracle_to_option(oracle):
    """Map current oracle state to option id 0-3."""
    phase = oracle.phase
    is_second = getattr(oracle, '_has_switched', False)
    if phase in PUSH_PHASES:
        return 3 if is_second else 1
    else:
        return 2 if is_second else 0


# ── generation ────────────────────────────────────────────────────────────────
def generate(n_episodes: int, image_size: tuple, out_dir: str, max_steps: int = 350):
    os.makedirs(out_dir, exist_ok=True)

    env = BlockPushMultimodal(
        image_size=image_size,
        control_frequency=10.0,
        goal_dist_tolerance=0.04,
    )

    succeeded = 0
    for ep_idx in tqdm(range(n_episodes), desc="Generating episodes"):
        env.seed(ep_idx)
        obs = env.reset()

        oracle = MultimodalOrientedPushOracle(_GymEnvWrapper(env), goal_dist_tolerance=0.04)
        oracle.reset()

        images, states, actions, option_ids = [], [], [], []

        # Build initial time_step for oracle
        from tf_agents.trajectories.time_step import TimeStep
        time_step = TimeStep(step_type=0, observation=obs)

        done = False
        step = 0
        while not done and step < max_steps:
            # Record current frame
            img = obs.get("rgb", None)
            if img is not None:
                images.append(img.astype(np.uint8))
            state = np.concatenate([v for k, v in obs.items() if k != "rgb"])
            states.append(state.astype(np.float32))

            # Oracle action
            action_step = oracle.action(time_step, None)
            action = action_step.action
            actions.append(action.astype(np.float32))

            # Option label from oracle state AFTER action computed
            option_ids.append(oracle_to_option(oracle))

            # Step env
            obs, reward, done, info = env.step(action)
            time_step = TimeStep(step_type=2 if done else 1, observation=obs)
            step += 1

        if len(images) == 0:
            continue

        out_path = os.path.join(out_dir, f"episode_{ep_idx:04d}.npz")
        np.savez_compressed(
            out_path,
            images    = np.array(images,     dtype=np.uint8),
            state     = np.array(states,     dtype=np.float32),
            actions   = np.array(actions,    dtype=np.float32),
            option_id = np.array(option_ids, dtype=np.int32),
        )
        succeeded += 1

    env.close()
    print(f"\nSaved {succeeded}/{n_episodes} episodes to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-episodes", type=int, default=200)
    parser.add_argument("--image-size", type=int, nargs=2, default=[240, 320],
                        metavar=("H", "W"))
    parser.add_argument("--out-dir", default="data/raw/block_pushing")
    parser.add_argument("--max-steps", type=int, default=350)
    args = parser.parse_args()

    generate(
        n_episodes = args.n_episodes,
        image_size = tuple(args.image_size),
        out_dir    = args.out_dir,
        max_steps  = args.max_steps,
    )
