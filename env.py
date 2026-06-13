"""Environment adapter for LunarLander-v3.

Single home for everything that mediates between the discrete 0-3 action space
this project uses and the continuous Gymnasium env: the action encoding, the
env-step (with done-folding and 13-dim state assembly), the reset padding, and
env construction. Every entrypoint imports from here so the action mapping and
state layout live in exactly one place.
"""
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from observation import is_done_state
from configuration import POSSIBLE_ACTIONS

# Discrete action -> 2-D continuous LunarLander action vector.
# 0=off, 1=right, 2=left, 3=reverse (main engine).
ACTION_VECTORS = {
    0: np.array([0.0, 0.0], dtype=np.float32),
    1: np.array([0.0, 1.0], dtype=np.float32),
    2: np.array([1.0, 0.0], dtype=np.float32),
    3: np.array([0.0, -1.0], dtype=np.float32),
}

# Appended to gym's 8-dim observation to form the initial 13-dim state:
# [done=0, prev-action one-hot = action 0 (null)].
RESET_PADDING = [0.0, 1.0, 0.0, 0.0, 0.0]


def make_env(record=False, video_folder="./videos", render_mode="rgb_array"):
    """Build the LunarLander-v3 env. Wraps RecordVideo when record=True
    (which requires render_mode='rgb_array'); pass render_mode=None to skip
    rendering entirely for headless evaluation."""
    env = gym.make("LunarLander-v3", continuous=True, render_mode=render_mode)
    if record:
        env = RecordVideo(env, video_folder=video_folder, episode_trigger=lambda x: True, disable_logger=True)
    return env


def reset_state(env):
    """Reset the env and return the padded 13-dim initial state."""
    state, _ = env.reset()
    return np.concatenate((state, RESET_PADDING))


def step_action(action, env):
    """Step the env with the encoded action; return (13-dim next state, done).
    done folds gym's done/truncated together with is_done_state."""
    next_state, _, done, truncated, _ = env.step(ACTION_VECTORS[action])
    done = done or truncated or is_done_state(next_state)
    onehot = np.zeros(POSSIBLE_ACTIONS, dtype=np.float32)
    onehot[action] = 1.0
    next_state = np.concatenate((next_state, [float(done)], onehot))
    return next_state, done
