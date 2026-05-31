
import numpy as np
from collections import namedtuple
import torch

from configuration import STATE_SIZE

# Clip bounds derived from pure in-flight data (no legs touching, no done transitions)
CLIP_MIN = np.array([-1.0, -0.5, -2.5, -2.5, -3.5, -3.5], dtype=np.float32)
CLIP_MAX = np.array([ 1.0,  4.0,  2.5,  2.0,  3.5,  3.5], dtype=np.float32)

def clip_state(state):
    if isinstance(state, torch.Tensor):
        mins = torch.tensor(CLIP_MIN, dtype=state.dtype, device=state.device)
        maxs = torch.tensor(CLIP_MAX, dtype=state.dtype, device=state.device)
        clipped = torch.clamp(state[..., :6], mins, maxs)
        return torch.cat([clipped, state[..., 6:]], dim=-1)
    clipped = np.clip(state[..., :6], CLIP_MIN, CLIP_MAX)
    return np.concatenate([clipped, state[..., 6:]], axis=-1)


FAILURE_ANGLE = np.pi / 1.5  # quarter turn — past this, the ship is considered to have failed
FAILURE_Y_MIN = -0.5  # below ground / off the bottom of the operational envelope
FAILURE_Y_MAX = 2.0   # above the operational ceiling
LANDING_BONUS = 0.5      # per-step bonus when leg(s) on pad
LANDING_X_RADIUS = 0.3   # x distance from centerline that counts as "on pad"
# Altitude below which a leg is treated as touching ground in imaginary rollouts
# (real legs come from gym sensors). Derived empirically from replay-buffer
# leg-flip transitions: ~90% of real contacts occur at y < 0.05.
LEG_TOUCH_Y = 0.05


def is_done_state(state):
    # Tipped past recoverable angle (failed).
    if abs(float(state[4])) > FAILURE_ANGLE:
        return True
    # Out of the vertical envelope: crashed below the floor or flew above the ceiling (failed).
    y = float(state[1])
    if y < FAILURE_Y_MIN or y > FAILURE_Y_MAX:
        return True
    # Any legs touching the ground (landed).
    if float(state[6]) > 0.5 or float(state[7]) > 0.5:
        return True
    return False


def calculate_reward(obs):
    if not isinstance(obs, torch.Tensor):
        obs = torch.tensor(obs, dtype=torch.float32)
    assert obs.shape[-1] == STATE_SIZE, f"calculate_reward expected last dim {STATE_SIZE}, got {obs.shape[-1]}"

    NORMALIZATION_FACTORS = np.array([1, 1.75, 4, 4, 3.1415927, 5], dtype=np.float32)
    norm = torch.tensor(NORMALIZATION_FACTORS, dtype=obs.dtype, device=obs.device)
    sensors = clip_state(obs)
    sensors = sensors[..., :6] / norm
    sensors = torch.abs(sensors)
    sensors = 1.0 - sensors
    sensors = torch.clamp(sensors, 0, 1)
    done = obs[:, 8] > 0.5

    # y = sensors[..., 1:2]
    # sensors = torch.cat([sensors[..., :2], sensors[..., 2:6] * y], dim=-1)

    on_pad = obs[:, 0].abs() < LANDING_X_RADIUS
    # engines_off = obs[:, 9] > 0.5
    left_bonus = LANDING_BONUS * ((obs[:, 6] > 0.5) & on_pad).float()
    right_bonus = LANDING_BONUS * ((obs[:, 7] > 0.5) & on_pad).float()
    left_penalty = -LANDING_BONUS * ((obs[:, 6] > 0.5) & ~on_pad).float()
    right_penalty = -LANDING_BONUS * ((obs[:, 7] > 0.5) & ~on_pad).float()

    # reward = torch.norm(sensors, dim=1) / np.sqrt(6.0)
    reward = torch.prod(sensors, dim=1)
    reward = reward + left_bonus + right_bonus + left_penalty + right_penalty
    reward = torch.where(done & ~on_pad, torch.zeros_like(reward), reward)
    reward = torch.clamp(reward, 0, 2)
    return reward

def create_observation(prev_state, action, next_state):
    return Observation(prev_state, action, next_state)


# prev_state is the previous observation,
# action is the action taken (single int),
# next_state is the resulting observation after taking the action (next_state[-1] == done flag)
Observation = namedtuple('Observation', ['prev_state', 'action', 'next_state'])