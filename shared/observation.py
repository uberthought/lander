
import numpy as np
from collections import namedtuple
import torch

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


FAILURE_ANGLE = np.pi / 2  # quarter turn — past this, the ship is considered to have failed
FAILURE_Y_MIN = -0.5  # below ground / off the bottom of the operational envelope
FAILURE_Y_MAX = 2.0   # above the operational ceiling
LANDING_BONUS = 0.5      # per-step bonus when leg(s) on pad
LANDING_X_RADIUS = 0.3   # x distance from centerline that counts as "on pad"


def is_failure_state(state):
    if abs(float(state[4])) > FAILURE_ANGLE:
        return True
    y = float(state[1])
    if y < FAILURE_Y_MIN or y > FAILURE_Y_MAX:
        return True
    return False


def calculate_reward(obs):
    # Expects `obs` to be the full state (batch, 9) or the 6-dim sensor slice.
    # Clips internally; the leg/x-position bonus needs the full state.
    if not isinstance(obs, torch.Tensor):
        obs = torch.tensor(obs, dtype=torch.float32)

    NORMALIZATION_FACTORS = np.array([1, 1.75, 4, 4, 3.1415927, 5], dtype=np.float32)
    norm = torch.tensor(NORMALIZATION_FACTORS, dtype=obs.dtype, device=obs.device)
    sensors = clip_state(obs)[..., :6] / norm
    sensors = torch.abs(sensors)
    sensors = torch.clamp(sensors, 0, 1)
    sensors = 1.0 - sensors

    position = torch.norm(sensors[:, [0, 1]], dim=1) / np.sqrt(2.0)
    other = torch.norm(sensors[:, [2, 3, 4, 5]], dim=1) / np.sqrt(4.0)

    if obs.shape[-1] >= 13:
        on_pad = obs[:, 0].abs() < LANDING_X_RADIUS
        engines_off = obs[:, 9] > 0.5
        left_bonus = LANDING_BONUS * ((obs[:, 6] > 0.5) & on_pad & engines_off).float()
        right_bonus = LANDING_BONUS * ((obs[:, 7] > 0.5) & on_pad & engines_off).float()
        other = other + left_bonus + right_bonus
    return position * other

def create_observation(prev_state, action, next_state):
    return Observation(prev_state, action, next_state)


# prev_state is the previous observation,
# action is the action taken (single int),
# next_state is the resulting observation after taking the action (next_state[-1] == done flag)
Observation = namedtuple('Observation', ['prev_state', 'action', 'next_state'])