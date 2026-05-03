
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
        return torch.clamp(state[..., :6], mins, maxs)
    return np.clip(state[..., :6], CLIP_MIN, CLIP_MAX)


def calculate_reward(obs):
    # Normalize observations using the module-level NORMALIZATION_FACTORS
    # Expect `obs` to be a torch tensor of shape (batch, state_dim)
    if not isinstance(obs, torch.Tensor):
        obs = torch.tensor(obs, dtype=torch.float32)

    # Make normalization factors available at module level
    NORMALIZATION_FACTORS = np.array([1, 1.75, 4, 4, 3.1415927, 5], dtype=np.float32)
    # index 8 is done (0 or 1)
    norm = torch.tensor(NORMALIZATION_FACTORS, dtype=obs.dtype, device=obs.device)
    obs_norm = obs / norm

    sensors = obs_norm[:, :6]
    sensors = torch.abs(sensors)
    sensors = torch.clamp(sensors, 0, 1)
    sensors = 1.0 - sensors


    reward = torch.norm(sensors, dim=1) / np.sqrt(6.0)
    # reward = torch.norm(sensors[:, [0, 1, 4]], dim=1) / np.sqrt(3.0)

    return reward

def create_observation(prev_state, action, next_state):
    return Observation(prev_state, action, next_state)


# prev_state is the previous observation,
# action is the action taken (single int),
# next_state is the resulting observation after taking the action (next_state[-1] == done flag)
Observation = namedtuple('Observation', ['prev_state', 'action', 'next_state'])