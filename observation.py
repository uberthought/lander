
import numpy as np
from collections import namedtuple
import torch


def calculate_reward(obs):
    # Normalize observations using the module-level NORMALIZATION_FACTORS
    # Expect `obs` to be a torch tensor of shape (batch, state_dim)
    if not isinstance(obs, torch.Tensor):
        obs = torch.tensor(obs, dtype=torch.float32)

    # Make normalization factors available at module level
    NORMALIZATION_FACTORS = np.array([1, 1.75, 4, 4, 3.1415927, 5, 1, 1, 1], dtype=np.float32)
    # index 8 is done (0 or 1)
    norm = torch.tensor(NORMALIZATION_FACTORS, dtype=obs.dtype, device=obs.device)
    obs_norm = obs / norm

    sensors = obs_norm[:, :6]
    sensors = torch.abs(sensors)
    sensors = torch.clamp(sensors, 0, 1)
    sensors = 1.0 - sensors


    # reward = torch.norm(sensors, dim=1) / np.sqrt(7.0)

    reward = torch.norm(sensors[:, [0, 1, 4]], dim=1) / np.sqrt(3.0)

    return reward

def create_observation(prev_state, action, next_state):
    return Observation(prev_state, action, next_state)


# prev_state is the previous observation,
# action is the action taken (single int),
# next_state is the resulting observation after taking the action (next_state[-1] == done flag)
Observation = namedtuple('Observation', ['prev_state', 'action', 'next_state'])