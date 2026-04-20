
import numpy as np
from collections import namedtuple
import torch


def calculate_reward(obs):
    # Normalize observations using the module-level NORMALIZATION_FACTORS
    # Expect `obs` to be a torch tensor of shape (batch, state_dim)
    if not isinstance(obs, torch.Tensor):
        obs = torch.tensor(obs, dtype=torch.float32)

    # Make normalization factors available at module level
    NORMALIZATION_FACTORS = np.array([1, 1.75, 4, 4, 3.1415927, 5, 1, 1, 1, 1, 1], dtype=np.float32)
    # indices 9,10 are sin(angle) and cos(angle), already in [-1,1] so norm factor = 1
    norm = torch.tensor(NORMALIZATION_FACTORS, dtype=obs.dtype, device=obs.device)
    obs_norm = obs / norm

    sensors = obs_norm[:, :6]
    fuel = obs_norm[:, 8:9]

    sensors = torch.abs(sensors)
    sensors = torch.clamp(sensors, 0, 1)
    sensors = 1.0 - sensors
    sensors = torch.cat([sensors, fuel], dim=1)
    value = torch.mean(sensors, dim=1)

    return value

def create_observation(episode, time, prev_state, actions, next_state, done):
    done = float(done)
    return Observation(episode, time, prev_state, actions, next_state, done)


# Define a simple Observation namedtuple where
# episode is the episode id,
# time is the timestep within the episode,
# prev_state is the previous observation,
# actions is a list of actions taken,
# next_state is the resulting observation after taking the action,
# done is whether the episode ended after this transition
Observation = namedtuple('Observation', ['episode', 'time', 'prev_state', 'actions', 'next_state', 'done'])