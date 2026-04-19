import numpy as np
from collections import namedtuple
import torch

def calculate_reward(obs):
    sensors = obs[:, :6]
    fuel = obs[:, 8:9]

    sensors = torch.abs(sensors)
    sensors = torch.clamp(sensors, 0, 1)
    sensors = 1.0 - sensors
    sensors = torch.cat([sensors, fuel], dim=1)
    # value = torch.prod(sensors, dim=1)
    value = torch.mean(sensors, dim=1)

    return value

def normalize_state(state):
    # Define the normalization factors for the observation space
    # These values are based on the observation space of the LunarLander-v2 environment
    # and are used to scale the observations to a range of approximately [-1, 1]
    # x, y, v_x, v_y, angle, v_angle
    NORMALIZATION_FACTORS = np.array([1, 1.75, 4, 4, 3.1415927, 5, 1, 1], dtype=np.float32)
    state = np.array(state, dtype=np.float32)
    state = state / NORMALIZATION_FACTORS
    state = np.clip(state, -1.0, 1.0)
    return state

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