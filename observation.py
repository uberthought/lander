import numpy as np
from collections import namedtuple

import torch

def calculate_value(obs):
    sensors = obs[:, :6]
    fuel = obs[:, 8:9]

    sensors = torch.abs(sensors)
    sensors = torch.clamp(sensors, 0, 1)
    sensors = 1.0 - sensors
    sensors = torch.cat([sensors, fuel], dim=1)
    value = torch.prod(sensors, dim=1)

    return value
    

# Define a simple Observation namedtuple where
# episode is the episode id,
# time is the timestep within the episode,
# state is the current observation,
# action is the action taken,
# next_state is the resulting observation after taking the action,
# done is whether the episode ended after this transition
Observation = namedtuple('Observation', ['episode', 'time', 'state', 'action', 'next_state', 'done'])