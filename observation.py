import numpy as np
from collections import namedtuple

import torch

def calculate_reward(obs):
    sensors = obs[:, :6]
    fuel = obs[:, 8:9]

    fuel **= 3.0

    sensors = torch.abs(sensors)
    sensors = torch.clamp(sensors, 0, 1)
    sensors = 1.0 - sensors
    sensors = torch.cat([sensors, fuel], dim=1)

    value = torch.prod(sensors, dim=1)

    return value
    

Observation = namedtuple('Observation', ['state', 'action', 'next_state'])