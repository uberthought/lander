import numpy as np
from collections import namedtuple

import torch

def calculate_value(obs):
    sensors = obs[:, :6]
    fuel = obs[:, 8:9]

    sensors = torch.abs(sensors)
    sensors = torch.clamp(sensors, 0, 1)
    sensors = torch.cat([sensors, fuel], dim=1)

    # increase the penalty for vertical speed
    sensors[:, [4]] **= 2

    value = torch.prod(sensors, dim=1)

    return value
    

Observation = namedtuple('Observation', ['state', 'action', 'next_state'])