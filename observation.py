import numpy as np
from collections import namedtuple

import torch

def calculate_reward(obs):
    sensors = obs[:, :6]
    legs = obs[:, 6:8]
    fuel = obs[:, 8:9]
    done = obs[:, 9:10]

    sensors = torch.abs(sensors)
    sensors = torch.clamp(sensors, 0, 1)
    sensors = 1.0 - sensors
    sensors = torch.cat([sensors, fuel], dim=1)
    value = torch.prod(sensors, dim=1)

    # 25% bonus for each landed leg
    legs_landed = torch.sum(legs, dim=1)
    value = value * 0.5 + value * 0.25 * legs_landed

    return value
    

Observation = namedtuple('Observation', ['state', 'action', 'next_state'])