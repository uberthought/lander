import numpy as np
from collections import namedtuple

import torch

def calculate_value(obs):
    sensors = obs[:, :6]
    sensors = torch.abs(sensors)
    sensors = torch.clamp(sensors, 0, 1)
    sensors = 1 - sensors

    # for obs that are not done, only use x, y, angle, so remove v_x, v_y, v_angle from the sensors
    done_indexes = (obs[:, 9] < 0.5).nonzero(as_tuple=True)[0].unsqueeze(1)
    indexes_to_ignore = torch.tensor([2, 3, 5], device=obs.device)
    sensors[done_indexes, indexes_to_ignore] = 1.0

    # append the fuel level to the sensors
    fuel = obs[:, 8:9]
    sensors = torch.cat([sensors, fuel], dim=1)

    value = torch.prod(sensors, dim=1)

    return value
    

Observation = namedtuple('Observation', ['state', 'action', 'next_state'])