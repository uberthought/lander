import numpy as np
from collections import namedtuple

import torch

def calculate_value(obs, done):
    sensors = obs[:, :6]
    fuel = obs[:, 8:9]

    done_indices = done.nonzero(as_tuple=True)[0]

    sensors = torch.abs(sensors)
    sensors = torch.clamp(sensors, 0, 1)
    sensors = 1.0 - sensors
    sensors = torch.cat([sensors, fuel], dim=1)
    abridged_sensors = sensors[:, [0, 1]]
    value = torch.prod(abridged_sensors, dim=1)
    value[done_indices] = torch.prod(sensors[done_indices, :], dim=1)

    return value
    

Observation = namedtuple('Observation', ['state', 'action', 'next_state', 'done'])