import numpy as np
from collections import namedtuple

import tensorflow as tf

# Define the normalization factors for the observation space
# These values are based on the observation space of the LunarLander-v2 environment
# and are used to scale the observations to a range of approximately [-1, 1]
# x, y, v_x, v_y, angle, v_angle

# used to normalize the sensor values to [0, 1]
NORMALIZATION_FACTORS = tf.constant([1, 1.75, 4, 4, 3.1415927, 5], dtype=tf.float32)

def calculate_value(obs):
    """
    Calculates a scalar value from an observation, which is used as a proxy for the
    "goodness" of the state. This value is then used as an input to the actor model.

    Args:
        obs: The observation including if it's done from the environment.

    Returns:
        A scalar value representing the goodness of the state
    """

    # if only a single observation is provided, reshape it to be a batch of one

    obs = tf.convert_to_tensor(obs, dtype=tf.float32)

    if len(obs.shape) == 1:
        obs = tf.expand_dims(obs, axis=0)

    sensors = obs[:, :6]
    sensors = sensors / tf.reshape(NORMALIZATION_FACTORS, [-1, 6])
    sensors = tf.abs(sensors)
    sensors = tf.clip_by_value(sensors, 0, 1)
    sensors = 1 - sensors
    # append the fuel level to the sensors
    fuel = obs[:, 8:9]
    sensors = tf.concat([sensors, fuel], axis=1)
    value = tf.reduce_prod(sensors, axis=1)
    leg0 = obs[:, 6]
    leg1 = obs[:, 7]
    leg_multiplier = (leg0 + leg1) / 2.0
    value = value * 0.7 + leg_multiplier * 0.3

    return value

Observation = namedtuple('Observation', ['state', 'action', 'next_state'])