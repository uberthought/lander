"""Imaginary rollouts through the world model.

Generates synthetic episodes by stepping the WorldModel under the actor's
policy. Lives in its own module so entrypoints import rollout logic from a
library rather than from another training script.
"""
import numpy as np
import torch

from observation import create_observation, is_done_state, LEG_TOUCH_Y
from configuration import DONE_THRESHOLD


def _clamp_state(state):
    state[6] = np.clip(state[6], 0.0, 1.0)
    state[7] = np.clip(state[7], 0.0, 1.0)
    state[8] = np.clip(state[8], 0.0, 1.0)
    state[9:13] = np.clip(state[9:13], 0.0, 1.0)
    return state


def _is_done(state, t, max_steps):
    if t >= max_steps:
        return True
    if state[8] >= DONE_THRESHOLD:
        return True
    if is_done_state(state):
        return True
    return False


def run_imaginary_episode(seed_state, actor_model, world_model, max_steps=1000):
    transitions = []
    prev_state = seed_state.copy()
    t = 0

    while True:
        t += 1
        action = actor_model.get_best_action(prev_state)
        next_state = prev_state.copy()
        next_state[:6] = world_model.predict(prev_state, action)[:6]
        # Legs are not sensors in imagination: synthesize ground contact from
        # altitude. Both legs set together (geometry can't distinguish L/R).
        contact = next_state[1] < LEG_TOUCH_Y
        next_state[6] = 1.0 if contact else 0.0
        next_state[7] = 1.0 if contact else 0.0
        next_state = _clamp_state(next_state)
        done = _is_done(next_state, t, max_steps)
        next_state[8] = float(done)
        next_state[9:13] = 0.0
        next_state[9 + int(action)] = 1.0

        transitions.append(create_observation(prev_state, action, next_state))

        next_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

        prev_state = next_state
        if done:
            break

    return transitions
