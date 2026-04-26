import argparse
import time
import numpy as np
import torch
from collections import deque

from observation import create_observation, calculate_reward
from ReplayBuffer import ReplayBuffer
from world import WorldModel
from actor import ActorModel

LEG_DONE_THRESHOLD = 0.9
POS_DONE_THRESHOLD = 1.5
DONE_THRESHOLD = 0.9


def _clamp_state(state):
    state[6] = np.clip(state[6], 0.0, 1.0)
    state[7] = np.clip(state[7], 0.0, 1.0)
    state[8] = np.clip(state[8], 0.0, 1.0)
    return state


def _is_done(state, t, max_steps):
    if t >= max_steps:
        return True
    if state[8] >= DONE_THRESHOLD:
        return True
    return False


def _run_imaginary_episode(seed_state, actor_model, world_model, max_steps, episode_id):
    transitions = []
    rewards = []
    prev_state = seed_state.copy()
    t = 0

    while True:
        t += 1
        action = actor_model.get_best_action(prev_state)
        delta = world_model.predict(prev_state, action)
        next_state = _clamp_state(prev_state + delta)
        done = _is_done(next_state, t, max_steps)
        next_state[8] = float(done)

        transitions.append(create_observation(prev_state, action, next_state))

        next_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        # rewards.append(calculate_reward(next_tensor).item())

        prev_state = next_state
        if done:
            break

    return transitions


def train(seconds, rollout_steps):
    np.set_printoptions(formatter={'float': lambda x: f"{x:+0.4f}"})

    world_model = WorldModel()
    actor_model = ActorModel()
    replay_buffer = ReplayBuffer()

    if len(replay_buffer) == 0:
        raise RuntimeError(
            "Replay buffer is empty — run collect_random_data.py first."
        )

    recent_rewards = deque(maxlen=50)
    recent_lengths = deque(maxlen=50)

    start_time = time.time()
    iteration = 0

    while time.time() - start_time < seconds:
        iteration += 1

        seed_state = np.array(replay_buffer.sample(1)[0].next_state, dtype=np.float32)
        episode_id = replay_buffer.max_episode + iteration

        transitions = _run_imaginary_episode(
            seed_state=seed_state,
            actor_model=actor_model,
            world_model=world_model,
            max_steps=rollout_steps,
            episode_id=episode_id,
        )

        if transitions:
            actor_model.train(transitions)

        if iteration % 10 == 0:
            actor_model.save()

        elapsed = time.time() - start_time
        print(
            f"Iter {iteration:4d}  t={seconds - elapsed:.0f}s  "
            f"transitions={len(transitions):4d}  "
        )

    actor_model.save()
    print(f"Done. {iteration} iterations in {seconds}s. Actor saved.")


def main():
    parser = argparse.ArgumentParser(description="Train actor via world model imaginary rollouts.")
    parser.add_argument("--seconds", type=int, default=60, help="Wall-clock budget in seconds")
    parser.add_argument("--rollout-steps", type=int, default=20, help="Max steps per imaginary episode")
    args = parser.parse_args()

    train(args.seconds, args.rollout_steps)


if __name__ == "__main__":
    main()
