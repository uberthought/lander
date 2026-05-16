import argparse
import time
import numpy as np
import torch
import gymnasium as gym
from collections import deque

from observation import create_observation, calculate_reward, is_done_state
from ReplayBuffer import ReplayBuffer
from world import WorldModel
from actor import ActorModel
from configuration import POSSIBLE_ACTIONS
from offline_training import print_snr

LEG_DONE_THRESHOLD = 0.9
POS_DONE_THRESHOLD = 1.5
DONE_THRESHOLD = 0.9


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


def _run_imaginary_episode(seed_state, actor_model, world_model, max_steps, episode_id):
    transitions = []
    rewards = []
    prev_state = seed_state.copy()
    t = 0

    while True:
        t += 1
        action = actor_model.get_best_action(prev_state)
        next_state = prev_state.copy()
        next_state[:6] = world_model.predict(prev_state, action)[:6]
        next_state = _clamp_state(next_state)
        done = _is_done(next_state, t, max_steps)
        next_state[8] = float(done)
        next_state[9:13] = 0.0
        next_state[9 + int(action)] = 1.0

        transitions.append(create_observation(prev_state, action, next_state))

        next_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        # rewards.append(calculate_reward(next_tensor).item())

        prev_state = next_state
        if done:
            break

    return transitions


def _step_action(action, env):
    if action == 0:
        action0 = np.array([0.0, 0.0], dtype=np.float32)
    elif action == 1:
        action0 = np.array([0.0, 1.0], dtype=np.float32)
    elif action == 2:
        action0 = np.array([1.0, 0.0], dtype=np.float32)
    elif action == 3:
        action0 = np.array([0.0, -1.0], dtype=np.float32)
    next_state, _, done, truncated, _ = env.step(action0)
    done = done or truncated or is_done_state(next_state)
    onehot = np.zeros(POSSIBLE_ACTIONS, dtype=np.float32)
    onehot[action] = 1.0
    next_state = np.concatenate((next_state, [float(done)], onehot))
    return next_state, done


def _evaluate_in_real_env(actor_model, episodes):
    env = gym.make("LunarLander-v3", continuous=True)
    finals = []
    lengths = []
    for ep in range(episodes):
        prev_state, _ = env.reset()
        prev_state = np.concatenate((prev_state, [0.0, 1.0, 0.0, 0.0, 0.0]))
        done = False
        t = 0
        while not done:
            t += 1
            action = actor_model.get_best_action(prev_state)
            prev_state, done = _step_action(action, env)
        final = calculate_reward(
            torch.tensor(prev_state, dtype=torch.float32).unsqueeze(0)
        ).item()
        finals.append(final)
        lengths.append(t)
        print(f"  eval ep {ep + 1}/{episodes}  steps={t:3d}  final_reward={final:+.4f}")
    env.close()

    arr = np.array(finals, dtype=np.float32)
    print(
        f"Real-env eval over {episodes} episodes: "
        f"mean={arr.mean():+.4f}  std={arr.std():.4f}  "
        f"min={arr.min():+.4f}  max={arr.max():+.4f}  "
        f"avg_len={np.mean(lengths):.1f}"
    )


def train(seconds, rollout_steps, eval_episodes, sample_size):
    np.set_printoptions(formatter={'float': lambda x: f"{x:+0.4f}"})

    world_model = WorldModel()
    actor_model = ActorModel(model_path="checkpoints/actor_model_world.pt", load=True)
    replay_buffer = ReplayBuffer()

    if len(replay_buffer) == 0:
        raise RuntimeError(
            "Replay buffer is empty — run collect_random_data.py first."
        )

    recent_rewards = deque(maxlen=50)
    recent_lengths = deque(maxlen=50)

    start_time = time.time()
    iteration = 0
    imagined_transitions = []

    while time.time() - start_time < seconds:
        iteration += 1

        world_sample = replay_buffer.sample(2 ** sample_size)
        world_model.train(world_sample)

        seed_state = np.array(replay_buffer.sample(1)[0].next_state, dtype=np.float32)
        episode_id = replay_buffer.max_episode + iteration

        transitions = _run_imaginary_episode(
            seed_state=seed_state,
            actor_model=actor_model,
            world_model=world_model,
            max_steps=rollout_steps,
            episode_id=episode_id,
        )
        imagined_transitions.extend(transitions)

        if iteration % 10 == 0:
            if imagined_transitions:
                actor_model.train(imagined_transitions)
            imagined_transitions = []

            actor_model.save()
            world_model.save()

            elapsed = time.time() - start_time
            print_snr(world_model, world_sample, remain_time=seconds - elapsed, actor_model=actor_model)

    actor_model.save()
    world_model.save()
    print(f"Done. {iteration} iterations in {seconds}s. Actor saved to {actor_model.model_path}.")

    if eval_episodes > 0:
        print(f"\nEvaluating fresh-trained actor in real LunarLander-v3 env...")
        _evaluate_in_real_env(actor_model, eval_episodes)


def main():
    parser = argparse.ArgumentParser(description="Train actor via world model imaginary rollouts.")
    parser.add_argument("--seconds", type=int, default=600, help="Wall-clock budget in seconds")
    parser.add_argument("--rollout-steps", type=int, default=1000, help="Max steps per imaginary episode")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Real-env evaluation episodes after training (0 to skip)")
    parser.add_argument("--sample-size", type=int, default=16, help="Replay buffer batch size exponent for world-model training (2^N)")
    args = parser.parse_args()

    train(args.seconds, args.rollout_steps, args.eval_episodes, args.sample_size)


if __name__ == "__main__":
    main()
