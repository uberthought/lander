#!/usr/bin/env python3
import argparse
import time
import numpy as np
import torch
from collections import deque

from observation import calculate_reward
from ReplayBuffer import ReplayBuffer
from world import WorldModel
from actor import ActorModel
from env import step_action, reset_state, make_env
from rollout import run_imaginary_episode
from configuration import ACTOR_WORLD_MODEL_PATH
from metrics import print_snr


def _evaluate_in_real_env(actor_model, episodes):
    env = make_env(render_mode=None)
    finals = []
    lengths = []
    for ep in range(episodes):
        prev_state = reset_state(env)
        done = False
        t = 0
        while not done:
            t += 1
            action = actor_model.get_best_action(prev_state)
            prev_state, done = step_action(action, env)
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
    actor_model = ActorModel(model_path=ACTOR_WORLD_MODEL_PATH, load=True)
    replay_buffer = ReplayBuffer()

    if len(replay_buffer) == 0:
        raise RuntimeError(
            "Replay buffer is empty — run collect_random_data.py first."
        )

    start_time = time.time()
    iteration = 0
    imagined_transitions = []

    while time.time() - start_time < seconds:
        iteration += 1

        world_sample = replay_buffer.sample(2 ** sample_size)
        world_model.train(world_sample)

        seed_state = np.array(replay_buffer.sample(1)[0].next_state, dtype=np.float32)

        transitions = run_imaginary_episode(
            seed_state=seed_state,
            actor_model=actor_model,
            world_model=world_model,
            # max_steps=rollout_steps
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

    actor_model.save()
    world_model.save()

    print(f"Done. {iteration} iterations in {seconds}s. Actor saved to {actor_model.model_path}.")

    # if eval_episodes > 0:
    #     print(f"\nEvaluating fresh-trained actor in real LunarLander-v3 env...")
    #     _evaluate_in_real_env(actor_model, eval_episodes)


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
