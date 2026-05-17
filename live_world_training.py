import argparse
import os
import shutil
import time

import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import RecordVideo

from observation import create_observation, calculate_reward
from ReplayBuffer import ReplayBuffer
from actor import ActorModel
from world import WorldModel
from actor_training import _step_action
from world_training import _run_imaginary_episode
from offline_training import print_snr


def _save_video(env, video_folder, episode, t, final_state):
    video_path = f"{video_folder}/{env._video_name}.mp4"
    env.reset()
    final_value = calculate_reward(
        torch.tensor(final_state, dtype=torch.float32).unsqueeze(0)
    ).item()
    v_int = int(round(final_value * 10000))
    target = f"{video_folder}/episode_{episode}_t_{t}_v_{v_int}.mp4"
    if os.path.exists(video_path):
        shutil.move(video_path, target)
    for f in os.listdir(video_folder):
        if f.endswith(".json"):
            os.remove(os.path.join(video_folder, f))
        elif f.startswith("rl-video-episode-") and f.endswith(".mp4"):
            os.remove(os.path.join(video_folder, f))


def train(env, seconds, episodes, sample_size, rollout_steps, video_folder):
    actor_model = ActorModel(model_path="checkpoints/actor_model_world.pt", load=True)
    world_model = WorldModel()
    replay_buffer = ReplayBuffer()

    if len(replay_buffer) == 0:
        raise RuntimeError("Replay buffer empty - run collect_random_data.py first.")

    start_time = time.time()
    episode = 0

    while time.time() - start_time < seconds:
        # Collect `episodes` real-env episodes
        new_transitions = []
        for _ in range(episodes):
            if time.time() - start_time >= seconds:
                break
            replay_buffer.increment_episode()
            episode += 1

            prev_state, _ = env.reset()
            prev_state = np.concatenate((prev_state, [0.0, 1.0, 0.0, 0.0, 0.0]))
            done = False
            t = 0

            while not done:
                t += 1
                action = actor_model.get_best_action(prev_state)
                next_state, done = _step_action(action, env)

                transition = create_observation(prev_state, action, next_state)
                replay_buffer.add(transition)
                new_transitions.append(transition)

                prev_state = next_state

            _save_video(env, video_folder, episode, t, prev_state)

        # Step 1: train world model on new data + sample of old data
        old_sample = replay_buffer.sample(len(new_transitions) * sample_size)
        world_model.train(new_transitions + old_sample)

        # Step 2: collect `episodes` imagined transitions, then train actor on them
        imagined_transitions = []
        for i in range(episodes):
            seed_state = np.array(
                replay_buffer.sample(1)[0].next_state, dtype=np.float32
            )
            transitions = _run_imaginary_episode(
                seed_state=seed_state,
                actor_model=actor_model,
                world_model=world_model,
                # max_steps=rollout_steps,
            )
            imagined_transitions.extend(transitions)


        # if imagined_transitions:
        #     actor_model.train(imagined_transitions)

        actor_model.train(new_transitions + old_sample)

        actor_model.save()
        world_model.save()
        try:
            replay_buffer.save()
        except Exception as e:
            print(f"Autosave failed: {e}")

        remain = seconds - (time.time() - start_time)
        print_snr(world_model, old_sample, remain_time=remain, actor_model=actor_model)


def main():
    parser = argparse.ArgumentParser(
        description="Live env collection + world-model + imaginary-rollout actor training."
    )
    parser.add_argument("--seconds", type=int, default=600, help="Wall-clock budget in seconds")
    parser.add_argument("--episodes", type=int, default=8, help="Real episodes collected per training cycle; also the imagined-episode count")
    parser.add_argument("--sample-size", type=int, default=16, help="Multiplier on new-transitions count for the replay-buffer sample size")
    parser.add_argument("--rollout-steps", type=int, default=1000, help="Max steps per imaginary episode")
    parser.add_argument("--video-folder", type=str, default="./videos", help="Folder to write MP4 files to")
    args = parser.parse_args()

    np.set_printoptions(formatter={'float': lambda x: "{0:+0.4f}".format(x)})
    shutil.rmtree(args.video_folder, ignore_errors=True)

    env = gym.make("LunarLander-v3", continuous=True, render_mode="rgb_array")
    env = RecordVideo(env, video_folder=args.video_folder, episode_trigger=lambda x: True, disable_logger=True)
    try:
        train(
            env,
            args.seconds,
            args.episodes,
            args.sample_size,
            args.rollout_steps,
            args.video_folder,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
