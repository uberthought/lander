import argparse
import os
import shutil

import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import RecordVideo

from actor import ActorModel
from actor_training import step_action
from observation import calculate_reward


def play(model_path, episodes, video_folder):
    shutil.rmtree(video_folder, ignore_errors=True)

    env = gym.make("LunarLander-v3", continuous=True, render_mode="rgb_array")
    env = RecordVideo(env, video_folder=video_folder, episode_trigger=lambda x: True, disable_logger=True)

    actor_model = ActorModel(model_path=model_path, load=True)

    try:
        for episode in range(1, episodes + 1):
            done = False
            t = 0

            prev_state, _ = env.reset()
            prev_state = np.concatenate((prev_state, [0.0, 1.0, 0.0, 0.0, 0.0]))

            while not done:
                t += 1
                action = actor_model.get_best_action(prev_state)
                next_state, done = step_action(action, env)
                prev_state = next_state

            video_path = f"{video_folder}/{env._video_name}.mp4"
            env.reset()

            final_value = calculate_reward(
                torch.tensor(prev_state, dtype=torch.float32).unsqueeze(0)
            ).item()
            v_int = int(round(final_value * 10000))
            target = f"{video_folder}/play_episode_{episode}_t_{t}_v_{v_int}.mp4"
            if os.path.exists(video_path):
                shutil.move(video_path, target)
                print(f"Saved {target}")
            else:
                print(f"Warning: expected video at {video_path} not found")

            for f in os.listdir(video_folder):
                if f.endswith(".json"):
                    os.remove(os.path.join(video_folder, f))
                elif f.startswith("rl-video-episode-") and f.endswith(".mp4"):
                    os.remove(os.path.join(video_folder, f))
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description="Run a saved actor checkpoint on LunarLander-v3 and record video.")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to record")
    parser.add_argument("--video-folder", type=str, default="./videos", help="Folder to write MP4 files to")
    parser.add_argument("--model-path", type=str, default="checkpoints/actor_model_world.pt", help="Path to actor checkpoint")
    args = parser.parse_args()

    np.set_printoptions(formatter={'float': lambda x: "{0:+0.4f}".format(x)})

    play(args.model_path, args.episodes, args.video_folder)


if __name__ == "__main__":
    main()
