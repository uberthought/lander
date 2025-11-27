import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
import os
import shutil
import argparse
from observation import Observation
from ReplayBuffer import ReplayBuffer

NORMALIZATION_FACTORS = np.array([1, 1.75, 4, 4, 3.1415927, 5, 1, 1], dtype=np.float32)

def collect(env, episodes, video_folder):
    replay_buffer = ReplayBuffer()
    shutil.rmtree(video_folder, ignore_errors=True)
    os.makedirs(video_folder, exist_ok=True)

    for episode in range(episodes):
        done = False
        truncated = False
        t = 0
        obs, _ = env.reset()
        obs = normalize_observation(obs)
        obs = np.concatenate((obs, [1.0]))  # fuel level
        obs = np.concatenate((obs, [0.0]))  # done flag
        fuel = 1000
        while not done and not truncated:
            t += 1
            action = np.random.randint(0, 4)
            if action == 0:
                action0 = np.array([0.0, 0.0], dtype=np.float32)
            elif action == 1:
                action0 = np.array([0.0, 1.0], dtype=np.float32)
            elif action == 2:
                action0 = np.array([1.0, 0.0], dtype=np.float32)
            elif action == 3:
                action0 = np.array([0.0, -1.0], dtype=np.float32)
            next_obs, _, done, truncated, _ = env.step(action0)
            next_obs = normalize_observation(next_obs)
            if done or truncated:
                next_obs[0:6] = obs[0:6]
            if action != 0:
                fuel -= 1
            done = done or truncated
            next_obs = np.concatenate((next_obs, [fuel / 1000.0]))
            next_obs = np.concatenate((next_obs, [1.0 if done else 0.0]))
            transition = Observation(obs, action, next_obs)
            replay_buffer.add(transition)
            obs = next_obs

    try:
        replay_buffer.save()
        print(f"[autosave] Saved {len(replay_buffer)} observations -> {replay_buffer.filename}")
    except Exception as e:
        print(f"Autosave failed: {e}")

def normalize_observation(obs):
    obs = np.array(obs, dtype=np.float32)
    obs = obs / NORMALIZATION_FACTORS
    obs = np.clip(obs, -1.0, 1.0)
    return obs

def main():
    parser = argparse.ArgumentParser(description="Collect random data for LunarLander-v3.")
    parser.add_argument("--episodes", type=int, default=256, help="Number of episodes to collect")
    episodes = parser.parse_args().episodes
    np.set_printoptions(formatter={'float': lambda x: "{0:+0.4f}".format(x)})
    env = gym.make("LunarLander-v3", continuous=True, render_mode="rgb_array")
    collect(env, episodes, video_folder="./videos")
    env.close()

if __name__ == "__main__":
    main()
