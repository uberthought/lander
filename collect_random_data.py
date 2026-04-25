import gymnasium as gym
import numpy as np
import os
import shutil
import argparse
from observation import create_observation
from ReplayBuffer import ReplayBuffer

from configuration import POSSIBLE_ACTIONS

def step_action(action, env, fuel):
    if action == 0:
        action0 = np.array([0.0, 0.0], dtype=np.float32)
    elif action == 1:
        action0 = np.array([0.0, 1.0], dtype=np.float32)
    elif action == 2:
        action0 = np.array([1.0, 0.0], dtype=np.float32)
    elif action == 3:
        action0 = np.array([0.0, -1.0], dtype=np.float32)
    next_state, _, done, truncated, _ = env.step(action0)
    if action != 0:
        fuel -= 1
    done = done or truncated
    angle = next_state[4]
    next_state = np.concatenate((next_state, [fuel / 1000.0, np.sin(angle), np.cos(angle)]))

    return next_state, done, fuel

def collect(env, episodes, video_folder):
    replay_buffer = ReplayBuffer()
    shutil.rmtree(video_folder, ignore_errors=True)
    os.makedirs(video_folder, exist_ok=True)

    for episode in range(episodes):
        replay_buffer.increment_episode()
        done = False
        truncated = False
        t = 0
        prev_state, _ = env.reset()
        fuel = 1000
        prev_state = np.concatenate((prev_state, [fuel / 1000.0, np.sin(prev_state[4]), np.cos(prev_state[4])]))

        while not done and not truncated:
            t += 1

            actions = np.random.randint(0, POSSIBLE_ACTIONS, size=1).tolist()

            for action in actions:
                next_state, done, fuel = step_action(action, env, fuel)
                if done:
                    break

            transition = create_observation(episode, t, prev_state, actions, next_state, done)
            replay_buffer.add(transition)
            prev_state = next_state

    try:
        replay_buffer.save()
        print(f"[autosave] Saved {len(replay_buffer)} observations -> {replay_buffer.filename}")
    except Exception as e:
        print(f"Autosave failed: {e}")

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
