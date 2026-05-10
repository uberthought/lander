import gymnasium as gym
import numpy as np
import os
import shutil
import argparse
from shared.observation import create_observation, is_done_state
from shared.ReplayBuffer import ReplayBuffer

from shared.configuration import POSSIBLE_ACTIONS

def step_action(action, env):
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

def collect(env, episodes, video_folder):
    replay_buffer = ReplayBuffer()
    shutil.rmtree(video_folder, ignore_errors=True)
    os.makedirs(video_folder, exist_ok=True)

    for episode in range(episodes):
        replay_buffer.increment_episode()
        done = False
        truncated = False
        prev_state, _ = env.reset()
        prev_state = np.concatenate((prev_state, [0.0, 1.0, 0.0, 0.0, 0.0]))

        while not done and not truncated:
            action = np.random.randint(0, POSSIBLE_ACTIONS)
            next_state, done = step_action(action, env)

            transition = create_observation(prev_state, action, next_state)
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
