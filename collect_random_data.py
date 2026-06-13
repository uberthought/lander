import numpy as np
import os
import shutil
import argparse
from observation import create_observation
from ReplayBuffer import ReplayBuffer
from env import step_action, reset_state, make_env

from configuration import POSSIBLE_ACTIONS

def collect(env, episodes, video_folder):
    replay_buffer = ReplayBuffer()
    shutil.rmtree(video_folder, ignore_errors=True)
    os.makedirs(video_folder, exist_ok=True)

    for episode in range(episodes):
        replay_buffer.increment_episode()
        done = False
        truncated = False
        prev_state = reset_state(env)

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
    env = make_env()
    collect(env, episodes, video_folder="./videos")
    env.close()

if __name__ == "__main__":
    main()
