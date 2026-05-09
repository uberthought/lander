import torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
import os
import shutil
import argparse
from collections import deque

from shared.observation import create_observation, is_failure_state, calculate_reward
from shared.ReplayBuffer import ReplayBuffer
from model.qmodel import QModel
from .offline_training import print_q_snr, print_q_action_breakdown
from .live_training import _step_action

import time


def train(env, seconds, train_every_n_episodes, sample_multiplier, video_folder):
    q_model = QModel()

    replay_buffer = ReplayBuffer()
    replay_buffer0 = deque(maxlen=40000)

    start_time = time.time()
    episode = 0
    while time.time() - start_time < seconds:
        replay_buffer.increment_episode()
        episode += 1
        done = False
        truncated = False
        t = 0

        prev_state, _ = env.reset()
        prev_state = np.concatenate((prev_state, [0.0, 1.0, 0.0, 0.0, 0.0]))

        do_training = train_every_n_episodes > 0 and episode % train_every_n_episodes == 0

        while not done and not truncated:
            t += 1
            action = q_model.get_best_action(prev_state)
            next_state, done = _step_action(action, env)

            transition = create_observation(prev_state, action, next_state)
            replay_buffer.add(transition)
            replay_buffer0.append(transition)

            prev_state = next_state

        video_path = f"{video_folder}/{env._video_name}.mp4"
        env.reset()
        final_value = calculate_reward(
            torch.tensor(prev_state, dtype=torch.float32).unsqueeze(0)
        ).item()
        v_int = int(round(final_value * 10000))
        video_name_with_final_value = f"{video_folder}/episode_{episode+1}_t_{t}_v_{v_int}.mp4"
        shutil.move(video_path, video_name_with_final_value)

        json_files = [f for f in os.listdir(video_folder) if f.endswith(".json")]
        for json_file in json_files:
            os.remove(os.path.join(video_folder, json_file))
        mp4_files = [f for f in os.listdir(video_folder) if f.startswith("rl-video-episode-") and f.endswith(".mp4")]
        for mp4_file in mp4_files:
            os.remove(os.path.join(video_folder, mp4_file))

        if do_training:
            print_q_snr(q_model, replay_buffer0, remain_time=seconds - (time.time() - start_time))
            # print_q_action_breakdown(q_model, list(replay_buffer0))

            replay_buffer0 = list(replay_buffer0)
            sample_len = len(replay_buffer0) * sample_multiplier

            start_train_time = time.time()
            i = 0
            while time.time() - start_train_time < 10:
                training_sample = replay_buffer.sample(sample_len) + replay_buffer0
                q_model.train(training_sample)
                i += 1

            q_model.save()

            try:
                replay_buffer.save()
            except Exception as e:
                print(f"Autosave failed: {e}")

            replay_buffer0 = deque(maxlen=40000)


def main():
    parser = argparse.ArgumentParser(description="Live Q-training for the LunarLander-v3 environment.")
    parser.add_argument("--seconds", type=int, default=60, help="Number of seconds to train")
    parser.add_argument('--train-every', type=int, default=4, help='Number of episodes between training sessions')
    parser.add_argument('--sample-multiplier', type=int, default=4, help='Replay-buffer sample size = recent-buffer length * this multiplier')
    args = parser.parse_args()

    np.set_printoptions(formatter={'float': lambda x: "{0:+0.4f}".format(x)})

    shutil.rmtree("./videos", ignore_errors=True)

    env = gym.make("LunarLander-v3", continuous=True, render_mode="rgb_array")
    env = RecordVideo(env, video_folder="./videos", episode_trigger=lambda x: True, disable_logger=True)

    train(env, args.seconds, args.train_every, args.sample_multiplier, video_folder="./videos")

    env.close()


if __name__ == "__main__":
    main()
