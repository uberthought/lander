import torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
import os
import shutil
import argparse
from collections import deque

from observation import create_observation
from ReplayBuffer import ReplayBuffer
from world import WorldModel, STATE_PARAMETER_NAMES
from actor import ActorModel
from training import compute_validation_snr

import time

from configuration import POSSIBLE_ACTIONS

CONTINUOUS_STATE_DIM = 6
VALIDATION_SAMPLE_SIZE = 2 ** 10

def _compute_snr(state, predicted):
    state = state[:CONTINUOUS_STATE_DIM]
    predicted = predicted[:CONTINUOUS_STATE_DIM]
    signal = np.mean(state ** 2)
    noise = np.mean((state - predicted) ** 2)
    if noise == 0 or signal == 0:
        return 0.0
    return 10 * np.log10(signal / noise)

def _step_action(action, env, fuel):
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

def train(env, seconds, train_every_n_episodes, video_folder):
    world_model = WorldModel()
    actor_model = ActorModel(world_model)

    # Main long-term buffer (persistent) and recent buffer for on-policy-ish updates
    # State shape is 8 from LunarLander-v3 plus fuel level
    replay_buffer = ReplayBuffer()
    replay_buffer0 = deque(maxlen=40000)
    validation_sample = replay_buffer.sample(VALIDATION_SAMPLE_SIZE)

    # remove the videos folder
    shutil.rmtree(video_folder, ignore_errors=True)
            
    os.makedirs(video_folder, exist_ok=True)

    start_time = time.time()
    episode = 0
    snr_list = []
    while time.time() - start_time < seconds:
        replay_buffer.increment_episode()
        episode += 1
        done = False
        truncated = False
        t = 0

        prev_state, _ = env.reset()
        fuel = 1000
        prev_state = np.concatenate((prev_state, [fuel / 1000.0, np.sin(prev_state[4]), np.cos(prev_state[4])]))

        do_training = train_every_n_episodes > 0 and episode % train_every_n_episodes == 0

        #########
        # Live testing loop start
        #########

        while not done and not truncated:
            t += 1
            action = actor_model.get_best_action(prev_state)
            next_state, done, fuel = _step_action(action, env, fuel)

            legs = next_state[6:8]
            if legs[0] == 1 and legs[1] == 1:
                done = True

            transition = create_observation(episode, t, prev_state, [action], next_state, done)
            replay_buffer.add(transition)
            replay_buffer0.append(transition)

            prev_state = next_state
        
        #########
        # Live testing loop end
        #########


        # save video with final value in the filename

        video_path = f"{video_folder}/{env._video_name}.mp4"
        env.reset()
        video_name_with_final_value = f"{video_folder}/episode_{episode+1}_t_{t}.mp4"
        shutil.move(video_path, video_name_with_final_value)

        # remove every *.json and rl-video-episode-*.mp4 file that's created alongside the video
        json_files = [f for f in os.listdir(video_folder) if f.endswith(".json")]
        for json_file in json_files:
            os.remove(os.path.join(video_folder, json_file))
        mp4_files = [f for f in os.listdir(video_folder) if f.startswith("rl-video-episode-") and f.endswith(".mp4")]
        for mp4_file in mp4_files:
            os.remove(os.path.join(video_folder, mp4_file))

        # if it's time to train the model, do so

        if do_training:
            replay_buffer0 = list(replay_buffer0)
            sample_len = len(replay_buffer0) * 4

            start_train_time = time.time()
            i = 0
            while time.time() - start_train_time < 10:
                training_sample = replay_buffer.sample(sample_len) + replay_buffer0
                world_model.train(training_sample)
                actor_model.train(training_sample)
                i += 1

            world_model.save()
            actor_model.save()

            try:
                replay_buffer.save()
            except Exception as e:
                print(f"Autosave failed: {e}")

        snr_by_dim_b = compute_validation_snr(world_model, replay_buffer0)
        short_names = ['x','y','vx','vy','a','va']
        per_dim = ','.join(f"{n}:{v:.1f}" for n, v in zip(short_names, snr_by_dim_b))
        print(f"SNR={np.mean(snr_by_dim_b):.1f} [{per_dim}]")


def main():
    parser = argparse.ArgumentParser(description="Live training for the LunarLander-v2 environment.")
    parser.add_argument("--seconds", type=int, default=600, help="Number of seconds to train")
    parser.add_argument('--train-every', type=int, default=4, help='Number of episodes between training sessions')
    seconds = parser.parse_args().seconds
    train_every = parser.parse_args().train_every

    np.set_printoptions(formatter={'float': lambda x: "{0:+0.4f}".format(x)})

    env = gym.make("LunarLander-v3", continuous=True, render_mode="rgb_array")
    env = RecordVideo(env, video_folder="./videos", episode_trigger=lambda x: True, disable_logger=True)

    train(env, seconds, train_every, video_folder="./videos")

    env.close()

if __name__ == "__main__":
    main()