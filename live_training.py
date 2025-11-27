import torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
import os
import shutil
import argparse
from collections import deque

from observation import Observation, calculate_value, create_observation, normalize_state
from ReplayBuffer import ReplayBuffer
from world import WorldModel

import time

def train(env, seconds, train_every_n_episodes, video_folder):
    world_model = WorldModel()

    # Main long-term buffer (persistent) and recent buffer for on-policy-ish updates
    # State shape is 8 from LunarLander-v3 plus fuel level
    replay_buffer = ReplayBuffer()
    replay_buffer0 = deque(maxlen=40000)

    # remove the videos folder
    shutil.rmtree(video_folder, ignore_errors=True)
            
    os.makedirs(video_folder, exist_ok=True)

    start_time = time.time()
    episode = 0
    while time.time() - start_time < seconds:
        replay_buffer.increment_episode()
        episode += 1
        done = False
        truncated = False
        t = 0

        prev_state, _ = env.reset()
        prev_state = normalize_state(prev_state)
        fuel = 1000
        prev_state = np.concatenate((prev_state, [fuel / 1000.0]))
        prev_transition = Observation(episode, t, prev_state, 0, prev_state, 0.0)

        do_training = (episode + 1) > 0 and (episode + 1) % train_every_n_episodes == 0
        cumulative_snr = 0.0

        #########
        # Live testing loop start
        #########

        while not done and not truncated:
            t += 1
            action = np.random.randint(0, 4)

            # actual action for continuous LunarLander-v3 based on discrete action
            if action == 0:
                action0 = np.array([0.0, 0.0], dtype=np.float32)  # do nothing
            elif action == 1:
                action0 = np.array([0.0, 1.0], dtype=np.float32)  # right thruster
            elif action == 2:
                action0 = np.array([1.0, 0.0], dtype=np.float32)  # main thruster
            elif action == 3:
                action0 = np.array([0.0, -1.0], dtype=np.float32)  # left thruster
            next_state, _, done, truncated, _ = env.step(action0)

            # normalize the next observation
            next_state = normalize_state(next_state)

            # if the action is not to do nothing, decrease fuel
            if action != 0:
                fuel -= 1

            done = done or truncated
            next_state = np.concatenate((next_state, [fuel / 1000.0]))
            transition = create_observation(episode, t, prev_transition, action, next_state, done)
            replay_buffer.add(transition)
            prev_transition = transition

            replay_buffer.add(transition)
            replay_buffer0.append(transition)

            # print the next world prediction from the world model
            predicted_change = world_model.predict(prev_state, action)
            state_change = next_state - prev_state
            signal = np.mean(state_change ** 2)
            noise = np.mean((state_change - predicted_change) ** 2)
            if noise == 0 or signal == 0:
                snr = 0.0
            else:
                snr = 10 * np.log10(signal / noise)
            # print(f"SNR={snr: 8.4f} dB")

            if np.isfinite(snr):
                cumulative_snr += snr

            prev_state = next_state
        
        avg_snr = cumulative_snr / t
        print(f"Episode {episode} ended after {t} timesteps with average SNR={avg_snr: 8.4f} dB")

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
            sample_len = len(replay_buffer0) * 8
            replay_buffer0 = list(replay_buffer0)
            for k in range(32):
                training_sample = replay_buffer.sample(sample_len) + replay_buffer0
                world_model.train(training_sample)

            world_model.save()

            replay_buffer0 = deque(maxlen=40000)
            # Autosave observations
            try:
                replay_buffer.save()
                print(f"[autosave] Saved {len(replay_buffer)} observations -> {replay_buffer.filename}")
            except Exception as e:
                print(f"Autosave failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Live training for the LunarLander-v2 environment.")
    parser.add_argument("--seconds", type=int, default=600, help="Number of seconds to train")
    parser.add_argument('--train-every', type=int, default=4, help='Number of episodes between training sessions')
    parser.add_argument('--discount-factor', type=float, default=0.95, help='Discount factor for future rewards')
    seconds = parser.parse_args().seconds
    train_every = parser.parse_args().train_every
    discount_factor = parser.parse_args().discount_factor

    np.set_printoptions(formatter={'float': lambda x: "{0:+0.4f}".format(x)})

    # env = gym.make("LunarLander-v3", render_mode="rgb_array")
    env = gym.make("LunarLander-v3", continuous=True, render_mode="rgb_array")
    env = RecordVideo(env, video_folder="./videos", episode_trigger=lambda x: True, disable_logger=True)

    train(env, seconds, train_every, video_folder="./videos")

    env.close()

if __name__ == "__main__":
    main()