import torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
import os
import shutil
import argparse
from collections import deque

from observation import Observation, calculate_value
from ReplayBuffer import ReplayBuffer
from world import WorldModel

# Define the normalization factors for the observation space
# These values are based on the observation space of the LunarLander-v2 environment
# and are used to scale the observations to a range of approximately [-1, 1]
# x, y, v_x, v_y, angle, v_angle
NORMALIZATION_FACTORS = np.array([1, 1.75, 4, 4, 3.1415927, 5, 1, 1], dtype=np.float32)

def train(env, episodes, train_every_n_episodes, video_folder):
    world_model = WorldModel()

    # Main long-term buffer (persistent) and recent buffer for on-policy-ish updates
    # State shape is 8 from LunarLander-v3 plus fuel level
    replay_buffer = ReplayBuffer(state_shape=(9,))
    replay_buffer0 = deque(maxlen=40000)

    # remove the videos folder
    shutil.rmtree(video_folder, ignore_errors=True)
            
    os.makedirs(video_folder, exist_ok=True)

    for episode in range(episodes):
        done = False
        truncated = False
        t = 0

        do_training = (episode + 1) > 0 and (episode + 1) % train_every_n_episodes == 0

        obs, _ = env.reset()
        obs = normalize_observation(obs)
        obs = np.concatenate((obs, [1.0]))  # fuel level

        fuel = 1000
    
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
            next_obs, _, done, truncated, _ = env.step(action0)

            # normalize the next observation
            next_obs = normalize_observation(next_obs)

            # if done, use the previous observation and add the legs and fuel level
            if done or truncated:
                next_obs[0] = obs[0]
                next_obs[1] = obs[1]
                next_obs[2] = obs[2]
                next_obs[3] = obs[3]
                next_obs[4] = obs[4]
                next_obs[5] = obs[5]

            # if the action is not to do nothing, decrease fuel
            if action != 0:
                fuel -= 1

            done = done or truncated

            # add the fuel level to the observation
            next_obs = np.concatenate((next_obs, [fuel / 1000.0]))

            transition = Observation(obs, action, next_obs, done)

            replay_buffer.add(transition)
            replay_buffer0.append(transition)

            # print the next world prediction from the world model

            world_prediction = world_model.predict(obs, action)
            world_prediction_delta = world_prediction - next_obs
            # world_prediction_rmse = np.sqrt(np.mean(world_prediction_delta ** 2))
            # print("world prediction delta: [" + ", ".join(f"{x:+0.4f}" for x in world_prediction_delta) + "]")
            # print(f"world prediction RMSE: {world_prediction_rmse:.4f}")

            signal = next_obs - obs
            world_prediction_snr = 10 * np.log10(np.mean(signal ** 2) / (np.mean(world_prediction_delta ** 2) + 1e-8))
            print(f"world prediction SNR: {world_prediction_snr:.4f} dB")

            obs = next_obs

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
            sample_len = len(replay_buffer0) * 32
            # sample_len = 2 ** 14
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
    parser.add_argument("--episodes", type=int, default=128, help="Number of training episodes")
    parser.add_argument('--train-every', type=int, default=4, help='Number of episodes between training sessions')
    parser.add_argument('--discount-factor', type=float, default=0.95, help='Discount factor for future rewards')
    episodes = parser.parse_args().episodes
    train_every = parser.parse_args().train_every
    discount_factor = parser.parse_args().discount_factor

    np.set_printoptions(formatter={'float': lambda x: "{0:+0.4f}".format(x)})

    # env = gym.make("LunarLander-v3", render_mode="rgb_array")
    env = gym.make("LunarLander-v3", continuous=True, render_mode="rgb_array")
    env = RecordVideo(env, video_folder="./videos", episode_trigger=lambda x: True, disable_logger=True)

    train(env, episodes, train_every, video_folder="./videos")

    env.close()

def normalize_observation(obs):
    obs = np.array(obs, dtype=np.float32)
    obs = obs / NORMALIZATION_FACTORS
    obs = np.clip(obs, -1.0, 1.0)
    return obs

if __name__ == "__main__":
    main()