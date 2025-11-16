import torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
import os
import shutil
import argparse
from collections import deque

from observation import Observation, calculate_reward
from q_learning import QLearningModel
from ReplayBuffer import ReplayBuffer

# Define the normalization factors for the observation space
# These values are based on the observation space of the LunarLander-v3 environment
# and are used to scale the observations to a range of approximately [-1, 1]
# x, y, v_x, v_y, angle, v_angle
NORMALIZATION_FACTORS = np.array([1, 1.75, 4, 4, 3.1415927, 5, 1, 1], dtype=np.float32)

def train(env, qlearning_model: QLearningModel, episodes, train_every_n_episodes, video_folder):
    # Main long-term buffer (persistent) and recent buffer for on-policy-ish updates
    # State shape is 9: 8 from LunarLander-v3 + 1 for done flag
    replay_buffer = ReplayBuffer(state_shape=(10,))
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
        obs = np.concatenate((obs, [1.0, 0.0]))  # fuel level and not done

        fuel = 1000

        rl_video_files = [f for f in os.listdir(video_folder) if f.startswith("rl-video-episode-") and f.endswith(".mp4")]
        for rl_video_file in rl_video_files:
            os.remove(os.path.join(video_folder, rl_video_file))

    
        #########
        # Live testing loop start
        #########

        while not done and not truncated:
            t += 1
            action, prediction = qlearning_model.get_optimal_action(obs)

            next_obs, _, done, truncated, _ = env.step(action)

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

            # add the fuel level and done to the observation
            next_obs = np.concatenate((next_obs, [fuel / 1000.0, 1.0 if done else 0.0]))

            transition = Observation(obs, action, next_obs)

            replay_buffer.add(transition)
            replay_buffer0.append(transition)

            value1 = calculate_reward(torch.tensor(next_obs.reshape(1, -1), dtype=torch.float32, device=qlearning_model.device)).item()

            main_thruster_emoji = "▼"
            right_thruster_emoji = "▶"
            left_thruster_emoji = "◀"
            no_op_emoji = " "
            rocket = no_op_emoji if action == 0 else right_thruster_emoji if action == 1 else main_thruster_emoji if action == 2 else left_thruster_emoji
            print(f"Episode: {episode+1}/{episodes}, Step: {t}, Value: {value1:.4f}, Rocket: {rocket}, Prediction: {prediction}")

            obs = next_obs

        #########
        # Live testing loop end
        #########


        # save video with final value in the filename

        env.reset()

        video_path = f"{video_folder}/rl-video-episode-*.mp4"
        video_name = [f for f in os.listdir(video_folder) if f.startswith("rl-video-episode-") and f.endswith(".mp4")][0]
        video_path = os.path.join(video_folder, video_name)
        value1 = calculate_reward(torch.tensor(next_obs.reshape(1, -1), dtype=torch.float32, device=qlearning_model.device)).item()
        video_name_with_final_value = f"{video_folder}/episode_{episode+1}_value_{value1:.4f}_t_{t}.mp4"
        shutil.move(video_path, video_name_with_final_value)

        json_files = [f for f in os.listdir(video_folder) if f.endswith(".json")]
        for json_file in json_files:
            os.remove(os.path.join(video_folder, json_file))

        # if it's time to train the model, do so

        if do_training:
            new_observations = list(replay_buffer0)
            qlearning_model.train(new_observations, replay_buffer, iterations=24)

            qlearning_model.save()

            replay_buffer0 = deque(maxlen=40000)
            # Autosave observations
            try:
                replay_buffer.save()
                print(f"[autosave] Saved {len(replay_buffer)} observations -> {replay_buffer.filename}")
            except Exception as e:
                print(f"Autosave failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Live training for the LunarLander-v3 environment.")
    parser.add_argument("--episodes", type=int, default=128, help="Number of training episodes")
    parser.add_argument('--train-every', type=int, default=4, help='Number of episodes between training sessions')
    parser.add_argument('--discount-factor', type=float, default=0.95, help='Discount factor for future rewards')
    episodes = parser.parse_args().episodes
    train_every = parser.parse_args().train_every
    discount_factor = parser.parse_args().discount_factor

    np.set_printoptions(formatter={'float': lambda x: "{0:+0.4f}".format(x)})

    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    env = RecordVideo(env, video_folder="./videos", episode_trigger=lambda x: True, disable_logger=True)
    model = QLearningModel(discount_factor=discount_factor)

    train(env, model, episodes, train_every, video_folder="./videos")

    env.close()

def normalize_observation(obs):
    obs = np.array(obs, dtype=np.float32)
    obs = obs / NORMALIZATION_FACTORS
    obs = np.clip(obs, -1.0, 1.0)
    return obs

if __name__ == "__main__":
    main()