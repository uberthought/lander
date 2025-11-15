import torch
from critic import CriticModel
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
import os
import shutil
import argparse
from collections import deque

from observation import Observation, calculate_value
from actor import ActorModel
from ReplayBuffer import ReplayBuffer

# Define the normalization factors for the observation space
# These values are based on the observation space of the LunarLander-v2 environment
# and are used to scale the observations to a range of approximately [-1, 1]
# x, y, v_x, v_y, angle, v_angle
NORMALIZATION_FACTORS = np.array([1, 1.75, 4, 4, 3.1415927, 5, 1, 1], dtype=np.float32)

def train(env, actor_model: ActorModel, critic_model: CriticModel, episodes, train_every_n_episodes, video_folder):
    # Main long-term buffer (persistent) and recent buffer for on-policy-ish updates
    # State shape is 9: 8 from LunarLander-v2 + 1 for done flag
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
    
        #########
        # Live testing loop start
        #########

        while not done and not truncated:
            t += 1
            action, prediction = actor_model.get_optimal_action(obs)



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

            # add the fuel level and done to the observation
            next_obs = np.concatenate((next_obs, [fuel / 1000.0, 1.0 if done else 0.0]))

            transition = Observation(obs, action, next_obs)

            replay_buffer.add(transition)
            replay_buffer0.append(transition)

            value1 = calculate_value(torch.tensor(next_obs.reshape(1, -1), dtype=torch.float32, device=actor_model.device)).item()

            # main_thruster_emoji = "▼"
            # right_thruster_emoji = "▶"
            # left_thruster_emoji = "◀"
            # no_op_emoji = " "
            # rocket = no_op_emoji if action == 0 else right_thruster_emoji if action == 1 else main_thruster_emoji if action == 2 else left_thruster_emoji
            # print(f"Episode: {episode+1}/{episodes}, Step: {t}, Value: {value1:.4f}, Rocket: {rocket}, Prediction: {prediction}")

            actions = np.arange(critic_model.num_actions)
            actions_onehot = np.eye(critic_model.num_actions)[actions]
            sensors_tiled = np.tile(obs.reshape((1, -1)), (critic_model.num_actions, 1))
            sensors_tiled_tensor = torch.tensor(sensors_tiled, dtype=torch.float32, device=critic_model.device)
            actions_onehot_tensor = torch.tensor(actions_onehot, dtype=torch.float32, device=critic_model.device)
            critic_predictions = critic_model.q1_model(sensors_tiled_tensor, actions_onehot_tensor).cpu().detach().numpy().flatten()
            print(f"value: {value1:.4f}  critic: {critic_predictions}  actor: {prediction} next_obs: {next_obs[0:6]}")


            obs = next_obs

        #########
        # Live testing loop end
        #########


        # save video with final value in the filename

        video_path = f"{video_folder}/{env._video_name}.mp4"
        env.reset()
        value1 = calculate_value(torch.tensor(next_obs.reshape(1, -1), dtype=torch.float32, device=actor_model.device)).item()
        video_name_with_final_value = f"{video_folder}/episode_{episode+1}_value_{value1:.4f}_t_{t}.mp4"
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
                print(f"Training iteration {k} discount_factor={critic_model.discount_factor}...")
                training_sample = replay_buffer.sample(sample_len) + replay_buffer0
                critic_model.train(training_sample)
                actor_model.train(training_sample)

            critic_model.save()
            actor_model.save()

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
    actor_model = ActorModel()
    critic_model = CriticModel(discount_factor=discount_factor)
    actor_model.set_critic_model(critic_model)
    critic_model.set_actor_model(actor_model)

    train(env, actor_model, critic_model, episodes, train_every, video_folder="./videos")

    env.close()

def normalize_observation(obs):
    obs = np.array(obs, dtype=np.float32)
    obs = obs / NORMALIZATION_FACTORS
    obs = np.clip(obs, -1.0, 1.0)
    return obs

if __name__ == "__main__":
    main()