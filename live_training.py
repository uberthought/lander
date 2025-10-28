import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
import os
import shutil
import argparse
from collections import deque

import tensorflow as tf

from observation import Observation, calculate_value
from actor import ActorModel
from ReplayBuffer import ReplayBuffer

def train(env, model: ActorModel, episodes, train_every_n_episodes, video_folder):
    # Main long-term buffer (persistent) and recent buffer for on-policy-ish updates
    # State shape is 9: 8 from LunarLander-v2 + 1 for done flag
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
        obs = np.array(obs, dtype=np.float32)
        obs = np.append(obs, 0)  # not done
        prev_obs = obs

        #########
        # Live testing loop start
        #########

        while not done and not truncated:
            t += 1
            action, prediction = model.get_optimal_action(obs)

            # randomly explore 5% of the time
            if np.random.rand() < 0.05:
                action = np.random.randint(0, 4)

            next_obs, _, done, truncated, _ = env.step(action)

            if done:
                # use the sensors from the last observation and the legs from the current observation
                next_obs = np.append(obs[:6], next_obs[6:8])

            done = done or truncated

            next_obs = np.array(next_obs, dtype=np.float32)
            next_obs = np.append(next_obs, 1 if done else 0)
            transition = Observation(prev_obs, obs, action, next_obs)

            if prev_obs is not None:
                replay_buffer.add(transition)
                replay_buffer0.append(transition)

            main_thruster_emoji = "▼"
            right_thruster_emoji = "▶"
            left_thruster_emoji = "◀"
            no_op_emoji = " "

            rocket = no_op_emoji if action == 0 else right_thruster_emoji if action == 1 else main_thruster_emoji if action == 2 else left_thruster_emoji
            value1 = calculate_value(next_obs).numpy()[0]
            value0 = calculate_value(obs).numpy()[0]
            delta = value1 - value0
            print(f"Episode: {episode+1}/{episodes}, Step: {t}, Value: {value1:.4f}, Rocket: {rocket}, prediction: {prediction}, Delta: {delta:.4f}")

            prev_obs = obs
            obs = next_obs

        #########
        # Live testing loop end
        #########


        # save video with final value in the filename

        video_path = env.video_recorder.path
        value1 = calculate_value(next_obs).numpy()[0]
        video_name_with_final_value = f"{video_folder}/episode_{episode+1}_value_{value1:.4f}_t_{t}.mp4"
        shutil.move(video_path, video_name_with_final_value)

        # remove every *.json file that's created alongside the video
        json_files = [f for f in os.listdir(video_folder) if f.endswith(".json")]
        for json_file in json_files:
            os.remove(os.path.join(video_folder, json_file))

        # if it's time to train the model, do so

        if do_training:
            sample_len = len(replay_buffer0) * 16
            replay_buffer0 = list(replay_buffer0)
            for k in range(32):
                print(f"Training iteration {k} discount_factor={model.discount_factor}...")
                training_sample = replay_buffer.sample(sample_len) + replay_buffer0
                model.train(training_sample)

            model.save()

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

    env = gym.make("LunarLander-v2", render_mode="rgb_array")
    env = RecordVideo(env, video_folder="./videos", episode_trigger=lambda x: True, disable_logger=True)
    model = ActorModel(discount_factor=discount_factor)

    train(env, model, episodes, train_every, video_folder="./videos")

    env.close()

if __name__ == "__main__":
    main()