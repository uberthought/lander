import os
import torch
import numpy as np
import shutil
import argparse
from collections import deque

from observation import create_observation, calculate_reward
from ReplayBuffer import ReplayBuffer
from actor import ActorModel
from world import WorldModel
from env import step_action, reset_state, make_env
from metrics import actor_stats_str, print_snr

import time

from configuration import VALIDATION_SAMPLE_SIZE, RECENT_BUFFER_SIZE, REPLAY_SAMPLE_MULTIPLIER


def train(env, seconds, train_every_n_episodes, video_folder):
    actor_model = ActorModel()
    world_model = WorldModel()

    # Main long-term buffer (persistent) and recent buffer for on-policy-ish updates
    replay_buffer = ReplayBuffer()
    replay_buffer0 = deque(maxlen=RECENT_BUFFER_SIZE)
    validation_sample = replay_buffer.sample(VALIDATION_SAMPLE_SIZE)

    start_time = time.time()
    episode = 0
    while time.time() - start_time < seconds:
        replay_buffer.increment_episode()
        episode += 1
        done = False
        truncated = False
        t = 0

        prev_state = reset_state(env)

        do_training = train_every_n_episodes > 0 and episode % train_every_n_episodes == 0

        #########
        # Live testing loop start
        #########

        while not done:
            t += 1
            action = actor_model.get_best_action(prev_state)
            next_state, done = step_action(action, env)

            transition = create_observation(prev_state, action, next_state)
            replay_buffer.add(transition)
            replay_buffer0.append(transition)

            prev_state = next_state
        
        #########
        # Live testing loop end
        #########


        # save video with final value in the filename

        video_path = f"{video_folder}/{env._video_name}.mp4"
        env.reset()
        final_value = calculate_reward(
            torch.tensor(prev_state, dtype=torch.float32).unsqueeze(0)
        ).item()
        v_int = int(round(final_value * 10000))
        video_name_with_final_value = f"{video_folder}/episode_{episode+1}_t_{t}_v_{v_int}.mp4"
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
            remain = seconds - (time.time() - start_time)
            print_snr(world_model, replay_buffer0, remain_time=remain, actor_model=actor_model)

            replay_buffer0 = list(replay_buffer0)
            sample_len = len(replay_buffer0) * REPLAY_SAMPLE_MULTIPLIER

            start_train_time = time.time()
            i = 0
            while time.time() - start_train_time < 10:
                training_sample = replay_buffer.sample(sample_len) + replay_buffer0
                actor_model.train(training_sample)
                world_model.train(training_sample)
                i += 1

            actor_model.save()
            world_model.save()

            try:
                replay_buffer.save()
            except Exception as e:
                print(f"Autosave failed: {e}")
            
            replay_buffer0 = deque(maxlen=RECENT_BUFFER_SIZE)



def main():
    parser = argparse.ArgumentParser(description="Live training for the LunarLander-v2 environment.")
    parser.add_argument("--seconds", type=int, default=3600, help="Number of seconds to train")
    parser.add_argument('--train-every', type=int, default=4, help='Number of episodes between training sessions')
    seconds = parser.parse_args().seconds
    train_every = parser.parse_args().train_every

    np.set_printoptions(formatter={'float': lambda x: "{0:+0.4f}".format(x)})

    shutil.rmtree("./videos", ignore_errors=True)

    env = make_env(record=True, video_folder="./videos")

    train(env, seconds, train_every, video_folder="./videos")

    env.close()

if __name__ == "__main__":
    main()