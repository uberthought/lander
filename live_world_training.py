import os
import torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
import shutil
import argparse
from collections import deque

from observation import create_observation, is_done_state, calculate_reward, clip_state
from ReplayBuffer import ReplayBuffer
from actor import ActorModel
from world import WorldModel
from world_training import run_imaginary_episode
from offline_training import actor_stats_str, print_snr

import time

from configuration import POSSIBLE_ACTIONS

CONTINUOUS_STATE_DIM = 6
VALIDATION_SAMPLE_SIZE = 2 ** 10

def _step_action(action, env):
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


def train(env, seconds, train_every_n_episodes, video_folder):
    actor_model = ActorModel(model_path="checkpoints/actor_model_world.pt", load=True)
    world_model = WorldModel()

    # Main long-term buffer (persistent) and recent buffer for on-policy-ish updates
    replay_buffer = ReplayBuffer()
    replay_buffer0 = deque(maxlen=40000)
    validation_sample = replay_buffer.sample(VALIDATION_SAMPLE_SIZE)

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

        #########
        # Live testing loop start
        #########

        while not done:
            t += 1
            action = actor_model.get_best_action(prev_state)
            next_state, done = _step_action(action, env)

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
            sample_len = len(replay_buffer0) * 4

            # Step 1: train world model on recent transitions + some random samples from the main buffer
            start_train_time = time.time()
            while time.time() - start_train_time < 5:
                training_sample = replay_buffer.sample(sample_len) + replay_buffer0
                world_model.train(training_sample)

            # Step 2: collect `episodes` imagined transitions, then train actor on them
            imagined_transitions = []
            for i in range(len(replay_buffer0)):
                seed_state = np.array(
                    # replay_buffer.sample(1)[0].next_state, dtype=np.float32
                    replay_buffer0[i].next_state, dtype=np.float32
                )
                transitions = run_imaginary_episode(
                    seed_state=seed_state,
                    actor_model=actor_model,
                    world_model=world_model,
                    max_steps=10,
                )
                imagined_transitions.extend(transitions)

            # Step 3: train actor model on recent real transitions + imagined transitions
            start_train_time = time.time()
            while time.time() - start_train_time < 5:
                actor_model.train(replay_buffer0 + imagined_transitions)

            actor_model.save()
            world_model.save()

            try:
                replay_buffer.save()
            except Exception as e:
                print(f"Autosave failed: {e}")
            
            replay_buffer0 = deque(maxlen=40000)



def main():
    parser = argparse.ArgumentParser(description="Live training for the LunarLander-v2 environment.")
    parser.add_argument("--seconds", type=int, default=3600, help="Number of seconds to train")
    parser.add_argument('--train-every', type=int, default=4, help='Number of episodes between training sessions')
    seconds = parser.parse_args().seconds
    train_every = parser.parse_args().train_every

    np.set_printoptions(formatter={'float': lambda x: "{0:+0.4f}".format(x)})

    shutil.rmtree("./videos", ignore_errors=True)

    env = gym.make("LunarLander-v3", continuous=True, render_mode="rgb_array")
    env = RecordVideo(env, video_folder="./videos", episode_trigger=lambda x: True, disable_logger=True)

    train(env, seconds, train_every, video_folder="./videos")

    env.close()

if __name__ == "__main__":
    main()