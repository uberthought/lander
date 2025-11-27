import numpy as np
import argparse

from ReplayBuffer import ReplayBuffer
from world import WorldModel

def main():
    parser = argparse.ArgumentParser(description="Training for the LunarLander-v3 environment.")
    parser.add_argument("--episodes", type=int, default=8, help="Number of training episodes")
    parser.add_argument('--sample-size', type=int, default=18, help='Number of samples for training')
    episodes = parser.parse_args().episodes
    sample_size = parser.parse_args().sample_size

    np.set_printoptions(formatter={'float': lambda x: "{0:+0.4f}".format(x)})

    replay_buffer = ReplayBuffer()
    world_model = WorldModel()

    for i in range(episodes):
        print(f"Training iteration {i} ...")
        training_sample = replay_buffer.sample(2**sample_size)
        world_model.train(training_sample)

    world_model.save()

if __name__ == "__main__":
    main()