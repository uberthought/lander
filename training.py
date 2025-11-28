import numpy as np
import argparse

from ReplayBuffer import ReplayBuffer
from world import WorldModel

def main():
    parser = argparse.ArgumentParser(description="Training for the LunarLander-v3 environment.")
    parser.add_argument("--seconds", type=int, default=60, help="Number of seconds to train")
    parser.add_argument('--sample-size', type=int, default=14, help='Number of samples for training')
    seconds = parser.parse_args().seconds
    sample_size = parser.parse_args().sample_size

    np.set_printoptions(formatter={'float': lambda x: "{0:+0.4f}".format(x)})

    replay_buffer = ReplayBuffer()
    world_model = WorldModel()

    import time
    start_time = time.time()
    i = 0
    while time.time() - start_time < seconds:
        remaining_time = seconds - (time.time() - start_time)
        print(f"Training iteration {i} ... remaining time {remaining_time:.2f} seconds")
        training_sample = replay_buffer.sample2(2**sample_size)
        world_model.train(training_sample)
        i += 1

    world_model.save()

if __name__ == "__main__":
    main()