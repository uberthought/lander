#!/usr/bin/env python3
import argparse
import time

import numpy as np

from ReplayBuffer import ReplayBuffer
from actor import ActorModel
from world import WorldModel
from metrics import print_snr


def main():
    parser = argparse.ArgumentParser(description="Training for the LunarLander-v3 environment.")
    parser.add_argument("--seconds", type=int, default=60, help="Number of seconds to train")
    parser.add_argument('--sample-size', type=int, default=16, help='Number of samples for training')
    parser.add_argument('--fixed-batch', action='store_true', help='Sample once and reuse the same batch every iteration (overfit test)')
    args = parser.parse_args()
    seconds = args.seconds
    sample_size = args.sample_size

    np.set_printoptions(formatter={'float': lambda x: "{0:+0.4f}".format(x)})

    replay_buffer = ReplayBuffer()
    world_model = WorldModel()
    actor_model = ActorModel()

    start_time = time.time()
    i = 0
    fixed_sample = replay_buffer.sample(2**sample_size) if args.fixed_batch else None
    while time.time() - start_time < seconds:
        remaining_time = seconds - (time.time() - start_time)
        training_sample = fixed_sample if fixed_sample is not None else replay_buffer.sample(2**sample_size)

        actor_model.train(training_sample)
        world_model.train(training_sample)
        
        i += 1

        # save every 10 iterations
        if i % 10 == 0:
            actor_model.save()
            world_model.save()

            print_snr(world_model, training_sample, remain_time=remaining_time, actor_model=actor_model)

if __name__ == "__main__":
    main()
