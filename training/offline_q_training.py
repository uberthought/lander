import time
import numpy as np
import argparse

from shared.ReplayBuffer import ReplayBuffer
from model.qmodel import QModel
from .offline_training import print_q_snr, print_q_breakdown, print_q_action_breakdown


def main():
    parser = argparse.ArgumentParser(description="Offline Q-training for the LunarLander-v3 environment.")
    parser.add_argument("--seconds", type=int, default=60, help="Number of seconds to train")
    parser.add_argument('--sample-size', type=int, default=16, help='Number of samples for training (2^N)')
    parser.add_argument('--fixed-batch', action='store_true', help='Sample once and reuse the same batch every iteration (overfit test)')
    parser.add_argument('--breakdown-every', type=int, default=0, help='Print stage-by-stage Q-S vs Q-V breakdown every N iterations (0 = never)')
    parser.add_argument('--action-breakdown-every', type=int, default=0, help='Print per-action Q breakdown every N iterations (0 = never)')
    args = parser.parse_args()

    np.set_printoptions(formatter={'float': lambda x: "{0:+0.4f}".format(x)})

    replay_buffer = ReplayBuffer()
    q_model = QModel()

    start_time = time.time()
    i = 0
    fixed_sample = replay_buffer.sample(2**args.sample_size) if args.fixed_batch else None
    while time.time() - start_time < args.seconds:
        remaining_time = args.seconds - (time.time() - start_time)
        training_sample = fixed_sample if fixed_sample is not None else replay_buffer.sample(2**args.sample_size)

        q_model.train(training_sample)
        i += 1

        if i % 10 == 0:
            q_model.save()
            print_q_snr(q_model, training_sample, remain_time=remaining_time)

        if args.breakdown_every > 0 and i % args.breakdown_every == 0:
            print_q_breakdown(q_model, training_sample)

        if args.action_breakdown_every > 0 and i % args.action_breakdown_every == 0:
            print_q_action_breakdown(q_model, training_sample)


if __name__ == "__main__":
    main()
