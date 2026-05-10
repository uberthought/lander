import numpy as np
import argparse

from ReplayBuffer import ReplayBuffer
from actor import ActorModel


def _actor_stats_str(actor_model, training_sample):
    import torch
    actor_model.model.eval()
    with torch.no_grad():
        prediction, targets = actor_model._compute_prediction_and_targets(training_sample)

    def _stats(label, pred, tgt, show_means):
        mse = torch.mean((pred - tgt) ** 2).item()
        signal = torch.mean(tgt ** 2).item()
        snr = 0.0 if mse == 0 or signal == 0 else 10 * np.log10(signal / mse)
        base = f"{label} SNR={snr:.1f}"
        if show_means:
            base += f" pred={pred.mean().item():+.3f} tgt={tgt.mean().item():+.3f}"
        return base

    return _stats("Actor-Q", prediction[:, 0], targets[:, 0], show_means=True)


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
    actor_model = ActorModel()

    import time
    start_time = time.time()
    i = 0
    fixed_sample = replay_buffer.sample(2**sample_size) if args.fixed_batch else None
    while time.time() - start_time < seconds:
        remaining_time = seconds - (time.time() - start_time)
        training_sample = fixed_sample if fixed_sample is not None else replay_buffer.sample(2**sample_size)

        actor_model.train(training_sample)

        i += 1

        # save every 10 iterations
        if i % 10 == 0:
            actor_model.save()
            print(f"Iter t={remaining_time:.0f}s " + _actor_stats_str(actor_model, training_sample))


if __name__ == "__main__":
    main()
