import numpy as np
import argparse

from ReplayBuffer import ReplayBuffer
from actor import ActorModel
from world import WorldModel

def compute_validation_snr(world_model, validation_sample):
    states_0 = np.array([obs.prev_state for obs in validation_sample], dtype=np.float32)
    states_1 = np.array([obs.next_state for obs in validation_sample], dtype=np.float32)
    actions = np.array([int(obs.action) for obs in validation_sample], dtype=np.int64)
    predicted_next = world_model.predict_batch(states_0, actions)
    snr_by_dim = []
    for dim in range(states_0.shape[1]):
        signal = np.mean(states_1[:, dim] ** 2)
        noise = np.mean((states_1[:, dim] - predicted_next[:, dim]) ** 2)
        snr = 0.0 if noise == 0 or signal == 0 else 10 * np.log10(signal / noise)
        snr_by_dim.append(snr)
    return np.array(snr_by_dim, dtype=np.float32)

def main():
    parser = argparse.ArgumentParser(description="Training for the LunarLander-v3 environment.")
    parser.add_argument("--seconds", type=int, default=60, help="Number of seconds to train")
    parser.add_argument('--sample-size', type=int, default=16, help='Number of samples for training')
    args = parser.parse_args()
    seconds = args.seconds
    sample_size = args.sample_size

    np.set_printoptions(formatter={'float': lambda x: "{0:+0.4f}".format(x)})

    replay_buffer = ReplayBuffer()
    world_model = WorldModel()
    actor_model = ActorModel(world_model)

    import time
    start_time = time.time()
    i = 0
    while time.time() - start_time < seconds:
        remaining_time = seconds - (time.time() - start_time)
        training_sample = replay_buffer.sample(2**sample_size)

        actor_model.train(training_sample)
        world_model.train(training_sample)

        snr_by_dim_b = compute_validation_snr(world_model, training_sample)
        short_names = ['x','y','vx','vy','a','va','ll','rl','fuel','done']
        per_dim = ','.join(f"{n}:{v:.1f}" for n, v in zip(short_names, snr_by_dim_b))
        print(f"Iter {i} t={remaining_time:.0f}s SNR={np.mean(snr_by_dim_b):.1f} [{per_dim}]")
        i += 1

        # save every 10 iterations
        if i % 10 == 0:
            actor_model.save()
            world_model.save()

if __name__ == "__main__":
    main()