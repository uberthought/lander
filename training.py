import numpy as np
import argparse

from ReplayBuffer import ReplayBuffer
from actor import ActorModel
from world import WorldModel

def main():
    parser = argparse.ArgumentParser(description="Training for the LunarLander-v3 environment.")
    parser.add_argument("--seconds", type=int, default=60, help="Number of seconds to train")
    parser.add_argument('--sample-size', type=int, default=10, help='Number of samples for training')
    args = parser.parse_args()
    seconds = args.seconds
    sample_size = args.sample_size

    np.set_printoptions(formatter={'float': lambda x: "{0:+0.4f}".format(x)})

    replay_buffer = ReplayBuffer()
    actor_model = ActorModel()
    world_model = WorldModel()

    import time
    start_time = time.time()
    i = 0
    while time.time() - start_time < seconds:
        remaining_time = seconds - (time.time() - start_time)
        print(f"Training iteration {i} ... remaining time {remaining_time:.2f} seconds")
        training_sample = replay_buffer.sample(2**sample_size)
        actor_model.train(training_sample)
        world_metrics = world_model.train(training_sample)
        if world_metrics is not None:
            per_parameter_loss = ", ".join(
                f"{name}={value:.6f}" for name, value in world_metrics['per_parameter_loss'].items()
            )
            # print(f"World model error: {world_metrics['loss']:.6f}")
            # print(f"World model error by parameter: {per_parameter_loss}")
        i += 1

    actor_model.save()
    world_model.save()

if __name__ == "__main__":
    main()