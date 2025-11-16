from critic import CriticModel
import numpy as np
import argparse

from actor import ActorModel
from replay_buffer import ReplayBuffer

def main():
    parser = argparse.ArgumentParser(description="Training for the LunarLander-v3 environment.")
    parser.add_argument("--episodes", type=int, default=8, help="Number of training episodes")
    parser.add_argument('--discount-factor', type=float, default=0.95, help='Discount factor for future rewards')
    parser.add_argument('--sample-size', type=int, default=18, help='Number of samples for training')
    episodes = parser.parse_args().episodes
    discount_factor = parser.parse_args().discount_factor
    sample_size = parser.parse_args().sample_size

    np.set_printoptions(formatter={'float': lambda x: "{0:+0.4f}".format(x)})

    replay_buffer = ReplayBuffer(state_shape=(10,))
    actor_model = ActorModel()
    critic_model = CriticModel(discount_factor=discount_factor)
    actor_model.set_critic_model(critic_model)
    critic_model.set_actor_model(actor_model)

    size = min(replay_buffer.size, 2 ** sample_size)
    if size > 0:
        for i in range(episodes):
            print(f"Training iteration {i} discount_factor={discount_factor}...")
            training_sample = replay_buffer.sample(size)
            # training_sample = replay_buffer
            critic_model.train(training_sample)
            actor_model.train(training_sample)

    critic_model.save()
    actor_model.save()

if __name__ == "__main__":
    main()