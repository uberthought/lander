import numpy as np
import argparse

from q_learning import QLearningModel
from ReplayBuffer import ReplayBuffer

def main():
    parser = argparse.ArgumentParser(description="Live training for the LunarLander-v3 environment.")
    parser.add_argument("--episodes", type=int, default=8, help="Number of training episodes")
    parser.add_argument('--discount-factor', type=float, default=0.95, help='Discount factor for future rewards')
    episodes = parser.parse_args().episodes
    discount_factor = parser.parse_args().discount_factor

    np.set_printoptions(formatter={'float': lambda x: "{0:+0.4f}".format(x)})

    replay_buffer = ReplayBuffer(state_shape=(10,))
    model = QLearningModel(discount_factor=discount_factor)

    if replay_buffer.size > 0:
        model.train([], replay_buffer, iterations=episodes)

    model.save()

if __name__ == "__main__":
    main()