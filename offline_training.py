import argparse
import time

import numpy as np
import torch

from ReplayBuffer import ReplayBuffer
from actor import ActorModel
from observation import calculate_reward, clip_state
from world import WorldModel

WORLD_DIM_LABELS = ['x', 'y', 'vx', 'vy', 'ang', 'vang', 'lLeg', 'rLeg', 'done', 'a0', 'a1', 'a2', 'a3']

def compute_world_value_snr(world_model, validation_sample):
    states_0 = np.array([obs.prev_state for obs in validation_sample], dtype=np.float32)
    states_1 = np.array([obs.next_state for obs in validation_sample], dtype=np.float32)
    actions = np.array([int(obs.action) for obs in validation_sample], dtype=np.int64)

    leg_changed = (states_0[:, 6:8] != states_1[:, 6:8]).any(axis=1)
    done_transition = states_1[:, 8] > 0.5
    mask = ~(leg_changed | done_transition)
    states_0 = states_0[mask]
    states_1 = states_1[mask]
    actions = actions[mask]

    predicted_next_full = world_model.predict_batch(states_0, actions)
    actual_reward = calculate_reward(torch.tensor(states_1, dtype=torch.float32)).numpy()
    predicted_reward = calculate_reward(torch.tensor(predicted_next_full, dtype=torch.float32)).numpy()

    mse = float(np.mean((predicted_reward - actual_reward) ** 2))
    signal = float(np.mean(actual_reward ** 2))
    snr = 0.0 if mse == 0 or signal == 0 else 10 * np.log10(signal / mse)
    return snr, mse, float(np.mean(predicted_reward)), float(np.mean(actual_reward))


def compute_world_per_dim_snr(world_model, validation_sample):
    states_0 = np.array([obs.prev_state for obs in validation_sample], dtype=np.float32)
    states_1 = np.array([obs.next_state for obs in validation_sample], dtype=np.float32)
    actions = np.array([int(obs.action) for obs in validation_sample], dtype=np.int64)

    predicted = world_model.predict_batch(states_0, actions)  # [N, 13]
    # Match training space: continuous head learns clipped delta, so compare in clipped space.
    target = clip_state(states_1)

    results = []
    for d in range(predicted.shape[1]):
        pred_d = predicted[:, d]
        tgt_d = target[:, d]
        mse = float(np.mean((pred_d - tgt_d) ** 2))
        signal = float(np.mean(tgt_d ** 2))
        snr = 0.0 if mse == 0 or signal == 0 else 10 * np.log10(signal / mse)
        results.append((WORLD_DIM_LABELS[d], snr))
    return results


def print_snr(world_model, training_sample, remain_time=None, actor_model=None):
    prefix = ''
    if remain_time is not None:
        prefix = f"Iter t={remain_time:.0f}s "
    v_snr, _, _, _ = compute_world_value_snr(world_model, training_sample)
    per_dim = compute_world_per_dim_snr(world_model, training_sample)
    per_dim_str = ' '.join(f"{label}={snr:.1f}" for label, snr in per_dim)
    sections = [f"World SNR={v_snr:.1f} {per_dim_str}"]
    if actor_model is not None:
        sections.append(actor_stats_str(actor_model, training_sample))
    print(prefix + ' | '.join(sections))


def actor_stats_str(actor_model, training_sample):
    actor_model.model.eval()
    with torch.no_grad():
        prediction, targets = actor_model.compute_prediction_and_targets(training_sample)

    pred = prediction[:, 0]
    tgt = targets[:, 0]
    return f"Actor-Q pred={pred.mean().item():+.3f} tgt={tgt.mean().item():+.3f}"


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
