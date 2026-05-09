import numpy as np
import argparse

from shared.ReplayBuffer import ReplayBuffer
from model.actor import ActorModel
from shared.observation import calculate_reward
# from model.world import WorldModel

def compute_validation_snr(world_model, validation_sample):
    states_0 = np.array([obs.prev_state for obs in validation_sample], dtype=np.float32)
    states_1 = np.array([obs.next_state for obs in validation_sample], dtype=np.float32)
    actions = np.array([int(obs.action) for obs in validation_sample], dtype=np.int64)

    # Match the training filter: exclude leg-contact and done transitions
    leg_changed = (states_0[:, 6:8] != states_1[:, 6:8]).any(axis=1)
    done_transition = states_1[:, 8] > 0.5
    mask = ~(leg_changed | done_transition)
    states_0 = states_0[mask]
    states_1 = states_1[mask]
    actions = actions[mask]

    actual = states_1[:, :6]
    predicted = world_model.predict_batch(states_0, actions)
    snr_by_dim = []
    for dim in range(actual.shape[1]):
        signal = np.mean(actual[:, dim] ** 2)
        noise = np.mean((actual[:, dim] - predicted[:, dim]) ** 2)
        snr = 0.0 if noise == 0 or signal == 0 else 10 * np.log10(signal / noise)
        snr_by_dim.append(snr)
    return np.array(snr_by_dim, dtype=np.float32)

def compute_world_value_snr(world_model, validation_sample):
    import torch
    states_0 = np.array([obs.prev_state for obs in validation_sample], dtype=np.float32)
    states_1 = np.array([obs.next_state for obs in validation_sample], dtype=np.float32)
    actions = np.array([int(obs.action) for obs in validation_sample], dtype=np.int64)

    leg_changed = (states_0[:, 6:8] != states_1[:, 6:8]).any(axis=1)
    done_transition = states_1[:, 8] > 0.5
    mask = ~(leg_changed | done_transition)
    states_0 = states_0[mask]
    states_1 = states_1[mask]
    actions = actions[mask]

    predicted_next_6 = world_model.predict_batch(states_0, actions)
    actual_reward = calculate_reward(torch.tensor(states_1[:, :6], dtype=torch.float32)).numpy()
    predicted_reward = calculate_reward(torch.tensor(predicted_next_6, dtype=torch.float32)).numpy()

    mse = float(np.mean((predicted_reward - actual_reward) ** 2))
    signal = float(np.mean(actual_reward ** 2))
    snr = 0.0 if mse == 0 or signal == 0 else 10 * np.log10(signal / mse)
    return snr, mse, float(np.mean(predicted_reward)), float(np.mean(actual_reward))


def print_snr(world_model, training_sample, remain_time=None, actor_model=None):
    prefix = ''
    if remain_time is not None:
        prefix = f"Iter t={remain_time:.0f}s "
    v_snr, _, _, _ = compute_world_value_snr(world_model, training_sample)
    sections = [f"World SNR={v_snr:.1f}"]
    if actor_model is not None:
        sections.append(_actor_stats_str(actor_model, training_sample))
    print(prefix + ' | '.join(sections))


def print_q_snr(q_model, training_sample, remain_time=None):
    prefix = ''
    if remain_time is not None:
        prefix = f"Iter t={remain_time:.0f}s "
    print(prefix + _qmodel_stats_str(q_model, training_sample))


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

    r_section = _stats("Actor-R", prediction[:, 0], targets[:, 0], show_means=False)
    q_section = _stats("Actor-Q", prediction[:, 1], targets[:, 1], show_means=True)
    return f"{r_section} | {q_section}"


def _qmodel_stats_str(q_model, training_sample):
    import torch
    from shared.configuration import CONTINUOUS_STATE_DIM
    from shared.observation import calculate_reward, clip_state

    q_model.model.eval()
    with torch.no_grad():
        prediction, targets, _ = q_model._compute_prediction_and_targets(training_sample)

        states_1 = torch.tensor(
            np.array([obs.next_state for obs in training_sample]), dtype=torch.float32, device=q_model.device
        )
        states_0 = torch.tensor(
            np.array([obs.prev_state for obs in training_sample]), dtype=torch.float32, device=q_model.device
        )
        states_0_clip = clip_state(states_0)[:, :CONTINUOUS_STATE_DIM]

        delta_pred = prediction[:, :CONTINUOUS_STATE_DIM]
        q_rest_pred = prediction[:, CONTINUOUS_STATE_DIM]
        v_pred = calculate_reward(states_0_clip + delta_pred)
        v_tgt = calculate_reward(states_1)

        q_rest_tgt = targets[:, CONTINUOUS_STATE_DIM]
        full_q_pred = v_pred + q_rest_pred
        full_q_tgt = v_tgt + q_rest_tgt

    def _snr(pred, tgt):
        mse = torch.mean((pred - tgt) ** 2).item()
        signal = torch.mean(tgt ** 2).item()
        return 0.0 if mse == 0 or signal == 0 else 10 * np.log10(signal / mse)

    v_snr = _snr(v_pred, v_tgt)
    q_snr = _snr(full_q_pred, full_q_tgt)
    return (
        f"Q-V SNR={v_snr:.1f} | "
        f"Q-Q SNR={q_snr:.1f} pred={full_q_pred.mean().item():+.3f} tgt={full_q_tgt.mean().item():+.3f}"
    )


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

    import time
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

            print_snr(world_model, training_sample, remain_time=remaining_time)

if __name__ == "__main__":
    main()