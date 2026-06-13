"""Training metrics and live reporting shared by every training entrypoint.

World-model SNR (aggregate value SNR and per-dim SNR), the actor-Q snapshot
string, and the combined live-plot reporting line. Lives in its own module so
the training scripts depend on a library, not on each other.
"""
import numpy as np
import torch

from observation import calculate_reward, clip_state

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
    import live_plot

    prefix = ''
    if remain_time is not None:
        prefix = f"Iter t={remain_time:.0f}s "
    v_snr, _, _, _ = compute_world_value_snr(world_model, training_sample)
    per_dim = compute_world_per_dim_snr(world_model, training_sample)
    per_dim_str = ' '.join(f"{label}={snr:.1f}" for label, snr in per_dim)
    sections = [f"World SNR={v_snr:.1f} {per_dim_str}"]
    cont_loss, bool_loss, loss_ratio = world_model.pop_loss_stats()
    sections.append(f"WorldLoss cont={cont_loss:.4g} bool={bool_loss:.4g} ratio={loss_ratio:.3g}")
    actor_pred = None
    actor_tgt = None
    if actor_model is not None:
        text, actor_pred, actor_tgt = actor_stats_str(actor_model, training_sample)
        sections.append(text)
    print(prefix + ' | '.join(sections))

    live_plot.record(
        world_snr=v_snr,
        per_dim=per_dim,
        actor_pred=actor_pred if actor_pred is not None else float('nan'),
        actor_tgt=actor_tgt if actor_tgt is not None else float('nan'),
    )


def actor_stats_str(actor_model, training_sample):
    actor_model.model.eval()
    with torch.no_grad():
        prediction, targets = actor_model.compute_prediction_and_targets(training_sample)

    pred = prediction[:, 0]
    tgt = targets[:, 0]
    pred_mean = pred.mean().item()
    tgt_mean = tgt.mean().item()
    return f"Actor-Q pred={pred_mean:+.3f} tgt={tgt_mean:+.3f}", pred_mean, tgt_mean
