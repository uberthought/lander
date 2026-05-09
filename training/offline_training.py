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
    predicted = world_model.predict_batch(states_0, actions)[:, :6]
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

    predicted_next_full = world_model.predict_batch(states_0, actions)
    actual_reward = calculate_reward(torch.tensor(states_1, dtype=torch.float32)).numpy()
    predicted_reward = calculate_reward(torch.tensor(predicted_next_full, dtype=torch.float32)).numpy()

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


def print_q_breakdown(q_model, training_sample):
    """Stage-by-stage breakdown of where Q-S and Q-V diverge.

    Walks the calculate_reward pipeline and reports per-stage SNR plus the
    fraction of samples that hit the clamps / clipping boundaries.
    """
    import torch
    import numpy as np
    from shared.configuration import STATE_SIZE, CONTINUOUS_STATE_DIM
    from shared.observation import clip_state, CLIP_MIN, CLIP_MAX

    NORMS = np.array([1, 1.75, 4, 4, 3.1415927, 5], dtype=np.float32)
    DIM_NAMES = ['x', 'y', 'vx', 'vy', 'a', 'va', 'leg1', 'leg2', 'done', 'a0', 'a1', 'a2', 'a3']

    def _snr(pred, tgt):
        pred_t = pred if isinstance(pred, torch.Tensor) else torch.tensor(pred)
        tgt_t = tgt if isinstance(tgt, torch.Tensor) else torch.tensor(tgt)
        mse = torch.mean((pred_t - tgt_t) ** 2).item()
        signal = torch.mean(tgt_t ** 2).item()
        return 0.0 if mse == 0 or signal == 0 else 10 * np.log10(signal / mse)

    q_model.model.eval()
    with torch.no_grad():
        prediction, _, _ = q_model._compute_prediction_and_targets(training_sample)
        states_0 = torch.tensor(
            np.array([obs.prev_state for obs in training_sample]), dtype=torch.float32, device=q_model.device
        )
        states_1 = torch.tensor(
            np.array([obs.next_state for obs in training_sample]), dtype=torch.float32, device=q_model.device
        )
        states_0_clip = clip_state(states_0)
        delta_pred = prediction[:, :STATE_SIZE]

        # Stage 1: predicted next state (full 13-dim) vs actual.
        next_pred = states_0_clip + delta_pred
        next_tgt = states_1

        next_pred_cpu = next_pred.cpu().numpy()
        next_tgt_cpu = next_tgt.cpu().numpy()

        print("=== Q-Model breakdown ===")
        print(f"N samples: {len(training_sample)}")

        # Per-dim raw state SNR.
        print("Stage 1: predicted next-state (raw)")
        for i, name in enumerate(DIM_NAMES):
            sig = float(np.mean(next_tgt_cpu[:, i] ** 2))
            mse = float(np.mean((next_pred_cpu[:, i] - next_tgt_cpu[:, i]) ** 2))
            snr = 0.0 if sig == 0 or mse == 0 else 10 * np.log10(sig / mse)
            print(f"  {name:>3}: signal={sig:8.4f} mse={mse:.2e} snr={snr:6.2f} dB"
                  f"  pred_mean={next_pred_cpu[:, i].mean():+.3f} tgt_mean={next_tgt_cpu[:, i].mean():+.3f}")
        agg_snr_raw = _snr(next_pred, next_tgt)
        print(f"  AGG: snr={agg_snr_raw:6.2f} dB")

        # Stage 2 onward operate on the 6 continuous dims (calculate_reward's pipeline).
        next_pred_c = next_pred[:, :CONTINUOUS_STATE_DIM]
        next_tgt_c = next_tgt[:, :CONTINUOUS_STATE_DIM]
        clip_min = torch.tensor(CLIP_MIN, dtype=next_pred_c.dtype, device=next_pred_c.device)
        clip_max = torch.tensor(CLIP_MAX, dtype=next_pred_c.dtype, device=next_pred_c.device)
        next_pred_clipped = torch.clamp(next_pred_c, clip_min, clip_max)
        next_tgt_clipped = torch.clamp(next_tgt_c, clip_min, clip_max)

        clipped_pred_frac = ((next_pred_c != next_pred_clipped).float().mean(dim=0)).cpu().numpy()
        clipped_tgt_frac = ((next_tgt_c != next_tgt_clipped).float().mean(dim=0)).cpu().numpy()
        print("Stage 2: after clip_state (continuous dims clamped to envelope)")
        for i in range(CONTINUOUS_STATE_DIM):
            print(f"  {DIM_NAMES[i]:>4}: pred_clipped_frac={clipped_pred_frac[i]:.3f} tgt_clipped_frac={clipped_tgt_frac[i]:.3f}")
        print(f"  AGG: snr={_snr(next_pred_clipped, next_tgt_clipped):6.2f} dB")

        # Stage 3: divide by NORMS.
        norms_t = torch.tensor(NORMS, device=next_pred.device)
        s_pred = next_pred_clipped / norms_t
        s_tgt = next_tgt_clipped / norms_t
        print("Stage 3: / NORMALIZATION_FACTORS")
        print(f"  AGG: snr={_snr(s_pred, s_tgt):6.2f} dB")

        # Stage 4: |.|, then clamp to [0,1].
        s_pred_abs = torch.clamp(s_pred.abs(), 0, 1)
        s_tgt_abs = torch.clamp(s_tgt.abs(), 0, 1)
        sign_flip_frac = ((s_pred.sign() != s_tgt.sign()).float().mean(dim=0)).cpu().numpy()
        sat_pred_frac = ((s_pred_abs >= 1.0).float().mean(dim=0)).cpu().numpy()
        sat_tgt_frac = ((s_tgt_abs >= 1.0).float().mean(dim=0)).cpu().numpy()
        print("Stage 4: |x|, clamp to [0,1]")
        for i in range(CONTINUOUS_STATE_DIM):
            print(f"  {DIM_NAMES[i]:>4}: sign_flip_frac={sign_flip_frac[i]:.3f}"
                  f"  sat_pred_frac={sat_pred_frac[i]:.3f} sat_tgt_frac={sat_tgt_frac[i]:.3f}")
        print(f"  AGG: snr={_snr(s_pred_abs, s_tgt_abs):6.2f} dB")

        # Stage 5: 1 - x.
        s_pred_inv = 1.0 - s_pred_abs
        s_tgt_inv = 1.0 - s_tgt_abs
        print(f"Stage 5: 1 - x  -> snr={_snr(s_pred_inv, s_tgt_inv):6.2f} dB")

        # Stage 6: position / other factors.
        pos_pred = torch.norm(s_pred_inv[:, [0, 1]], dim=1) / np.sqrt(2.0)
        pos_tgt = torch.norm(s_tgt_inv[:, [0, 1]], dim=1) / np.sqrt(2.0)
        oth_pred = torch.norm(s_pred_inv[:, [2, 3, 4, 5]], dim=1) / np.sqrt(4.0)
        oth_tgt = torch.norm(s_tgt_inv[:, [2, 3, 4, 5]], dim=1) / np.sqrt(4.0)
        print(f"Stage 6: position-norm  snr={_snr(pos_pred, pos_tgt):6.2f} dB"
              f"  pred_mean={pos_pred.mean().item():.3f} tgt_mean={pos_tgt.mean().item():.3f}")
        print(f"         other-norm     snr={_snr(oth_pred, oth_tgt):6.2f} dB"
              f"  pred_mean={oth_pred.mean().item():.3f} tgt_mean={oth_tgt.mean().item():.3f}")

        # Stage 7: position * other = base reward.
        r_pred = pos_pred * oth_pred
        r_tgt = pos_tgt * oth_tgt
        print(f"Stage 7: position * other (base reward)  snr={_snr(r_pred, r_tgt):6.2f} dB"
              f"  pred_mean={r_pred.mean().item():.3f} tgt_mean={r_tgt.mean().item():.3f}")

        # Stage 8: full calculate_reward (includes landing bonus on full 13-dim state).
        from shared.observation import calculate_reward
        r_full_pred = calculate_reward(next_pred)
        r_full_tgt = calculate_reward(next_tgt)
        print(f"Stage 8: full calculate_reward (with landing bonus)  snr={_snr(r_full_pred, r_full_tgt):6.2f} dB"
              f"  pred_mean={r_full_pred.mean().item():.3f} tgt_mean={r_full_tgt.mean().item():.3f}")

        # Compare Q-S vs Q-V end-to-end.
        print(f"--- End-to-end ---")
        print(f"Q-S (full next state SNR): {agg_snr_raw:.2f} dB")
        print(f"Q-V (reward SNR):          {_snr(r_full_pred, r_full_tgt):.2f} dB")
        print(f"Gap: {_snr(r_full_pred, r_full_tgt) - agg_snr_raw:+.2f} dB")


def print_q_action_breakdown(q_model, training_sample, sample_subset=256):
    """Per-action diagnostic: shows where full_Q's gap between actions actually lives.

    Reports for each action a in {0..POSSIBLE_ACTIONS-1}:
      - r(s + predicted_delta)  mean of the immediate-reward component of full_Q
      - q_rest mean             mean of the residual head
      - full_q mean             sum of the two
      - argmax %                fraction of batch rows where this action wins argmax full_q
      - sampled %               fraction of get_best_action() outcomes (over sample_subset rows)
      - shaping mean            γ·r(s₁) − r(s₀) restricted to rows whose buffer action == a
    """
    import torch
    import torch.nn.functional as F
    from shared.configuration import STATE_SIZE, POSSIBLE_ACTIONS
    from shared.observation import calculate_reward, clip_state

    q_model.model.eval()
    A = POSSIBLE_ACTIONS
    device = q_model.device

    states_0_full = torch.tensor(
        np.array([obs.prev_state for obs in training_sample]), dtype=torch.float32, device=device
    )
    states_1_full = torch.tensor(
        np.array([obs.next_state for obs in training_sample]), dtype=torch.float32, device=device
    )
    buf_actions = torch.tensor(
        [int(obs.action) for obs in training_sample], dtype=torch.long, device=device
    )
    states_0 = clip_state(states_0_full)

    actions_all = torch.arange(A, device=device)
    actions_all_onehot = F.one_hot(actions_all, num_classes=A).float()  # (A, A)
    actions_all_onehot = actions_all_onehot.unsqueeze(0).repeat(states_0.size(0), 1, 1)  # (B, A, A)
    states_0_tile = states_0.unsqueeze(1).repeat(1, A, 1)  # (B, A, 13)

    with torch.no_grad():
        heads = q_model.model(states_0_tile, actions_all_onehot)  # (B, A, 14)
        delta = heads[..., :STATE_SIZE]
        q_rest = heads[..., STATE_SIZE]                            # (B, A)
        predicted_next = states_0_tile + delta                     # (B, A, 13)
        flat = predicted_next.reshape(-1, STATE_SIZE)
        r_pred = calculate_reward(flat).reshape(predicted_next.shape[:-1])  # (B, A)
        full_q = r_pred + q_rest                                   # (B, A)

        prev_r = calculate_reward(states_0_full)
        curr_r = calculate_reward(states_1_full)
        shaping = q_model.discount_factor * curr_r - prev_r        # (B,)

        argmax_idx = full_q.argmax(dim=1)                          # (B,)
        argmax_pct = torch.zeros(A, device=device)
        for a in range(A):
            argmax_pct[a] = (argmax_idx == a).float().mean()

        # Sampling histogram via get_best_action — slow, so subsample.
        n_sub = min(sample_subset, states_0_full.size(0))
        sampled_counts = [0] * A
        states_0_full_cpu = states_0_full[:n_sub].cpu().numpy()
        for row in range(n_sub):
            a = q_model.get_best_action(states_0_full_cpu[row])
            sampled_counts[int(a)] += 1
        sampled_pct = [c / n_sub for c in sampled_counts]

        shaping_per_action = []
        for a in range(A):
            mask = buf_actions == a
            shaping_per_action.append(
                shaping[mask].mean().item() if mask.any() else float('nan')
            )

    names = ['noop', 'left', 'main', 'right'][:A]
    header = '                ' + '  '.join(f'a={a}({n})'.rjust(11) for a, n in enumerate(names))
    print(f"=== Q-action breakdown (N={len(training_sample)}, sampled subset={n_sub}) ===")
    print(header)

    def _row(label, values, fmt='{:+0.4f}'):
        cells = '  '.join(fmt.format(v).rjust(11) for v in values)
        print(f"{label:<16}{cells}")

    _row('r(s+Δ̂) mean',  r_pred.mean(dim=0).cpu().tolist())
    _row('q_rest mean',   q_rest.mean(dim=0).cpu().tolist())
    _row('full_q mean',   full_q.mean(dim=0).cpu().tolist())
    _row('argmax %',      [100 * v for v in argmax_pct.cpu().tolist()], fmt='{:0.2f}')
    _row('sampled %',     [100 * v for v in sampled_pct],               fmt='{:0.2f}')
    _row('shaping mean',  shaping_per_action)


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
    from shared.configuration import STATE_SIZE
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
        states_0_clip = clip_state(states_0)

        delta_pred = prediction[:, :STATE_SIZE]
        q_rest_pred = prediction[:, STATE_SIZE]
        next_state_pred = states_0_clip + delta_pred
        next_state_tgt = states_1
        v_pred = calculate_reward(next_state_pred)
        v_tgt = calculate_reward(states_1)

        q_rest_tgt = targets[:, STATE_SIZE]
        full_q_pred = v_pred + q_rest_pred
        full_q_tgt = v_tgt + q_rest_tgt

    def _snr(pred, tgt):
        mse = torch.mean((pred - tgt) ** 2).item()
        signal = torch.mean(tgt ** 2).item()
        return 0.0 if mse == 0 or signal == 0 else 10 * np.log10(signal / mse)

    s_snr = _snr(next_state_pred, next_state_tgt)
    # Q-S(v): SNR over only the dims calculate_reward actually reads
    # (x, y, vx, vy, angle, vangle = 0..5; left_leg, right_leg = 6,7; engine onehot = 9).
    reward_dims = [0, 1, 2, 3, 4, 5, 6, 7, 9]
    sv_snr = _snr(next_state_pred[:, reward_dims], next_state_tgt[:, reward_dims])
    v_snr = _snr(v_pred, v_tgt)
    q_snr = _snr(full_q_pred, full_q_tgt)
    return (
        f"Q-S SNR={s_snr:.1f} | "
        f"Q-S(v) SNR={sv_snr:.1f} | "
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