import argparse
import csv
import numpy as np
import torch
import gymnasium as gym

from observation import calculate_reward, clip_state, is_done_state
from world import WorldModel
from actor import ActorModel
from configuration import POSSIBLE_ACTIONS
from world_training import _step_action, _clamp_state, _is_done


DIM_NAMES = ["x", "y", "vx", "vy", "angle", "vangle"]


def collect_real_episode(actor_model, env):
    states = []
    actions = []
    prev_state, _ = env.reset()
    prev_state = np.concatenate((prev_state, [0.0, 1.0, 0.0, 0.0, 0.0])).astype(np.float32)
    states.append(prev_state.copy())
    done = False
    while not done:
        action = actor_model.get_best_action(prev_state)
        prev_state, done = _step_action(action, env)
        prev_state = prev_state.astype(np.float32)
        actions.append(int(action))
        states.append(prev_state.copy())
    return states, actions


def rollout_world(seed_state, actor_model, world_model, horizon):
    states = [seed_state.copy()]
    cur = seed_state.copy()
    for t in range(1, horizon + 1):
        action = actor_model.get_best_action(cur)
        nxt = cur.copy()
        nxt[:6] = world_model.predict(cur, action)[:6]
        nxt = _clamp_state(nxt)
        done = _is_done(nxt, t, horizon)
        nxt[8] = float(done)
        nxt[9:13] = 0.0
        nxt[9 + int(action)] = 1.0
        states.append(nxt.copy())
        cur = nxt
        if done:
            break
    return states


def _reward_scalar(state):
    t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    return float(calculate_reward(t).item())


def evaluate(episodes, horizon, stride, output_path):
    actor_model = ActorModel(model_path="checkpoints/actor_model.pt", load=True)
    world_model = WorldModel()

    env = gym.make("LunarLander-v3", continuous=True)

    # Accumulators per step offset k=1..horizon
    cont_abs_sum = np.zeros((horizon, 6), dtype=np.float64)
    reward_abs_sum = np.zeros(horizon, dtype=np.float64)
    reward_abs_sq_sum = np.zeros(horizon, dtype=np.float64)
    counts = np.zeros(horizon, dtype=np.int64)

    for ep in range(episodes):
        real_states, _real_actions = collect_real_episode(actor_model, env)
        ep_len = len(real_states) - 1
        print(f"episode {ep+1}/{episodes}: real length = {ep_len}")

        for i in range(0, ep_len, stride):
            seed = real_states[i]
            max_k = min(horizon, ep_len - i)
            if max_k <= 0:
                continue
            imagined = rollout_world(seed, actor_model, world_model, max_k)
            # imagined[0] == seed; compare imagined[k] vs real_states[i+k]
            for k in range(1, len(imagined)):
                if k > max_k:
                    break
                imag = imagined[k]
                real = real_states[i + k]
                imag_clip = clip_state(imag.astype(np.float32))
                real_clip = clip_state(real.astype(np.float32))
                cont_abs_sum[k - 1] += np.abs(imag_clip[:6] - real_clip[:6])
                r_imag = _reward_scalar(imag)
                r_real = _reward_scalar(real)
                d = abs(r_imag - r_real)
                reward_abs_sum[k - 1] += d
                reward_abs_sq_sum[k - 1] += d * d
                counts[k - 1] += 1

    env.close()

    # Build aggregates
    valid = counts > 0
    cont_mean = np.zeros_like(cont_abs_sum)
    cont_mean[valid] = cont_abs_sum[valid] / counts[valid, None]
    reward_mean = np.zeros(horizon, dtype=np.float64)
    reward_mean[valid] = reward_abs_sum[valid] / counts[valid]
    reward_var = np.zeros(horizon, dtype=np.float64)
    reward_var[valid] = np.maximum(
        reward_abs_sq_sum[valid] / counts[valid] - reward_mean[valid] ** 2, 0.0
    )
    reward_std = np.sqrt(reward_var)

    # Print table
    header = (
        f"{'k':>4} {'n':>7} " + " ".join(f"{d+'_l1':>10}" for d in DIM_NAMES)
        + f" {'r_l1_mean':>12} {'r_l1_std':>10}"
    )
    print(header)
    print("-" * len(header))
    for k in range(horizon):
        if counts[k] == 0:
            continue
        row = (
            f"{k+1:>4} {counts[k]:>7} "
            + " ".join(f"{cont_mean[k, d]:>10.4f}" for d in range(6))
            + f" {reward_mean[k]:>12.4f} {reward_std[k]:>10.4f}"
        )
        print(row)

    # CSV
    with open(output_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["step", "n"]
            + [f"{d}_l1" for d in DIM_NAMES]
            + ["reward_l1_mean", "reward_l1_std"]
        )
        for k in range(horizon):
            if counts[k] == 0:
                continue
            w.writerow(
                [k + 1, int(counts[k])]
                + [f"{cont_mean[k, d]:.6f}" for d in range(6)]
                + [f"{reward_mean[k]:.6f}", f"{reward_std[k]:.6f}"]
            )
    print(f"\nWrote {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Measure world-model prediction drift vs real env over rollout horizon."
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--stride", type=int, default=1,
                        help="Seed a world-model rollout from every Nth real step.")
    parser.add_argument("--output", type=str, default="world_horizon_eval.csv")
    args = parser.parse_args()

    evaluate(args.episodes, args.horizon, args.stride, args.output)


if __name__ == "__main__":
    main()
