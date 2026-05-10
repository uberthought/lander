# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Deep Reinforcement Learning agent for the **LunarLander-v3** Gymnasium environment (continuous action space).

- **`model/actor.py`** — `ActorModel`. Single scalar full-Q head per `(state, action)`. Trained against a potential-shaped self-bootstrap target (see below). Variance-normalized MSE loss.
- **`model/world.py`** — `WorldModel`. Predicts the next-state delta (full 13-dim, masked at training time so leg-changes/done rows are dropped). Used for diagnostics and dreamed-rollout training of the actor.

Training entrypoints, all sharing the same `ReplayBuffer`:
- `live_training` — collects on-policy episodes via the **actor** and trains **both** actor and world together; uses `print_snr` to report combined SNR.
- `actor_training` — same on-policy collection loop, but **only** trains the actor (world model untouched).
- `offline_training` — pure replay-buffer training of both networks (no env stepping).
- `world_training` — alternates real-env evaluation episodes with **dreamed rollouts** through the world model used as actor training data.

## Common Commands

```bash
# --- Bootstrap ---
python3 -m training.collect_random_data --episodes 1024     # seed replay_buffer.dat

# --- Training ---
python3 -m training.offline_training --seconds 60 --sample-size 16   # 2^16 transitions
python3 -m training.live_training    --seconds 600 --train-every 4
python3 -m training.actor_training   --seconds 600 --train-every 4
python3 -m training.world_training   --seconds 60 --rollout-steps 10

# --- Playback / inspection ---
python3 play_actor.py --model-path checkpoints/actor.pt --episodes 10  # records videos of greedy(-ish) rollouts

# --- Tests / cleanup ---
python3 -m tests.test_ReplayBuffer
bash clean.sh                                                # wipes checkpoints/, videos/, data/replay_buffer.dat*
```

## Architecture

### State & Action Space

- **State**: 13-dimensional vector laid out as `[x, y, vx, vy, angle, vangle, left_leg, right_leg, done, prev_a0, prev_a1, prev_a2, prev_a3]`. The first 6 are continuous sensors (`CONTINUOUS_STATE_DIM = 6`); legs and done are 0/1; the last 4 are a one-hot of the action that produced this state. The initial state after `env.reset()` is padded with `[0, 1, 0, 0, 0]` (legs=01, done=0, prev-action=null).
- **Actions**: flat 4-action discrete space (`POSSIBLE_ACTIONS = 4`). Mapping in `training/live_training._step_action` — `0=off`, `1=right`, `2=left`, `3=reverse` — encoded as a 2-D continuous LunarLander action vector.
- Actions are **one-hot encoded** into a 4-dim vector for network input.

### Neural Network Design

```
Input: state (13) + action (4) — projected separately and summed
       Linear(13, 64) + Linear(4, 64)  →  Σ
            ↓
       16× SkipBlock(64)         (LAYER_COUNT × NODE_COUNT)
            ↓
       Linear(64, out_dim)
```

`SkipBlock`: Linear → LeakyReLU → Linear → (+ residual) → LeakyReLU. Optimizer: AdamW.

Output dims by model:
- `model/actor.py` `ActorNet`: `1` — scalar full Q-value
- `model/world.py` `WorldNet`: `STATE_SIZE` (13) — full next-state delta. Loss is **masked** at training time to exclude transitions where a leg flag changes (`states_0[:, 6:8] != states_1[:, 6:8]`) or `done` fires on `s₁`; SNR diagnostics use the same mask.

### Reward (`shared/observation.calculate_reward`)

Not the LunarLander default reward. The reward used everywhere here is geometric and dense:

1. Clip the 6 continuous sensors to `CLIP_MIN/CLIP_MAX` (in-flight envelope).
2. Normalize by `[1, 1.75, 4, 4, π, 5]`, take `1 - |·|`, clamp to `[0, 1]`.
3. `position = ‖(x, y)‖₂ / √2`, `other = ‖(vx, vy, angle, vangle)‖₂ / √4`.
4. **Landing bonus**: when `|x| < 0.3`, engines off (`prev_a0 > 0.5`), and a leg flag is set, add `0.5` per leg to `other`.
5. Return `position * other`.

Done states (`shared/observation.is_done_state`): `|angle| > π/2`, `y < -0.5`, `y > 2.0`, or both legs touching down. Treated as done **in addition** to gym's `done`/`truncated`; the `done` flag in the 13-dim state vector is set from this combined condition. The training loops in `live_training.py` / `actor_training.py` exit on `done` alone — there is no separate `truncated` branch any more.

### Potential-shaped Q target (ActorModel)

ActorModel predicts full Q directly and bootstraps off its own next-state max:

```
target_full_Q = r(s₁) + [γ · r(s₁) − r(s₀)] + γ · max_a' Q(s₁, a')   # non-terminal
target_full_Q = r(s₁)                                                # terminal (done flag set)
```

with `discount_factor = 0.97`. The reward function stays explicit on the `s₁` side; the bootstrap absorbs the shaping potential `γ·r(s₁) − r(s₀)`. Loss is MSE on `(pred − target)²` divided by `target.var()` (variance-normalized).

### Storage

`shared/ReplayBuffer.py` — `data/replay_buffer.dat`, memmap, ~369 MB, capacity 2²², atomic metadata. Used by every training script.

### Live training on-policy mixing

`training/live_training.py` and `training/actor_training.py` maintain a short-horizon `deque(maxlen=40000)` of recent transitions alongside the persistent `ReplayBuffer`. Each training cycle (every `--train-every` episodes) mixes `len(replay_buffer0) * 4` random samples from the main buffer with **all** recent samples, biasing toward fresh on-policy data; the multiplier `4` is hardcoded in both files (no CLI flag). The inner training loop is wall-clock-budgeted to 10 seconds per cycle.

Per-cycle output is one line via `print_snr` / `print_actor_snr` (`Iter t=Xs World SNR=… | Actor-Q SNR=… pred=… tgt=…`); `actor_training.py` reuses `_actor_stats_str` from `offline_training.py` to keep that format identical across scripts.

### Configuration (`shared/configuration.py`)

| Constant | Value | Meaning |
|----------|-------|---------|
| `STATE_SIZE` | 13 | 6 sensors + 2 leg flags + done + 4 one-hot prev action |
| `CONTINUOUS_STATE_DIM` | 6 | First-N dims that go through clipping/normalization |
| `POSSIBLE_ACTIONS` | 4 | Flat discrete action count |
| `LAYER_COUNT` | 16 | Number of `SkipBlock`s in the trunk |
| `NODE_COUNT` | 64 | Hidden width |
| `WORLD_LOSS_DELTA` | 1.0 | Huber delta for dynamics regression |

Hyperparameters defined in model files (not configuration.py): `discount_factor = 0.97` (actor.py), `lr = 0.001` (world.py only — actor uses AdamW defaults).

### Conventions

- **Device**: auto-selects MPS (Apple Silicon) → CPU fallback. Models, tensors, and loaded checkpoints all `.to(self.device)`; loading from a checkpoint saved on a different device requires `map_location`.
- **Action sampling** in `get_best_action`: `softmax(softmax(full_q) * 100)` then `multinomial` — a sharpened temperature trick that's near-greedy but keeps exploration alive.
- **Validation metric**: SNR (dB) per dim = `10·log10(signal / noise)` over the masked validation sample — higher is better dynamics accuracy.
- **Observation namedtuple**: `Observation(prev_state, action, next_state)` — `action` is a single int 0–3, not a tuple.
- **Atomic writes**: every persistent artifact (replay buffer metadata, model `.pt`) writes to a temp file then `os.replace()`s into place. Never write directly to a final path.
- **Checkpoint format**: `{'model_state_dict': ..., 'optimizer_state_dict': ...}`. Loaders handle both this dict form and a bare state-dict for backward compat. When loading optimizer state, the model must be `.to(device)` *before* `optimizer.load_state_dict(...)` so optimizer state lands on the same device as the parameters.
