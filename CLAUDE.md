# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Deep Reinforcement Learning agent for the **LunarLander-v3** Gymnasium environment. Two neural networks are trained in tandem:
- **Actor Model** (`actor.py`) — predicts Q-values (expected rewards) for each of 16 possible action combinations
- **World Model** (`world.py`) — predicts next-state deltas given current state + action

## Common Commands

```bash
# Collect random experience to bootstrap the replay buffer
python3 collect_random_data.py --episodes 1024

# Offline training (learn from replay buffer, no environment interaction)
python3 training.py --seconds 60 --sample-size 16   # sample-size is exponent: 2^16 = 65536

# Live/on-policy training (interacts with environment, trains every N episodes)
python3 live_training.py --seconds 600 --train-every 4

# Run the full pipeline (collect → offline train → live train)
bash training_run.sh

# Run ReplayBuffer unit tests
python3 test_ReplayBuffer.py

# Clean up models, videos, and replay buffer
bash clean.sh
```

## Architecture

### State & Action Space

- **State**: 10-dimensional — `[x, y, vx, vy, angle, vangle, left_leg, right_leg, fuel/1000, done]`
- **Actions**: 2 simultaneous action slots, each 0–3 (off/right/left/reverse), giving **16 combinations** (`4^2`)
- Actions are **one-hot encoded** into a 16-dim vector for network input

### Neural Network Design (shared by both models)

```
Input (state + action) → Linear(in, 64) → 4× SkipBlock(64) → Linear(64, out)
SkipBlock: Linear→LeakyReLU→Linear→ReLU + residual
Optimizer: AdamW
```

- **Actor loss**: MSE (predicted Q-value vs. cumulative reward target)
- **World loss**: Huber (delta=1.0) with per-dimension weights `[1,1,1,2,1,4,1,1,1]` — extra weight on `vy` and `vangle`

### Data Flow

```
collect_random_data.py → ReplayBuffer (replay_buffer.dat, memmap, ~369 MB, capacity 2^22)
                              ↓
         training.py (offline) / live_training.py (on-policy)
                              ↓
                    models/actor_model.pt + models/world_model.pt
```

### Key Conventions

- **Device**: auto-selects MPS (Apple Silicon) → CPU fallback
- **Reward**: normalized mean of `1 - abs(sensor/norm_factor)` across first 6 dims + fuel; clamped to [0,1]
- **Normalization factors** (in `observation.py`): `[1, 1.75, 4, 4, π, 5, 1, 1, 1, 1]` — last entry is done (already in [0,1])
- **Validation metric**: SNR (dB) = `10 * log10(signal / noise)` — higher is better world model accuracy
- **Observation namedtuple**: `(episode, time, prev_state, actions, next_state, done)`
- **Atomic writes**: both ReplayBuffer metadata and model checkpoints use temp-file + rename

### Configuration (`configuration.py`)

| Constant | Value | Meaning |
|----------|-------|---------|
| `STATE_SIZE` | 10 | Dimensions of state vector |
| `POSSIBLE_ACTIONS` | 4 | Options per action slot |
| `LAYER_COUNT` | 4 | Number of skip blocks |
| `NODE_COUNT` | 64 | Width of hidden layers |
