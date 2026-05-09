# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Deep Reinforcement Learning agent for the **LunarLander-v3** Gymnasium environment. Three training modes use two neural networks:
- **Actor Model** (`model/actor.py`) — predicts Q-values (expected rewards) for each of 16 possible action combinations
- **World Model** (`model/world.py`) — predicts next-state deltas given current state + action
- **World training** (`training/world_training.py`) — trains the actor via imaginary rollouts through the world model (model-based RL, no environment interaction)

## Common Commands

```bash
# Collect random experience to bootstrap the replay buffer (default --episodes 256)
python3 -m training.collect_random_data --episodes 1024

# Offline training (learn from replay buffer, no environment interaction)
python3 -m training.offline_training --seconds 60 --sample-size 16   # sample-size is exponent: 2^16 = 65536

# Live/on-policy training (interacts with environment, trains every N episodes)
python3 -m training.live_training --seconds 600 --train-every 4

# World-model imagination training (model-based, no environment interaction)
python3 -m training.world_training --seconds 60 --rollout-steps 10

# Run the full pipeline (collect → offline train → live train)
bash training_run.sh

# Run ReplayBuffer unit tests
python3 -m tests.test_ReplayBuffer

# Clean up models, videos, and replay buffer
bash clean.sh
```

## Architecture

### State & Action Space

- **State**: 9-dimensional — `[x, y, vx, vy, angle, vangle, left_leg, right_leg, done]`
- **Actions**: 2 simultaneous action slots, each 0–3 (off/right/left/reverse), giving **16 combinations** (`4^2`)
- Actions are **one-hot encoded** into a 16-dim vector for network input

### Neural Network Design (shared by both models)

```
Input (state + action) → Linear(in, 64) → 4× SkipBlock(64) → Linear(64, out)
SkipBlock: Linear→LeakyReLU→Linear→ReLU + residual
Optimizer: AdamW
```

- **Actor loss**: MSE (predicted Q-value vs. cumulative reward target)
- **World loss**: Huber (delta=1.0) with per-dimension weights `1/std(delta)`, normalized per batch

### Data Flow

```
training/collect_random_data.py → ReplayBuffer (replay_buffer.dat, memmap, ~369 MB, capacity 2^22)
                              ↓
  training/offline_training.py / training/live_training.py / training/world_training.py
                              ↓
                    checkpoints/actor_model.pt + checkpoints/world_model.pt
```

### Key Conventions

- **Device**: auto-selects MPS (Apple Silicon) → CPU fallback
- **Reward**: `torch.norm(sensors[:, [0, 1, 4]], dim=1) / sqrt(3)` — L2 norm of x, y, angle sensors (dims 0, 1, 4 only), clamped to [0,1]
- **Normalization factors** (in `shared/observation.py`): `[1, 1.75, 4, 4, π, 5, 1, 1, 1]` — last entry is done (already in [0,1])
- **Validation metric**: SNR (dB) = `10 * log10(signal / noise)` — higher is better world model accuracy
- **Observation namedtuple**: `(episode, time, prev_state, actions, next_state, done)`
- **Atomic writes**: both ReplayBuffer metadata and model checkpoints use temp-file + rename

### Configuration (`shared/configuration.py`)

| Constant | Value | Location | Meaning |
|----------|-------|----------|---------|
| `STATE_SIZE` | 13 | `shared/configuration.py` | Dimensions of state vector |
| `POSSIBLE_ACTIONS` | 4 | `shared/configuration.py` | Options per action slot |
| `LAYER_COUNT` | 4 | `shared/configuration.py` | Number of skip blocks |
| `NODE_COUNT` | 64 | `shared/configuration.py` | Width of hidden layers |
| `WORLD_LOSS_DELTA` | 1.0 | `shared/configuration.py` | Huber loss delta for world model |
| `discount_factor` | 0.95 | `model/actor.py` | Q-value discount rate |
| `lr` (AdamW) | 0.001 | `model/world.py` | World model learning rate |
| `replay_buffer0.maxlen` | 40000 | `training/live_training.py` | Recent on-policy experience window |

### Live Training On-Policy Mixing

`training/live_training.py` maintains a short-horizon deque (`replay_buffer0`, maxlen=40000) of recent experience alongside the main replay buffer. Each training call mixes `len(replay_buffer0) * 4` samples from the recent buffer with the same count from the main buffer, prioritizing fresh on-policy data.
