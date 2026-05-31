# LunarLander-v3 Deep RL Project

Deep reinforcement learning agent for the **LunarLander-v3** Gymnasium environment (continuous action space) in PyTorch.

For agent-oriented architecture notes, see [CLAUDE.md](CLAUDE.md).

## Architecture

- **`actor.py`** — `ActorModel`. Single scalar Q-value head per `(state, action)`. Trained against a target-network-bootstrapped, potential-shaped target: for non-terminal transitions `target = r(s₁) + γ·r(s₁) − r(s₀) + γ·max_a' Q_target(s₁,a')`; on terminal `target = r(s₁)`. Smooth L1 (Huber) loss; target network updated by Polyak averaging.
- **`world.py`** — `WorldModel`. Splits its 13-dim output: `[:6]` is a continuous next-state delta trained with Huber loss; `[6:13]` is logits for the boolean dims (legs, done, prev-action one-hot) trained with BCE-with-logits. Total = `Huber + BOOL_LOSS_WEIGHT * BCE`. Used by `world_training.py` / `live_world_training.py` to roll out imaginary episodes for actor training.

All Python lives flat at the project root. Run scripts from there.

## Setup

- Python 3.9+
- `torch`, `gymnasium`, `numpy` (no `requirements.txt` — install manually)
- Apple Silicon: uses MPS if available, otherwise CPU

## Usage

```bash
# Bootstrap the replay buffer
python3 collect_random_data.py --episodes 1024

# Offline training (samples from the replay buffer)
python3 offline_training.py --seconds 60 --sample-size 16   # 2^16 transitions

# Live training (collects on-policy transitions while training)
python3 actor_training.py   --seconds 600 --train-every 4

# World-model training (offline) and combined live actor + world-model training
python3 world_training.py      --seconds 600 --rollout-steps 1000 --eval-episodes 10 --sample-size 16
python3 live_world_training.py --seconds 3600 --train-every 4

# Evaluation: load a checkpoint and record episode videos
python3 play_actor.py --model-path checkpoints/actor_model_world.pt --episodes 10

# Tests / cleanup
python3 test_ReplayBuffer.py
bash clean.sh                                                # wipes checkpoints/, videos/, replay buffer
```

## Architecture Notes

- **State** (`STATE_SIZE = 13`): `[x, y, vx, vy, angle, vangle, left_leg, right_leg, done, prev_a0, prev_a1, prev_a2, prev_a3]`. The first 6 are continuous sensors; legs and done are 0/1; the last 4 are a one-hot of the previous action.
- **Actions** (`POSSIBLE_ACTIONS = 4`): flat discrete space — `0=off`, `1=right`, `2=left`, `3=reverse` — encoded as a 2-D continuous action vector inside `_step_action`.
- **Network**: state and action are projected to 64-dim and summed, passed through 16 `SkipBlock`s (Linear → LeakyReLU → Linear → +residual → LeakyReLU), then a final Linear to the head dim. Optimizer: AdamW.
- **Reward** (`observation.calculate_reward`): geometric and dense — clip continuous sensors, normalize, take `1 − |·|`, combine `position` and `other` norms, plus a landing bonus when `|x| < 0.3`, engines off, and a leg flag is set. Not the LunarLander default reward.
- **Storage**: `ReplayBuffer.py` — `data/replay_buffer.dat`, memmap, atomic metadata.
- **Atomic writes everywhere**: every persistent artifact writes to a temp path then `os.replace()`s into place.

## Configuration (`configuration.py`)

| Constant | Value | Meaning |
|----------|-------|---------|
| `STATE_SIZE` | 13 | 6 sensors + 2 leg flags + done + 4 one-hot prev action |
| `CONTINUOUS_STATE_DIM` | 6 | Dims that go through clipping/normalization |
| `POSSIBLE_ACTIONS` | 4 | Flat discrete action count |
| `LAYER_COUNT` | 16 | Number of `SkipBlock`s |
| `NODE_COUNT` | 64 | Hidden width |

`DISCOUNT_FACTOR = 0.97`, target-network `TAU = 0.05`, `ACTION_SHARPENING = 100`, `WORLD_LOSS_DELTA = 1.0` (Huber delta for `WorldModel`'s continuous head), and `BOOL_LOSS_WEIGHT = 1.0` (multiplier on the BCE term in `WorldModel`'s total loss) all live in `configuration.py`.

## Understanding Training Output

Each training cycle logs a line like:

```
Iter t=540s Actor-Q SNR=29.7 pred=+22.5 tgt=+22.6
```

- **Actor-Q SNR** — Signal-to-Noise Ratio of the actor's full-Q prediction vs. the potential-shaped target (`10·log10(signal / mse)`). Higher = more accurate Q-value estimate.
- **pred / tgt** — mean predicted Q vs. mean target Q across the batch.

## License

MIT (add a LICENSE file if needed).
