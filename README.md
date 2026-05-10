# LunarLander-v3 Deep RL Project

Deep reinforcement learning agent for the **LunarLander-v3** Gymnasium environment (continuous action space) in PyTorch.

For agent-oriented architecture notes, see [CLAUDE.md](CLAUDE.md).

## Architecture

- **`model/actor.py`** — `ActorModel`. Single scalar Q-value head per `(state, action)`. Trained against a potential-shaped self-bootstrap target: for non-terminal transitions `target = r(s₁) + γ·r(s₁) − r(s₀) + γ·max_a' Q(s₁,a')`; on terminal `target = r(s₁)`. Variance-normalized MSE loss.
- **`model/world.py`** — `WorldModel`. Predicts the continuous next-state delta. Used for diagnostics and dreamed-rollout training.

## Setup

- Python 3.9+
- `torch`, `gymnasium`, `numpy` (no `requirements.txt` — install manually)
- Apple Silicon: uses MPS if available, otherwise CPU

## Usage

```bash
# Bootstrap the replay buffer
python3 -m training.collect_random_data --episodes 1024

# Offline training (samples from the replay buffer)
python3 -m training.offline_training --seconds 60 --sample-size 16   # 2^16 transitions

# Live training (collects on-policy transitions while training)
python3 -m training.live_training    --seconds 600 --train-every 4
python3 -m training.actor_training   --seconds 600 --train-every 4

# World-model rollouts (dreamed-trajectory training of the actor)
python3 -m training.world_training   --seconds 60 --rollout-steps 10

# Tests / cleanup
python3 -m tests.test_ReplayBuffer
bash clean.sh                                                        # wipes checkpoints/, videos/, replay buffer
```

## Architecture Notes

- **State** (`STATE_SIZE = 13`): `[x, y, vx, vy, angle, vangle, left_leg, right_leg, done, prev_a0, prev_a1, prev_a2, prev_a3]`. The first 6 are continuous sensors; legs and done are 0/1; the last 4 are a one-hot of the previous action.
- **Actions** (`POSSIBLE_ACTIONS = 4`): flat discrete space — `0=off`, `1=right`, `2=left`, `3=reverse` — encoded as a 2-D continuous action vector inside `_step_action`.
- **Network**: state and action are projected to 64-dim and summed, passed through 16 `SkipBlock`s (Linear → LeakyReLU → Linear → +residual → LeakyReLU), then a final Linear to the head dim. Optimizer: AdamW.
- **Reward** (`shared/observation.calculate_reward`): geometric and dense — clip continuous sensors, normalize, take `1 − |·|`, combine `position` and `other` norms, plus a landing bonus when `|x| < 0.3`, engines off, and a leg flag is set. Not the LunarLander default reward.
- **Storage**: `shared/ReplayBuffer.py` — `data/replay_buffer.dat`, memmap, atomic metadata.
- **Atomic writes everywhere**: every persistent artifact writes to a temp path then `os.replace()`s into place.

## Configuration (`shared/configuration.py`)

| Constant | Value | Meaning |
|----------|-------|---------|
| `STATE_SIZE` | 13 | 6 sensors + 2 leg flags + done + 4 one-hot prev action |
| `CONTINUOUS_STATE_DIM` | 6 | Dims that go through clipping/normalization |
| `POSSIBLE_ACTIONS` | 4 | Flat discrete action count |
| `LAYER_COUNT` | 16 | Number of `SkipBlock`s |
| `NODE_COUNT` | 64 | Hidden width |
| `WORLD_LOSS_DELTA` | 1.0 | Huber delta for dynamics regression |

`discount_factor = 0.97` is defined in the model files, not in `configuration.py`.

## Understanding Training Output

Live-training logs lines like:

```
Episode 100 ended after 60 timesteps with median SNR=  3.9036 dB, std=  5.5520 dB
```

- **median SNR** — Signal-to-Noise Ratio of the world / dynamics model's predictions vs. actual next states (`10·log10(signal / noise)`, per dim then aggregated). Higher = more accurate.

## License

MIT (add a LICENSE file if needed).
