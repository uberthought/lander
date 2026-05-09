## Understanding Training Output

During live training, you’ll see log lines like:

```
Episode 100 ended after 60 timesteps with median SNR=  3.9036 dB, std=  5.5520 dB
```

Here’s what each part means:

- **Episode 100 ended after 60 timesteps**: The agent completed episode 100, which lasted 60 steps (actions taken in the environment).
- **median SNR=  3.9036 dB**: The median Signal-to-Noise Ratio (SNR) for the episode. SNR measures how well the world model predicts actual state changes:
   - Higher SNR = model predictions are closer to reality (less noise/error).
   - Lower SNR = predictions are less accurate.
- **std=  5.5520 dB**: The standard deviation of SNR values during the episode, indicating how much the SNR varied from step to step.

In summary: Each line reports how long the episode lasted and how well (and consistently) the world model predicted state changes during that episode.
# LunarLander-v3 Deep RL Project

This repository implements a deep reinforcement learning (RL) agent for the LunarLander-v3 environment (continuous mode) using PyTorch. It features a custom replay buffer, actor and world models, and supports both offline and live training.

## Features
- **ReplayBuffer**: Fast, memory-mapped, atomic experience replay with persistence
- **ActorModel**: Neural network for Q-value prediction over all action combinations
- **WorldModel**: Neural network for next-state prediction
- **Training**: Supports both offline and live (on-policy) training
- **Apple Silicon (MPS) support**
- **Video recording**: Live training saves episode videos

## Project Structure
- `model/actor.py` — Actor model (Q-value predictor)
- `model/world.py` — World model (next-state predictor)
- `shared/ReplayBuffer.py` — Experience replay buffer
- `shared/observation.py` — Reward calculation and observation container
- `training/offline_training.py` — Offline training script
- `training/live_training.py` — Live training with environment interaction
- `training/world_training.py` — World-model imagination training
- `training/actor_training.py` — Actor training utilities
- `training/collect_random_data.py` — Collects random experience for buffer seeding
- `shared/configuration.py` — Central configuration constants
- `tests/test_ReplayBuffer.py` — Unit tests for the replay buffer
- `checkpoints/` — Saved model weights
- `videos/` — Saved episode videos

## Setup
1. **Dependencies**
   - Python 3.9+
   - `torch`, `gymnasium`, `numpy`
   - (No requirements.txt; install manually)

2. **Apple Silicon**
   - Uses MPS if available, otherwise CPU

## Usage
```
python3 -m training.collect_random_data --episodes 1024
python3 -m training.offline_training --seconds 60
python3 -m training.live_training --seconds 600 --train-every 4
python3 -m training.actor_training --seconds 60 --train-every 4
python3 -m training.world_training --seconds 60 --rollout-steps 10
python3 -m tests.test_ReplayBuffer
```

## Architecture
- **State**: 9-dimensional vector (8 base + done)
- **Actions**: 2 simultaneous, 4 options each (16 combos, one-hot encoded)
- **Neural Net**: 4 skip blocks, 64 nodes/layer, LeakyReLU, AdamW, MSELoss
- **Persistence**: Atomic file saves for buffer and metadata
- **Config**: All constants in `shared/configuration.py`

## Common Pitfalls
- Device mismatch (CPU/MPS)
- Action encoding/indexing errors
- State shape consistency
- ReplayBuffer atomicity
- Reward calculation assumptions
- Video path handling

## License
MIT (add a LICENSE file if needed)

---
For more details, see the code and [AGENTS.md](AGENTS.md).
