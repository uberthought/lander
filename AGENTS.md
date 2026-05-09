# AGENTS.md

This file provides essential guidance for AI coding agents working in this codebase. It summarizes build/test commands, architecture, conventions, and common pitfalls to ensure agents are immediately productive.

---

## Build and Test Commands

- **Collect random data:**
  - `python3 -m training.collect_random_data --episodes 1024`
- **Offline training:**
  - `python3 -m training.offline_training --seconds 60`
- **Live training (with environment interaction):**
  - `python3 -m training.live_training --seconds 600`
- **Unit tests:**
  - See [tests/test_ReplayBuffer.py](tests/test_ReplayBuffer.py); tests are functions, not auto-discovered.

---

## Architecture Overview

- **ActorModel ([model/actor.py](model/actor.py))**: Predicts Q-values for all 2-action combinations using a skip-block neural net.
- **WorldModel ([model/world.py](model/world.py))**: Predicts next-state deltas given state and action.
- **ReplayBuffer ([shared/ReplayBuffer.py](shared/ReplayBuffer.py))**: Memory-mapped, atomic, circular buffer for experience replay.
- **Data Flow:**
  1. `training/collect_random_data.py` → ReplayBuffer
  2. `training/offline_training.py` → trains ActorModel & WorldModel
  3. `training/live_training.py` → on-policy data collection & training

---

## Project Conventions

- **State:** 9-dim vector (8-dim base + done)
- **Actions:** 2 simultaneous, 4 options each (16 combos, one-hot encoded)
- **Neural net:** 4 skip blocks, 64 nodes/layer, LeakyReLU, AdamW, MSELoss
- **Device:** Uses MPS if available, else CPU
- **Persistence:** Atomic file saves for buffer and metadata
- **Config:** All constants in [shared/configuration.py](shared/configuration.py)

---

## Common Pitfalls

- **Device mismatch:** Model loaded on CPU, moved to MPS can cause errors
- **Action encoding:** 2-action meshgrid → 16-dim one-hot; indexing errors possible
- **State shape:** Must be consistent everywhere
- **ReplayBuffer:** Atomic saves required; partial writes corrupt data
- **Reward calculation:** Assumes specific state layout
- **Video handling:** Path logic assumes Gymnasium wrapper behavior

---

## Key Files

- [shared/configuration.py](shared/configuration.py): Central config
- [shared/ReplayBuffer.py](shared/ReplayBuffer.py): Persistence, sampling
- [model/actor.py](model/actor.py), [model/world.py](model/world.py): Model patterns
- [shared/observation.py](shared/observation.py): Reward and observation tuple
- [training/live_training.py](training/live_training.py): Training loop, SNR, video

---

## Environment

- **Dependencies:** torch, gymnasium, numpy (no requirements.txt)
- **LunarLander-v3** (continuous mode)
- **Apple Silicon (MPS) supported**

---

> For more details, see the referenced files. Update this file if you add new conventions or major components.
