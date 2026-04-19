# AGENTS.md

This file provides essential guidance for AI coding agents working in this codebase. It summarizes build/test commands, architecture, conventions, and common pitfalls to ensure agents are immediately productive.

---

## Build and Test Commands

- **Collect random data:**
  - `python3 collect_random_data.py --episodes 1024`
- **Offline training:**
  - `python3 training.py --seconds 60`
- **Live training (with environment interaction):**
  - `python3 live_training.py --seconds 600`
- **Unit tests:**
  - See [test_ReplayBuffer.py](test_ReplayBuffer.py); tests are functions, not auto-discovered.

---

## Architecture Overview

- **ActorModel ([actor.py](actor.py))**: Predicts Q-values for all 2-action combinations using a skip-block neural net.
- **WorldModel ([world.py](world.py))**: Predicts next-state deltas given state and action.
- **ReplayBuffer ([ReplayBuffer.py](ReplayBuffer.py))**: Memory-mapped, atomic, circular buffer for experience replay.
- **Data Flow:**
  1. `collect_random_data.py` → ReplayBuffer
  2. `training.py` → trains ActorModel & WorldModel
  3. `live_training.py` → on-policy data collection & training

---

## Project Conventions

- **State:** 9-dim vector (8-dim base + normalized fuel)
- **Actions:** 2 simultaneous, 4 options each (16 combos, one-hot encoded)
- **Neural net:** 4 skip blocks, 64 nodes/layer, LeakyReLU, AdamW, MSELoss
- **Device:** Uses MPS if available, else CPU
- **Persistence:** Atomic file saves for buffer and metadata
- **Config:** All constants in [configuration.py](configuration.py)

---

## Common Pitfalls

- **Device mismatch:** Model loaded on CPU, moved to MPS can cause errors
- **Action encoding:** 2-action meshgrid → 16-dim one-hot; indexing errors possible
- **State shape:** Must be consistent everywhere (esp. fuel level)
- **ReplayBuffer:** Atomic saves required; partial writes corrupt data
- **Reward calculation:** Assumes specific state layout
- **Video handling:** Path logic assumes Gymnasium wrapper behavior

---

## Key Files

- [configuration.py](configuration.py): Central config
- [ReplayBuffer.py](ReplayBuffer.py): Persistence, sampling
- [actor.py](actor.py), [world.py](world.py): Model patterns
- [observation.py](observation.py): State normalization, reward
- [live_training.py](live_training.py): Training loop, SNR, video

---

## Environment

- **Dependencies:** torch, gymnasium, numpy (no requirements.txt)
- **LunarLander-v3** (continuous mode)
- **Apple Silicon (MPS) supported**

---

> For more details, see the referenced files. Update this file if you add new conventions or major components.
