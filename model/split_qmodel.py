import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import tempfile

from shared.configuration import POSSIBLE_ACTIONS, LAYER_COUNT, NODE_COUNT, STATE_SIZE, WORLD_LOSS_DELTA
from shared.observation import calculate_reward, clip_state


class SkipBlock(nn.Module):
    def __init__(self, nodes):
        super().__init__()
        self.block1 = nn.Linear(nodes, nodes)
        self.relu1 = nn.LeakyReLU()
        self.block2 = nn.Linear(nodes, nodes)
        self.relu2 = nn.LeakyReLU()

    def forward(self, input):
        s = input
        x = self.block1(input)
        x = self.relu1(x)
        x = self.block2(x)
        x = x + s
        return self.relu2(x)


class _Trunk(nn.Module):
    def __init__(self, action_dim, nodes, layers, out_dim):
        super().__init__()
        self.layers = layers
        self.sensors_input = nn.Linear(STATE_SIZE, nodes)
        self.action_input = nn.Linear(action_dim, nodes)
        self.skip_layers = nn.ModuleList([SkipBlock(nodes) for _ in range(layers)])
        self.output = nn.Linear(nodes, out_dim)

    def forward(self, state, action):
        x = self.sensors_input(state) + self.action_input(action)
        for i in range(self.layers):
            x = self.skip_layers[i](x)
        return self.output(x)


class DynamicsNet(_Trunk):
    def __init__(self, action_dim, nodes, layers):
        super().__init__(action_dim, nodes, layers, STATE_SIZE)


class QRestNet(_Trunk):
    def __init__(self, action_dim, nodes, layers):
        super().__init__(action_dim, nodes, layers, 1)


class _CombinedView(nn.Module):
    """Diagnostic shim: presents the two subnets as one module producing a
    14-dim [delta, q_rest] output, matching QNet's layout so the existing
    breakdown helpers in training/offline_training.py work unchanged."""
    def __init__(self, dynamics, qrest):
        super().__init__()
        self.dynamics = dynamics
        self.qrest = qrest

    def forward(self, state, action):
        delta = self.dynamics(state, action)
        q_rest = self.qrest(state, action)
        return torch.cat([delta, q_rest], dim=-1)


class SplitQModel:
    def __init__(self,
                 dyn_path="checkpoints/split_dynamics.pt",
                 q_path="checkpoints/split_qrest.pt",
                 load=True):
        self.dyn_path = dyn_path
        self.q_path = q_path
        self.possible_actions = POSSIBLE_ACTIONS
        self.discount_factor = 0.97

        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        self.dynamics = DynamicsNet(POSSIBLE_ACTIONS, NODE_COUNT, LAYER_COUNT)
        self.qrest = QRestNet(POSSIBLE_ACTIONS, NODE_COUNT, LAYER_COUNT)
        self.dyn_optimizer = optim.AdamW(self.dynamics.parameters())
        self.q_optimizer = optim.AdamW(self.qrest.parameters())
        self.huber = nn.HuberLoss(delta=WORLD_LOSS_DELTA, reduction='none')

        self._load_one(self.dynamics, self.dyn_optimizer, self.dyn_path, load, "dynamics")
        self._load_one(self.qrest, self.q_optimizer, self.q_path, load, "qrest")

        # Combined view for diagnostics that expect a single 14-dim model.
        self.model = _CombinedView(self.dynamics, self.qrest).to(self.device)

    def _load_one(self, net, optimizer, path, load, label):
        if load and os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint and 'optimizer_state_dict' in checkpoint:
                net.load_state_dict(checkpoint['model_state_dict'])
                net.to(self.device)
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print(f"Loaded {label} model and optimizer state from {path}")
            else:
                net.load_state_dict(checkpoint)
                net.to(self.device)
                print(f"Loaded {label} model weights from {path} (no optimizer state)")
        else:
            if load:
                print(f"No checkpoint found at {path}; starting {label} with random weights.")
            else:
                print(f"Starting fresh {label} with random weights (will save to {path}).")
            net.to(self.device)

    def _full_q(self, base_states_full, delta, q_rest):
        # base_states_full: (..., 13); delta: (..., 13); q_rest: (..., 1) or (...,)
        if q_rest.dim() == delta.dim():
            q_rest = q_rest.squeeze(-1)
        predicted_next = base_states_full + delta
        flat = predicted_next.reshape(-1, STATE_SIZE)
        rewards = calculate_reward(flat).reshape(predicted_next.shape[:-1])
        return rewards + q_rest

    def _compute_prediction_and_targets(self, observations):
        actions = torch.tensor([int(obs.action) for obs in observations], dtype=torch.long, device=self.device)
        states_0_full = torch.tensor(np.array([obs.prev_state for obs in observations]), dtype=torch.float32, device=self.device)
        states_1_full = torch.tensor(np.array([obs.next_state for obs in observations]), dtype=torch.float32, device=self.device)
        states_0 = clip_state(states_0_full)
        states_1 = clip_state(states_1_full)

        actions_onehot = F.one_hot(actions, num_classes=self.possible_actions).float()

        actions_1 = torch.arange(self.possible_actions, device=self.device).unsqueeze(-1)
        actions_1_onehot = F.one_hot(actions_1, num_classes=self.possible_actions).view(-1, self.possible_actions).float()
        actions_1_onehot = actions_1_onehot.unsqueeze(0).repeat(states_1.size(0), 1, 1)
        states_1_tile = states_1.unsqueeze(1).repeat(1, self.possible_actions, 1)

        with torch.no_grad():
            next_delta = self.dynamics(states_1_tile, actions_1_onehot)             # (B, A, 13)
            next_qrest = self.qrest(states_1_tile, actions_1_onehot).squeeze(-1)    # (B, A)
            next_full_q = self._full_q(states_1_tile, next_delta, next_qrest)       # (B, A)
            future_rewards = next_full_q.max(dim=1)[0].unsqueeze(-1)

            current_rewards = calculate_reward(states_1_full).view(-1, 1)
            done_mask = states_1_full[:, 8:9] > 0.5

            # Bounded Bellman: target full_Q = (1−γ)·r(s₁) + γ·max_a' full_Q(s₁, a').
            # Since full_Q = r(s+Δ̂) + q_rest and r(s+Δ̂) ≈ r(s₁) when delta is accurate,
            # target q_rest = γ·(max_future − r(s₁)).
            nondone_q_rest = self.discount_factor * (future_rewards - current_rewards)
            q_rest_targets = torch.where(done_mask, torch.zeros_like(nondone_q_rest), nondone_q_rest)

            delta_targets = states_1 - states_0  # (B, 13)

            leg_changed = (states_0_full[:, 6:8] != states_1_full[:, 6:8]).any(dim=1)
            done_row = states_1_full[:, 8] > 0.5
            delta_mask = ~(leg_changed | done_row)

        delta_pred = self.dynamics(states_0, actions_onehot)             # (B, 13)
        q_rest_pred = self.qrest(states_0, actions_onehot)               # (B, 1)

        prediction = torch.cat([delta_pred, q_rest_pred], dim=-1)        # (B, 14)
        targets = torch.cat([delta_targets, q_rest_targets], dim=-1)     # (B, 14)
        return prediction, targets, delta_mask

    def train(self, observations):
        self.dynamics.train()
        self.qrest.train()

        actions = torch.tensor([int(obs.action) for obs in observations], dtype=torch.long, device=self.device)
        states_0_full = torch.tensor(np.array([obs.prev_state for obs in observations]), dtype=torch.float32, device=self.device)
        states_1_full = torch.tensor(np.array([obs.next_state for obs in observations]), dtype=torch.float32, device=self.device)
        states_0 = clip_state(states_0_full)
        states_1 = clip_state(states_1_full)
        actions_onehot = F.one_hot(actions, num_classes=self.possible_actions).float()

        # ----- Dynamics net: Huber on masked rows -----
        leg_changed = (states_0_full[:, 6:8] != states_1_full[:, 6:8]).any(dim=1)
        done_row = states_1_full[:, 8] > 0.5
        mask = ~(leg_changed | done_row)

        if mask.any():
            delta_target = (states_1 - states_0)[mask]
            self.dyn_optimizer.zero_grad()
            delta_pred = self.dynamics(states_0[mask], actions_onehot[mask])
            dyn_loss = self.huber(delta_pred, delta_target).mean()
            dyn_loss.backward()
            self.dyn_optimizer.step()

        # ----- Q-rest net: bounded-Bellman target -----
        actions_1 = torch.arange(self.possible_actions, device=self.device).unsqueeze(-1)
        actions_1_onehot = F.one_hot(actions_1, num_classes=self.possible_actions).view(-1, self.possible_actions).float()
        actions_1_onehot = actions_1_onehot.unsqueeze(0).repeat(states_1.size(0), 1, 1)
        states_1_tile = states_1.unsqueeze(1).repeat(1, self.possible_actions, 1)

        with torch.no_grad():
            next_delta = self.dynamics(states_1_tile, actions_1_onehot)
            next_qrest = self.qrest(states_1_tile, actions_1_onehot).squeeze(-1)
            next_full_q = self._full_q(states_1_tile, next_delta, next_qrest)
            future_rewards = next_full_q.max(dim=1)[0].unsqueeze(-1)
            current_rewards = calculate_reward(states_1_full).view(-1, 1)
            done_mask = states_1_full[:, 8:9] > 0.5
            nondone_q_rest = self.discount_factor * (future_rewards - current_rewards)
            q_rest_target = torch.where(done_mask, torch.zeros_like(nondone_q_rest), nondone_q_rest)

        self.q_optimizer.zero_grad()
        q_rest_pred = self.qrest(states_0, actions_onehot)  # (B, 1)
        q_sq = (q_rest_pred - q_rest_target) ** 2
        q_var = q_rest_target.var(unbiased=False).clamp(min=1e-6)
        q_loss = q_sq.mean() / q_var
        q_loss.backward()
        self.q_optimizer.step()

    def save(self):
        self._save_one(self.dynamics, self.dyn_optimizer, self.dyn_path, '.tmp_split_dyn_')
        self._save_one(self.qrest, self.q_optimizer, self.q_path, '.tmp_split_q_')

    def _save_one(self, net, optimizer, path, prefix):
        fd, tmp_path = tempfile.mkstemp(prefix=prefix, suffix='.pt', dir=os.path.dirname(path) or '.')
        os.close(fd)
        try:
            torch.save({
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, tmp_path)
            os.replace(tmp_path, path)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    def get_best_action(self, state):
        self.dynamics.eval()
        self.qrest.eval()
        state = clip_state(np.array(state, dtype=np.float32))
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        actions_1 = torch.arange(self.possible_actions, device=self.device).unsqueeze(-1)
        actions_1_onehot = F.one_hot(actions_1, num_classes=self.possible_actions).view(-1, self.possible_actions).float()
        actions_1_onehot = actions_1_onehot.unsqueeze(0).repeat(state_tensor.size(0), 1, 1)
        state_expanded = state_tensor.unsqueeze(1).repeat(1, self.possible_actions, 1)

        with torch.no_grad():
            delta = self.dynamics(state_expanded, actions_1_onehot)
            q_rest = self.qrest(state_expanded, actions_1_onehot).squeeze(-1)
            full_q = self._full_q(state_expanded, delta, q_rest)  # (1, A)

        probs = F.softmax(full_q, dim=1)
        probs = F.softmax(probs * 100, dim=1)
        best_action_index = torch.multinomial(probs, num_samples=1).squeeze(1)
        return actions_1[best_action_index].item()
