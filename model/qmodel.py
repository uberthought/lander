import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import tempfile

from shared.configuration import POSSIBLE_ACTIONS, LAYER_COUNT, NODE_COUNT, STATE_SIZE, CONTINUOUS_STATE_DIM
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
        x = input
        x = self.block1(x)
        x = self.relu1(x)
        x = self.block2(x)
        x = x + s
        x = self.relu2(x)
        return x


# Outputs CONTINUOUS_STATE_DIM next-state delta + 1 Q-rest scalar.
class QNet(nn.Module):
    def __init__(self, action_dim, nodes, layers):
        super().__init__()
        self.layers = layers
        self.sensors_input = nn.Linear(STATE_SIZE, nodes)
        self.action_input = nn.Linear(action_dim, nodes)
        self.skip_layers = nn.ModuleList([SkipBlock(nodes) for _ in range(layers)])
        self.output = nn.Linear(nodes, STATE_SIZE + 1)

    def forward(self, state, action):
        x = self.sensors_input(state) + self.action_input(action)
        for i in range(self.layers):
            x = self.skip_layers[i](x)
        return self.output(x)


class QModel:
    def __init__(self, model_path="checkpoints/q_model.pt", load=True):
        self.model_path = model_path
        self.possible_actions = POSSIBLE_ACTIONS
        self.discount_factor = 0.97

        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        self.model = QNet(POSSIBLE_ACTIONS, NODE_COUNT, LAYER_COUNT)
        self.optimizer = optim.AdamW(self.model.parameters())

        if load and os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint and 'optimizer_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.to(self.device)
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print(f"Loaded model and optimizer state from {self.model_path}")
            else:
                self.model.load_state_dict(checkpoint)
                self.model.to(self.device)
                print(f"Loaded model weights from {self.model_path} (no optimizer state)")
        else:
            if load:
                print(f"No checkpoint found at {self.model_path}; starting with random weights.")
            else:
                print(f"Starting fresh QModel with random weights (will save to {self.model_path}).")
            self.model.to(self.device)

    def _full_q_from_heads(self, base_states_full, heads):
        # base_states_full: (..., 13); heads: (..., 14) — last dim is [delta(13), q_rest(1)]
        delta = heads[..., :STATE_SIZE]
        q_rest = heads[..., STATE_SIZE]
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
            next_heads = self.model(states_1_tile, actions_1_onehot)  # (B, A, 14)
            next_states_full = states_1.unsqueeze(1).expand(-1, self.possible_actions, -1)
            next_full_q = self._full_q_from_heads(next_states_full, next_heads)  # (B, A)
            future_rewards = next_full_q.max(dim=1)[0].unsqueeze(-1)

            current_rewards = calculate_reward(states_1_full).view(-1, 1)
            done_mask = states_1_full[:, 8:9] > 0.5

            # Bounded Bellman: target full_Q = (1−γ)·r(s₁) + γ·max_a' full_Q(s₁, a').
            # Since full_Q = r(s+Δ̂) + q_rest and r(s+Δ̂) ≈ r(s₁) when delta is accurate,
            # target q_rest = γ·(max_future − r(s₁)).
            nondone_q_rest = self.discount_factor * (future_rewards - current_rewards)
            q_rest_targets = torch.where(done_mask, torch.zeros_like(nondone_q_rest), nondone_q_rest)

            delta_targets = states_1 - states_0  # (B, 13)
            targets = torch.cat([delta_targets, q_rest_targets], dim=-1)  # (B, 14)

            leg_changed = (states_0_full[:, 6:8] != states_1_full[:, 6:8]).any(dim=1)
            done_row = states_1_full[:, 8] > 0.5
            delta_mask = ~(leg_changed | done_row)

        prediction = self.model(states_0, actions_onehot)
        return prediction, targets, delta_mask

    def train(self, observations):
        self.model.train()
        self.optimizer.zero_grad()
        prediction, targets, delta_mask = self._compute_prediction_and_targets(observations)

        sq_err = (prediction - targets) ** 2
        var = targets.var(dim=0, unbiased=False).clamp(min=1e-6)

        # Q-rest entry (last column): standard mean / variance.
        q_mse = sq_err[:, STATE_SIZE].mean()
        q_loss = q_mse / var[STATE_SIZE]

        # Delta entries (all 13 dims): masked mean / masked variance.
        mask_f = delta_mask.float().unsqueeze(-1)
        denom = mask_f.sum().clamp(min=1.0)
        delta_sq = sq_err[:, :STATE_SIZE] * mask_f
        delta_mse = delta_sq.sum(dim=0) / denom  # (13,)
        if delta_mask.any():
            masked_targets = targets[delta_mask][:, :STATE_SIZE]
            delta_var = masked_targets.var(dim=0, unbiased=False).clamp(min=1e-6)
        else:
            delta_var = var[:STATE_SIZE]
        delta_loss = (delta_mse / delta_var).sum()

        loss = q_loss + delta_loss
        loss.backward()
        self.optimizer.step()

    def save(self):
        fd, tmp_path = tempfile.mkstemp(prefix='.tmp_qmodel_', suffix='.pt', dir=os.path.dirname(self.model_path) or '.')
        os.close(fd)
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, tmp_path)
            os.replace(tmp_path, self.model_path)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    def get_best_action(self, state):
        self.model.eval()
        state = clip_state(np.array(state, dtype=np.float32))
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        actions_1 = torch.arange(self.possible_actions, device=self.device).unsqueeze(-1)
        actions_1_onehot = F.one_hot(actions_1, num_classes=self.possible_actions).view(-1, self.possible_actions).float()
        actions_1_onehot = actions_1_onehot.unsqueeze(0).repeat(state_tensor.size(0), 1, 1)
        state_expanded = state_tensor.unsqueeze(1).repeat(1, self.possible_actions, 1)

        with torch.no_grad():
            heads = self.model(state_expanded, actions_1_onehot)
            full_q = self._full_q_from_heads(state_expanded, heads)  # (1, A)

        probs = F.softmax(full_q, dim=1)
        probs = F.softmax(probs * 100, dim=1)
        best_action_index = torch.multinomial(probs, num_samples=1).squeeze(1)
        return actions_1[best_action_index].item()
