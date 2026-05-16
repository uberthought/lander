import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import tempfile

from configuration import POSSIBLE_ACTIONS, LAYER_COUNT, NODE_COUNT, STATE_SIZE
from observation import calculate_reward, clip_state


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


class ActorNet(nn.Module):
    def __init__(self, action_dim, nodes, layers):
        super().__init__()
        self.layers = layers
        self.sensors_input = nn.Linear(STATE_SIZE, nodes)
        self.action_input = nn.Linear(action_dim, nodes)
        self.skip_layers = nn.ModuleList([SkipBlock(nodes) for _ in range(layers)])
        self.output = nn.Linear(nodes, 1)

    def forward(self, state, action):
        x = self.sensors_input(state) + self.action_input(action)
        for i in range(self.layers):
            x = self.skip_layers[i](x)
        return self.output(x)


class ActorModel:
    def __init__(self, model_path="checkpoints/actor_model.pt", load=True):
        self.model_path = model_path
        self.possible_actions = POSSIBLE_ACTIONS
        self.discount_factor = 0.97

        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        self.model = ActorNet(POSSIBLE_ACTIONS, NODE_COUNT, LAYER_COUNT)
        self.target_model = ActorNet(POSSIBLE_ACTIONS, NODE_COUNT, LAYER_COUNT)
        self.optimizer = optim.AdamW(self.model.parameters())

        self._load(load)

    def _load(self, load):
        if load and os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint and 'optimizer_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.to(self.device)
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print(f"Loaded actor model and optimizer state from {self.model_path}")
            else:
                self.model.load_state_dict(checkpoint)
                self.model.to(self.device)
                print(f"Loaded actor model weights from {self.model_path} (no optimizer state)")
        else:
            if load:
                print(f"No checkpoint found at {self.model_path}; starting actor with random weights.")
            else:
                print(f"Starting fresh actor with random weights (will save to {self.model_path}).")
            self.model.to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.to(self.device)

    def sync_target(self):
        self.target_model.load_state_dict(self.model.state_dict())
        print("[actor] target model synced")

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
            next_full_q = self.model(states_1_tile, actions_1_onehot).squeeze(-1)  # (B, A)
            future_rewards = next_full_q.max(dim=1)[0].unsqueeze(-1)               # (B, 1)

            prev_rewards = calculate_reward(states_0_full).view(-1, 1)
            current_rewards = calculate_reward(states_1_full).view(-1, 1)
            done_mask = states_1_full[:, 8:9] > 0.5

            shaping = self.discount_factor * current_rewards - prev_rewards
            nondone_full_q = current_rewards + shaping + self.discount_factor * future_rewards
            full_q_target = torch.where(done_mask, current_rewards, nondone_full_q)  # (B, 1)

        prediction = self.model(states_0, actions_onehot)  # (B, 1)
        return prediction, full_q_target

    def train(self, observations):
        self.model.train()

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
            next_full_q = self.target_model(states_1_tile, actions_1_onehot).squeeze(-1)
            future_rewards = next_full_q.max(dim=1)[0].unsqueeze(-1)
            prev_rewards = calculate_reward(states_0_full).view(-1, 1)
            current_rewards = calculate_reward(states_1_full).view(-1, 1)
            done_mask = states_1_full[:, 8:9] > 0.5
            shaping = self.discount_factor * current_rewards - prev_rewards
            nondone_full_q = current_rewards + shaping + self.discount_factor * future_rewards
            full_q_target = torch.where(done_mask, current_rewards, nondone_full_q)

        if not full_q_target.isfinite().all():
            print(f"[actor] non-finite target: mean={full_q_target.mean().item():.3g} max={full_q_target.abs().max().item():.3g}, skipping step")
            return

        self.optimizer.zero_grad()
        full_q_pred = self.model(states_0, actions_onehot)  # (B, 1)

        if not full_q_pred.isfinite().all():
            print(f"[actor] non-finite pred: mean={full_q_pred.mean().item():.3g} max={full_q_pred.abs().max().item():.3g}, skipping step")
            exit()

        sq = (full_q_pred - full_q_target) ** 2
        var = full_q_target.var(unbiased=False).clamp(min=1.0)
        loss = sq.mean() / var
        if not loss.isfinite():
            print(f"[actor] non-finite loss={loss.item():.3g} var={var.item():.3g}, skipping step")
            exit()
        loss.backward()
        # grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        # if grad_norm > 10.0:
        #     print(f"[actor] grad clipped: norm={grad_norm:.3g}")
        self.optimizer.step()

    def save(self):
        fd, tmp_path = tempfile.mkstemp(prefix='.tmp_actor_', suffix='.pt', dir=os.path.dirname(self.model_path) or '.')
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
            full_q = self.model(state_expanded, actions_1_onehot).squeeze(-1)  # (1, A)

        probs = F.softmax(full_q, dim=1)
        probs = F.softmax(probs * 100, dim=1)
        best_action_index = torch.multinomial(probs, num_samples=1).squeeze(1)
        return actions_1[best_action_index].item()
