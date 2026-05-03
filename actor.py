import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import tempfile

from configuration import POSSIBLE_ACTIONS, LAYER_COUNT, NODE_COUNT
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
        x = input
        x = self.block1(x)
        x = self.relu1(x)
        x = self.block2(x)
        x = x + s
        x = self.relu2(x)
        return x

# PyTorch Actor Model
# input is the current state and a list of actions (one-hot encoded)
# output is the predicted reward for the given state and action
class ActorNet(nn.Module):
    def __init__(self, action_dim, nodes, layers):
        super().__init__()
        self.layers = layers
        self.sensors_input = nn.Linear(6, nodes)
        self.action_input = nn.Linear(action_dim, nodes)
        self.skip_layers = nn.ModuleList([SkipBlock(nodes) for _ in range(layers)])
        self.output = nn.Linear(nodes, 1)

    def forward(self, state, action):
        sensors_embed = self.sensors_input(state)
        action_embed = self.action_input(action)
        x = sensors_embed + action_embed
        for i in range(self.layers):
            x = self.skip_layers[i](x)
        x = self.output(x)
        return x


class ActorModel:
    def __init__(self, model_path="models/actor_model.pt"):
        self.model_path = model_path
        self.possible_actions = POSSIBLE_ACTIONS
        self.discount_factor = 0.97

        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        self.model = ActorNet(POSSIBLE_ACTIONS, NODE_COUNT, LAYER_COUNT)
        self.optimizer = optim.AdamW(self.model.parameters())
        self.criterion = nn.MSELoss()

        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location="mps" if torch.backends.mps.is_available() else "cpu")
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
            print(f"No checkpoint found at {self.model_path}; starting with random weights.")
            self.model.to(self.device)
        
    def train(self, observations):
        self.model.train()
        actions = torch.tensor([int(obs.action) for obs in observations], dtype=torch.long, device=self.device)
        states_0 = torch.tensor(np.array([obs.prev_state for obs in observations]), dtype=torch.float32, device=self.device)
        states_1 = torch.tensor(np.array([obs.next_state for obs in observations]), dtype=torch.float32, device=self.device)
        states_0 = clip_state(states_0)
        states_1 = clip_state(states_1)

        actions_onehot = F.one_hot(actions, num_classes=self.possible_actions).float()

        actions_1 = torch.arange(self.possible_actions, device=self.device).unsqueeze(-1)
        actions_1_onehot = F.one_hot(actions_1, num_classes=self.possible_actions).view(-1, self.possible_actions).float()
        actions_1_onehot = actions_1_onehot.unsqueeze(0).repeat(states_1.size(0), 1, 1)
        states_1_tile = states_1.unsqueeze(1).repeat(1, self.possible_actions, 1)

        with torch.no_grad():
            q_rewards = self.model(states_1_tile, actions_1_onehot)
            dones = states_1[:, -1]
            future_rewards = q_rewards.squeeze(-1).max(dim=1)[0].unsqueeze(-1) * (1.0 - dones.view(-1, 1))
            current_rewards = calculate_reward(states_1).view(-1, 1)
            targets = current_rewards + self.discount_factor * future_rewards

        self.optimizer.zero_grad()
        prediction = self.model(states_0, actions_onehot)
        loss = self.criterion(prediction, targets)
        loss.backward()
        self.optimizer.step()

    def save(self):
        fd1, tmp_path = tempfile.mkstemp(prefix='.tmp_actor_', suffix='.pt', dir=os.path.dirname(self.model_path) or '.')
        os.close(fd1)
        
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, tmp_path)
            os.replace(tmp_path, self.model_path)
            
        except KeyboardInterrupt:
            for tmp_path in [tmp_path, tmp_path]:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            raise
        except Exception:
            for tmp_path in [tmp_path, tmp_path]:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            raise

    def get_best_action(self, state):
        self.model.eval()
        state = clip_state(np.array(state, dtype=np.float32))
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        actions_1 = torch.arange(self.possible_actions, device=self.device).unsqueeze(-1)
        actions_1_onehot = F.one_hot(actions_1, num_classes=self.possible_actions).unsqueeze(0)
        actions_1_onehot = actions_1_onehot.view(-1, self.possible_actions)

        actions_1_onehot = actions_1_onehot.repeat(state_tensor.size(0), 1, 1).float()
        state_expanded = state_tensor.unsqueeze(1).repeat(1, self.possible_actions, 1)

        with torch.no_grad():
            predicted_rewards = self.model(state_expanded, actions_1_onehot)
        predicted_rewards = predicted_rewards.view(state_tensor.size(0), self.possible_actions)

        probs = F.softmax(predicted_rewards, dim=1)
        probs = probs * 10
        probs = F.softmax(probs, dim=1)
        best_action_index = torch.multinomial(probs, num_samples=1).squeeze(1)
        best_action = actions_1[best_action_index]

        return best_action.item()