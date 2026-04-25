import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import tempfile

from configuration import STATE_SIZE, POSSIBLE_ACTIONS, LAYER_COUNT, NODE_COUNT
from observation import calculate_reward

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
    def __init__(self, state_dim, action_dim, nodes, layers, output_dim):
        super().__init__()
        self.nodes = nodes
        self.layers = layers

        self.state_input = nn.Linear(state_dim, nodes)
        self.action_input = nn.Linear(action_dim, nodes)
        self.skip_layers = nn.ModuleList([SkipBlock(nodes) for _ in range(layers)])
        self.output = nn.Linear(nodes, output_dim)

    def forward(self, state, action):
        x = self.state_input(state)
        y = self.action_input(action)
        x = x + y
        for i in range(self.layers):
            x = self.skip_layers[i](x)
        x = self.output(x)
        return x


class ActorModel:
    def __init__(self, model_path="models/actor_model.pt"):
        self.model_path = model_path
        self.input_dim = STATE_SIZE
        self.possible_actions = POSSIBLE_ACTIONS
        self.nodes = NODE_COUNT
        self.layers = LAYER_COUNT
        self.discount_factor = 0.95

        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        self.model = ActorNet(self.input_dim, self.possible_actions, self.nodes, self.layers, 1)
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, map_location="cpu"))
            print(f"Loaded model weights from {self.model_path}")
        else:
            print(f"No checkpoint found at {self.model_path}; starting with random weights.")
        self.model.to(self.device)
        
        self.optimizer = optim.AdamW(self.model.parameters())
        self.criterion = nn.MSELoss()
        
    def train(self, observations):
        self.model.train()

        states_0 = torch.tensor(np.array([obs.prev_state for obs in observations]), dtype=torch.float32, device=self.device)
        states_1 = torch.tensor(np.array([obs.next_state for obs in observations]), dtype=torch.float32, device=self.device)
        actions = torch.tensor([int(obs.action) for obs in observations], dtype=torch.long, device=self.device)

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
        outputs = self.model(states_0, actions_onehot)
        loss = self.criterion(outputs, targets)
        loss.backward()
        self.optimizer.step()

    def save(self):
        fd1, tmp_path = tempfile.mkstemp(prefix='.tmp_actor_', suffix='.pt', dir=os.path.dirname(self.model_path) or '.')
        os.close(fd1)
        
        try:
            torch.save(self.model.state_dict(), tmp_path)
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
        
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        actions_1 = torch.arange(self.possible_actions, device=self.device).unsqueeze(-1)
        actions_1_onehot = F.one_hot(actions_1, num_classes=self.possible_actions).unsqueeze(0)
        actions_1_onehot = actions_1_onehot.view(-1, self.possible_actions)

        actions_1_onehot = actions_1_onehot.repeat(state_tensor.size(0), 1, 1).float()
        state_expanded = state_tensor.unsqueeze(1).repeat(1, self.possible_actions, 1)

        with torch.no_grad():
            predicted_rewards = self.model(state_expanded, actions_1_onehot)
        predicted_rewards = predicted_rewards.view(state_tensor.size(0), self.possible_actions)
        print(predicted_rewards.cpu().numpy())

        probs = F.softmax(predicted_rewards, dim=1)
        best_action_index = torch.multinomial(probs, num_samples=1).squeeze(1)
        best_action = actions_1[best_action_index]

        return best_action.item()