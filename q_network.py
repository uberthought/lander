import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import tempfile

from observation import calculate_reward

# PyTorch Actor Model
# input is the current state plus the action one-hot encoded
# output is the predicted reward for the action
class QNet(nn.Module):
    def __init__(self, input_dim, num_actions, nodes, layers):
        super().__init__()
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.nodes = nodes
        self.layers = layers
        
        self.input = nn.Linear(self.input_dim + self.num_actions, nodes)

        self.skip_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(nodes, nodes),
                nn.LeakyReLU(),
                nn.Linear(nodes, nodes),
                nn.LeakyReLU()
            ) for _ in range(layers)
        ])

        self.output = nn.Sequential(
            nn.Linear(nodes, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.input(x)
        for skip_layer in self.skip_layers:
            x = x + skip_layer(x)
        x = self.output(x)
        return x


class QNetwork:
    def __init__(self, discount_factor=0.95, model_path="models/actor_model.pt"):
        self.discount_factor = discount_factor
        self.model_path = model_path
        self.input_dim = 10
        self.num_actions = 4
        self.nodes = self.input_dim * self.num_actions * 4
        self.layers = 8

        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = self._load_model() or self._create_model()
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.MSELoss()

    def _create_model(self):
        return QNet(self.input_dim, self.num_actions, self.nodes, self.layers)

    def _load_model(self):
        if os.path.exists(self.model_path):
            model = QNet(self.input_dim, self.num_actions, self.nodes, self.layers)
            model.load_state_dict(torch.load(self.model_path, map_location="cpu"))
            return model
        return None


    def train(self, observations):
        self.model.train()

        states_0 = torch.tensor(np.array([obs.state for obs in observations]), dtype=torch.float32, device=self.device)
        states_1 = torch.tensor(np.array([obs.next_state for obs in observations]), dtype=torch.float32, device=self.device)
        actions_0 = torch.tensor([int(obs.action) for obs in observations], dtype=torch.long, device=self.device)
        dones_1 = torch.tensor([obs.next_state[-1] for obs in observations], dtype=torch.float32, device=self.device)

        actions_0_onehot = F.one_hot(actions_0, num_classes=self.num_actions).float()

        states_1_tile = states_1.unsqueeze(1).repeat(1, self.num_actions, 1)
        actions_1_onehot = F.one_hot(torch.arange(self.num_actions, device=self.device), num_classes=self.num_actions).unsqueeze(0).repeat(len(states_1), 1, 1)
        actor_inputs_1 = torch.cat([states_1_tile, actions_1_onehot], dim=-1).view(-1, self.input_dim + self.num_actions)

        with torch.no_grad():
            q_rewards_2 = self.model(actor_inputs_1)
        q_rewards_2 = torch.max(q_rewards_2.view(-1, self.num_actions), dim=1)[0].view(-1, 1)


        current_rewards = calculate_reward(states_1).view(-1, 1)
        future_rewards = self.discount_factor * q_rewards_2
        rewards = current_rewards + future_rewards

        # Set rewards to current rewards for terminal states
        done_indicies = (dones_1 == 1.0).nonzero(as_tuple=True)[0]
        rewards[done_indicies] = current_rewards[done_indicies]

        for _ in range(4):
            self.optimizer.zero_grad()
            model_input = torch.cat([states_0, actions_0_onehot], dim=-1)
            outputs = self.model(model_input)
            loss = self.criterion(outputs, rewards)
            loss.backward()
            self.optimizer.step()

    def save(self):
        fd, tmp_path = tempfile.mkstemp(prefix='.tmp_actor_', suffix='.pt', dir=os.path.dirname(self.model_path) or '.')
        os.close(fd)
        try:
            torch.save(self.model.state_dict(), tmp_path)
            os.replace(tmp_path, self.model_path)
        except KeyboardInterrupt:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    def get_optimal_action(self, obs):
        action_rewards = self.get_all_actions(obs)

        # greedy action selection
        action = torch.argmax(action_rewards).item()

        # stochastic action selection
        # action_softmax = F.softmax(action_rewards.view(-1), dim=0)
        # action = torch.multinomial(action_softmax, num_samples=1).item()

        return int(action), action_rewards.view(-1).cpu().numpy()

    def get_all_actions(self, obs):
        self.model.eval()
        sensors = torch.tensor([obs], dtype=torch.float32, device=self.device)
        sensors_tile = sensors.unsqueeze(1).repeat(1, self.num_actions, 1)
        actions_arange = torch.arange(self.num_actions, device=self.device)
        actions_onehot = F.one_hot(actions_arange, num_classes=self.num_actions)
        actions_tile = actions_onehot.unsqueeze(0).repeat(len(sensors), 1, 1)
        sensors_tile = sensors.repeat(self.num_actions, 1)
        actions_tile = F.one_hot(torch.arange(0, self.num_actions, device=self.device), num_classes=self.num_actions).float()
        actor_inputs = torch.cat([sensors_tile, actions_tile], dim=-1).view(-1, self.input_dim + self.num_actions)
        with torch.no_grad():
            action_rewards = self.model(actor_inputs)

        return action_rewards