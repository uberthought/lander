import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import tempfile

from observation import calculate_value



# PyTorch Critic Model
class CriticNet(nn.Module):
    def __init__(self, input_dim, num_actions, nodes, layers):
        super().__init__()
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.nodes = nodes
        self.layers = layers

        self.input_nodes = input_dim + num_actions

        self.input = nn.Linear(self.input_nodes, nodes)

        self.skip_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(nodes, nodes),
                nn.LeakyReLU(),
                nn.Linear(nodes, nodes),
                nn.LeakyReLU()
            ) for _ in range(layers)
        ])

        self.attention_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(nodes, nodes),
                nn.LeakyReLU(),
                nn.Linear(nodes, nodes),
                nn.Tanh()
            ) for _ in range(layers)
        ])

        self.output = nn.Sequential(
            nn.Linear(nodes, 1),
            nn.Sigmoid()
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = self.input(x)
        for i in range(self.layers):
            skip_layer = self.skip_layers[i]
            attention_layer = self.attention_layers[i]
            x = x + skip_layer(x) * attention_layer(x)
        x = self.output(x)
        return x


class CriticModel:
    def __init__(self, discount_factor=0.95, model_path="models/critic_model.pt"):
        self.model_path = model_path
        self.input_dim = 10
        self.num_actions = 4
        self.discount_factor = discount_factor
        self.nodes = self.input_dim * self.num_actions * 2
        self.layers = 32

        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = self._load_model() or self._create_model()
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.MSELoss()

    def set_actor_model(self, actor_model):
        self.actor_model = actor_model

    def _create_model(self):
        return CriticNet(self.input_dim, self.num_actions, self.nodes, self.layers)

    def _load_model(self):
        if os.path.exists(self.model_path):
            model = CriticNet(self.input_dim, self.num_actions, self.nodes, self.layers)
            model.load_state_dict(torch.load(self.model_path, map_location="cpu"))
            return model
        return None

    def train(self, observations):
        self.model.train()
        actions = torch.tensor([int(obs.action) for obs in observations], dtype=torch.long, device=self.device)
        dones = torch.tensor([obs.next_state[-1] for obs in observations], dtype=torch.float32, device=self.device)

        states_0 = torch.tensor(np.array([obs.state for obs in observations]), dtype=torch.float32, device=self.device)
        states_1 = torch.tensor(np.array([obs.next_state for obs in observations]), dtype=torch.float32, device=self.device)

        actions_hot = F.one_hot(actions, num_classes=self.num_actions).float()

        # calculate the values for the next states
        values_1_np = calculate_value(states_1.cpu().numpy())
        values_1 = torch.tensor(values_1_np, dtype=torch.float32, device=self.device).view(-1, 1)

        # get the predicted actions for the next states
        with torch.no_grad():
            p_actions_probs = self.actor_model.model(states_1)
            p_actions = torch.argmax(p_actions_probs, dim=1)
        p_actions_hot = F.one_hot(p_actions, num_classes=self.num_actions).float()

        # get the predicted values for the next state-action pairs
        with torch.no_grad():
            p_values_2 = self.model(states_1, p_actions_hot)
        p_values_2 = p_values_2.view(-1, 1)

        dones = dones.view(-1, 1)
        not_dones = 1 - dones

        current_values = (1 - self.discount_factor) * values_1
        future_values = self.discount_factor * p_values_2

        values = (current_values + future_values) * not_dones + values_1 * dones

        # For done indices, set values to values_1
        done_indices = (dones == 1.0).nonzero(as_tuple=True)[0]
        values[done_indices] = values_1[done_indices]

        # Train for 4 epochs
        for _ in range(4):
            self.optimizer.zero_grad()
            outputs = self.model(states_0, actions_hot)
            loss = self.criterion(outputs, values)
            loss.backward()
            self.optimizer.step()

    def save(self):
        fd, tmp_path = tempfile.mkstemp(prefix='.tmp_critic_', suffix='.pt', dir=os.path.dirname(self.model_path) or '.')
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

