import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import tempfile

from observation import calculate_value


# PyTorch Critic Model
# input is the current state plus the action one-hot encoded
# output is the predicted reward for the action
class CriticNet(nn.Module):
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
            x = x + skip_layer(x)
        x = self.output(x)
        return x


class CriticModel:
    def __init__(self, discount_factor=0.95, model_path="models/critic_model.pt"):
        self.model_path = model_path
        self.input_dim = 9
        self.num_actions = 4
        self.discount_factor = discount_factor
        self.nodes = self.input_dim * self.num_actions * 4
        self.layers = 8

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
        dones = torch.tensor([obs.done for obs in observations], dtype=torch.float32, device=self.device)

        states_0 = torch.tensor(np.array([obs.state for obs in observations]), dtype=torch.float32, device=self.device)
        states_1 = torch.tensor(np.array([obs.next_state for obs in observations]), dtype=torch.float32, device=self.device)

        actions_hot = F.one_hot(actions, num_classes=self.num_actions).float()

        # calculate the values for the next states
        values_1 = calculate_value(states_1)
        values_1 = values_1.view(-1, 1)

        # get the predicted actions for the next states
        with torch.no_grad():
            p_actions_probs = self.actor_model.model(states_1)
            p_actions = torch.argmax(p_actions_probs, dim=1)
        p_actions_hot = F.one_hot(p_actions, num_classes=self.num_actions).float()

        with torch.no_grad():
            q_next = self.model(states_1, p_actions_hot)
        q_next = q_next.view(-1, 1)

        # compute target values
        immediate_reward = (1 - self.discount_factor) * values_1
        future_reward = self.discount_factor * q_next
        target_values = immediate_reward + future_reward

        # For done indices, set target values to values_1
        dones = dones.view(-1, 1)
        done_indices = (dones == 1.0).nonzero(as_tuple=True)[0]
        target_values[done_indices] = values_1[done_indices]

        # train the network
        self.optimizer.zero_grad()
        outputs = self.model(states_0, actions_hot)
        loss = self.criterion(outputs, target_values)
        loss.backward()
        self.optimizer.step()

    def save(self):
        fd1, tmp_path = tempfile.mkstemp(prefix='.tmp_critic_', suffix='.pt', dir=os.path.dirname(self.model_path) or '.')
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

