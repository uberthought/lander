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
        
        self.input = nn.Sequential(
            nn.Linear(self.input_dim + self.num_actions, nodes),
            nn.Linear(nodes, nodes),
            nn.LeakyReLU(),
            nn.Linear(nodes, nodes),
            nn.LeakyReLU()
        )

        self.skip_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(nodes, nodes),
                nn.LeakyReLU(),
                nn.Linear(nodes, nodes),
                nn.LeakyReLU()
            ) for _ in range(layers)
        ])

        self.output = nn.Sequential(
            nn.Linear(nodes, nodes),
            nn.LeakyReLU(),
            nn.Linear(nodes, nodes),
            nn.LeakyReLU(),
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
        self.q1_model_path = model_path.replace('.pt', '_q1.pt')
        self.q2_model_path = model_path.replace('.pt', '_q2.pt')
        self.input_dim = 10
        self.num_actions = 4
        self.discount_factor = discount_factor
        self.nodes = self.input_dim * self.num_actions * 4
        self.layers = 4

        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # Create two Q-networks (Q1 and Q2)
        self.q1_model = self._load_model(self.q1_model_path) or self._create_model()
        self.q2_model = self._load_model(self.q2_model_path) or self._create_model()
        
        self.q1_model.to(self.device)
        self.q2_model.to(self.device)
        
        # Separate optimizers for each network
        self.q1_optimizer = optim.Adam(self.q1_model.parameters())
        self.q2_optimizer = optim.Adam(self.q2_model.parameters())
        self.criterion = nn.MSELoss()
        
        # Keep backward compatibility - model refers to q1_model
        self.model = self.q1_model
        self.optimizer = self.q1_optimizer

    def set_actor_model(self, actor_model):
        self.actor_model = actor_model
    
    def get_q_values(self, states, actions):
        """Get Q-values from both networks"""
        self.q1_model.eval()
        self.q2_model.eval()
        with torch.no_grad():
            q1_values = self.q1_model(states, actions)
            q2_values = self.q2_model(states, actions)
        return q1_values, q2_values
    
    def get_min_q_values(self, states, actions):
        """Get minimum Q-values from both networks (SAC-style)"""
        q1_values, q2_values = self.get_q_values(states, actions)
        return torch.min(q1_values, q2_values)

    def _create_model(self):
        return CriticNet(self.input_dim, self.num_actions, self.nodes, self.layers)

    def _load_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        if os.path.exists(model_path):
            model = CriticNet(self.input_dim, self.num_actions, self.nodes, self.layers)
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
            return model
        return None

    def train(self, observations):
        self.q1_model.train()
        self.q2_model.train()
        
        actions = torch.tensor([int(obs.action) for obs in observations], dtype=torch.long, device=self.device)
        dones = torch.tensor([obs.next_state[-1] for obs in observations], dtype=torch.float32, device=self.device)

        states_0 = torch.tensor(np.array([obs.state for obs in observations]), dtype=torch.float32, device=self.device)
        states_1 = torch.tensor(np.array([obs.next_state for obs in observations]), dtype=torch.float32, device=self.device)

        actions_hot = F.one_hot(actions, num_classes=self.num_actions).float()

        # calculate the values for the current states
        # values_0 = calculate_value(states_0)
        # values_0 = values_0.view(-1, 1)

        # calculate the values for the next states
        values_1 = calculate_value(states_1)
        values_1 = values_1.view(-1, 1)

        # get the predicted actions for the next states
        with torch.no_grad():
            p_actions_probs = self.actor_model.model(states_1)
            p_actions = torch.argmax(p_actions_probs, dim=1)
        p_actions_hot = F.one_hot(p_actions, num_classes=self.num_actions).float()

        # get the predicted values for the next state-action pairs from both Q-networks
        # Use the minimum of Q1 and Q2 for target calculation (SAC-style)
        with torch.no_grad():
            q1_next = self.q1_model(states_1, p_actions_hot)
            q2_next = self.q2_model(states_1, p_actions_hot)
            min_q_next = torch.min(q1_next, q2_next)
        min_q_next = min_q_next.view(-1, 1)

        target_values = (1 - self.discount_factor) * values_1 + self.discount_factor * min_q_next

        # For done indices, set target values to values_1
        dones = dones.view(-1, 1)
        done_indices = (dones == 1.0).nonzero(as_tuple=True)[0]
        target_values[done_indices] = values_1[done_indices]

        # Train both Q-networks for 4 epochs
        for _ in range(4):
            # Train Q1 network
            self.q1_optimizer.zero_grad()
            q1_outputs = self.q1_model(states_0, actions_hot)
            q1_loss = self.criterion(q1_outputs, target_values)
            q1_loss.backward()
            self.q1_optimizer.step()
            
            # Train Q2 network
            self.q2_optimizer.zero_grad()
            q2_outputs = self.q2_model(states_0, actions_hot)
            q2_loss = self.criterion(q2_outputs, target_values)
            q2_loss.backward()
            self.q2_optimizer.step()

    def save(self):
        # Save Q1 network
        fd1, tmp_path1 = tempfile.mkstemp(prefix='.tmp_critic_q1_', suffix='.pt', dir=os.path.dirname(self.q1_model_path) or '.')
        os.close(fd1)
        
        # Save Q2 network
        fd2, tmp_path2 = tempfile.mkstemp(prefix='.tmp_critic_q2_', suffix='.pt', dir=os.path.dirname(self.q2_model_path) or '.')
        os.close(fd2)
        
        try:
            torch.save(self.q1_model.state_dict(), tmp_path1)
            torch.save(self.q2_model.state_dict(), tmp_path2)
            os.replace(tmp_path1, self.q1_model_path)
            os.replace(tmp_path2, self.q2_model_path)
            
            # Also save Q1 as the main model for backward compatibility
            fd_main, tmp_path_main = tempfile.mkstemp(prefix='.tmp_critic_', suffix='.pt', dir=os.path.dirname(self.model_path) or '.')
            os.close(fd_main)
            torch.save(self.q1_model.state_dict(), tmp_path_main)
            os.replace(tmp_path_main, self.model_path)
            
        except KeyboardInterrupt:
            for tmp_path in [tmp_path1, tmp_path2]:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            raise
        except Exception:
            for tmp_path in [tmp_path1, tmp_path2]:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            raise

