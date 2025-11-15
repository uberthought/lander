import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import tempfile

from observation import calculate_value

# PyTorch Actor Model
# input is the current state plus the action one-hot encoded
# output is the predicted reward for the action
class ActorNet(nn.Module):
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
            # nn.Sigmoid()
            # nn.Tanh()
        )

    def forward(self, x):
        x = self.input(x)
        for skip_layer in self.skip_layers:
            x = x + skip_layer(x)
        x = self.output(x)
        return x


class ActorModel:
    def __init__(self, discount_factor=0.95, model_path="models/actor_model.pt"):
        self.discount_factor = discount_factor
        self.model_path = model_path
        self.q1_model_path = model_path.replace(".pt", "_q1.pt")
        self.q2_model_path = model_path.replace(".pt", "_q2.pt")
        self.input_dim = 10
        self.num_actions = 4
        self.nodes = self.input_dim * self.num_actions * 4
        self.layers = 4

        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.q1_model, self.q2_model = self._load_models() or self._create_models()
        self.q1_model.to(self.device)
        self.q2_model.to(self.device)
        self.q1_optimizer = optim.Adam(self.q1_model.parameters())
        self.q2_optimizer = optim.Adam(self.q2_model.parameters())
        self.criterion = nn.MSELoss()

    def _create_models(self):
        q1_model = ActorNet(self.input_dim, self.num_actions, self.nodes, self.layers)
        q2_model = ActorNet(self.input_dim, self.num_actions, self.nodes, self.layers)
        return q1_model, q2_model

    def _load_models(self):
        if os.path.exists(self.q1_model_path) and os.path.exists(self.q2_model_path):
            q1_model = ActorNet(self.input_dim, self.num_actions, self.nodes, self.layers)
            q2_model = ActorNet(self.input_dim, self.num_actions, self.nodes, self.layers)
            q1_model.load_state_dict(torch.load(self.q1_model_path, map_location="cpu"))
            q2_model.load_state_dict(torch.load(self.q2_model_path, map_location="cpu"))
            return q1_model, q2_model
        return None


    def train(self, observations):
        self.q1_model.train()
        self.q2_model.train()

        states_0 = torch.tensor(np.array([obs.state for obs in observations]), dtype=torch.float32, device=self.device)
        states_1 = torch.tensor(np.array([obs.next_state for obs in observations]), dtype=torch.float32, device=self.device)
        actions_0 = torch.tensor([int(obs.action) for obs in observations], dtype=torch.long, device=self.device)
        dones_1 = torch.tensor([obs.next_state[-1] for obs in observations], dtype=torch.float32, device=self.device)

        actions_0_onehot = F.one_hot(actions_0, num_classes=self.num_actions).float()

        states_1_tile = states_1.unsqueeze(1).repeat(1, self.num_actions, 1)
        actions_1_onehot = F.one_hot(torch.arange(self.num_actions, device=self.device), num_classes=self.num_actions).unsqueeze(0).repeat(len(states_1), 1, 1)
        actor_inputs_1 = torch.cat([states_1_tile, actions_1_onehot], dim=-1).view(-1, self.input_dim + self.num_actions)

        with torch.no_grad():
            q1_values_2 = self.q1_model(actor_inputs_1).view(-1, self.num_actions)
            q2_values_2 = self.q2_model(actor_inputs_1).view(-1, self.num_actions)

        # Use minimum of the two Q-values (SAC approach)
        # p_values_2 = torch.min(q1_values_2, q2_values_2)
        p_values_2 = torch.max(q1_values_2, dim=1)[0].view(-1, 1)

        values_0 = calculate_value(states_0).view(-1, 1)
        values_1 = calculate_value(states_1).view(-1, 1)
        # current_values = (1 - self.discount_factor) * values_1
        current_values = values_1 - values_0
        future_values = self.discount_factor * p_values_2
        values = current_values + future_values

        done_indicies = (dones_1 == 1.0).nonzero(as_tuple=True)[0]
        values[done_indicies] = values_1[done_indicies]

        model_input = torch.cat([states_0, actions_0_onehot], dim=-1)

        for _ in range(4):
            # Train Q1 network
            self.q1_optimizer.zero_grad()
            q1_outputs = self.q1_model(model_input)
            q1_loss = self.criterion(q1_outputs, values)
            q1_loss.backward()
            self.q1_optimizer.step()
            
            # Train Q2 network
            self.q2_optimizer.zero_grad()
            q2_outputs = self.q2_model(model_input)
            q2_loss = self.criterion(q2_outputs, values)
            q2_loss.backward()
            self.q2_optimizer.step()

    def save(self):
        # Save Q1 model
        fd1, tmp_path1 = tempfile.mkstemp(prefix='.tmp_actor_q1_', suffix='.pt', dir=os.path.dirname(self.q1_model_path) or '.')
        os.close(fd1)
        # Save Q2 model
        fd2, tmp_path2 = tempfile.mkstemp(prefix='.tmp_actor_q2_', suffix='.pt', dir=os.path.dirname(self.q2_model_path) or '.')
        os.close(fd2)
        try:
            torch.save(self.q1_model.state_dict(), tmp_path1)
            torch.save(self.q2_model.state_dict(), tmp_path2)
            os.replace(tmp_path1, self.q1_model_path)
            os.replace(tmp_path2, self.q2_model_path)
        except KeyboardInterrupt:
            if os.path.exists(tmp_path1):
                os.remove(tmp_path1)
            if os.path.exists(tmp_path2):
                os.remove(tmp_path2)
            raise
        except Exception:
            if os.path.exists(tmp_path1):
                os.remove(tmp_path1)
            if os.path.exists(tmp_path2):
                os.remove(tmp_path2)
            raise

    def get_optimal_action(self, obs):
        action_values = self.get_all_actions(obs)

        # greedy action selection
        # action1 = torch.argmax(action_values).item()

        # stochastic action selection
        action_softmax = F.softmax(action_values.view(-1), dim=0)
        action2 = torch.multinomial(action_softmax, num_samples=1).item()

        action = action2

        # 5% chance to explore
        # action = action2 if np.random.rand() < 0.05 else action1


        
        return int(action), action_values.view(-1).cpu().numpy()

    def get_all_actions(self, obs):
        self.q1_model.eval()
        sensors = torch.tensor([obs], dtype=torch.float32, device=self.device)
        sensors_tile = sensors.unsqueeze(1).repeat(1, self.num_actions, 1)
        actions_arange = torch.arange(self.num_actions, device=self.device)
        actions_onehot = F.one_hot(actions_arange, num_classes=self.num_actions)
        actions_tile = actions_onehot.unsqueeze(0).repeat(len(sensors), 1, 1)
        actor_inputs = torch.cat([sensors_tile, actions_tile], dim=-1).view(-1, self.input_dim + self.num_actions)
        with torch.no_grad():
            action_values = self.q1_model(actor_inputs)

        return action_values