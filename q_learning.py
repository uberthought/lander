import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import tempfile

from observation import calculate_reward

# PyTorch QLearning Model
# input is the current state plus the action one-hot encoded
# output is the predicted reward for the action
class QLearningNet(nn.Module):
    def __init__(self, input_dim, num_actions, nodes, layers):
        super().__init__()
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.nodes = nodes
        self.layers = layers
        
        self.input = nn.Sequential(
            nn.Linear(self.input_dim + self.num_actions, nodes),
            nn.LeakyReLU(),
            nn.Linear(nodes, nodes),
            nn.LeakyReLU(),
        )

        self.skip_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(nodes, nodes),
                nn.LeakyReLU(),
                nn.Linear(nodes, nodes),
                nn.LeakyReLU(),
                nn.Linear(nodes, nodes),
                nn.BatchNorm1d(nodes),
            ) for _ in range(layers)
        ])

        self.output = nn.Sequential(
            nn.Linear(nodes, nodes),
            nn.LeakyReLU(),
            nn.Linear(nodes, 1)
        )

    def forward(self, x):
        x = self.input(x)
        for skip_layer in self.skip_layers:
            x = x + skip_layer(x)
        x = self.output(x)
        return x


class QLearningModel:
    def __init__(self, discount_factor=0.95, model_path="models/qlearning_model.pt"):
        self.discount_factor = discount_factor
        self.model_path = model_path
        self.q1_model_path = model_path.replace(".pt", "_q1.pt")
        self.q2_model_path = model_path.replace(".pt", "_q2.pt")
        self.input_dim = 10
        self.num_actions = 4
        self.nodes = self.input_dim * self.num_actions * 4
        self.q1_layers = 16
        self.q2_layers = 8

        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.q1_model, self.q2_model = self._load_models() or self._create_models()
        self.q1_model.to(self.device)
        self.q2_model.to(self.device)
        self.q1_optimizer = optim.Adam(self.q1_model.parameters())
        self.q2_optimizer = optim.Adam(self.q2_model.parameters())
        self.criterion = nn.MSELoss()

    def _create_models(self):
        q1_model = QLearningNet(self.input_dim, self.num_actions, self.nodes, self.q1_layers)
        q2_model = QLearningNet(self.input_dim, self.num_actions, self.nodes, self.q2_layers)
        return q1_model, q2_model

    def _load_models(self):
        if os.path.exists(self.q1_model_path) and os.path.exists(self.q2_model_path):
            q1_model = QLearningNet(self.input_dim, self.num_actions, self.nodes, self.q1_layers)
            q2_model = QLearningNet(self.input_dim, self.num_actions, self.nodes, self.q2_layers)
            q1_model.load_state_dict(torch.load(self.q1_model_path, map_location="cpu"))
            q2_model.load_state_dict(torch.load(self.q2_model_path, map_location="cpu"))
            return q1_model, q2_model
        return None


    def train(self, new_observations, replay_buffer, iterations=24):
        self.q1_model.train()
        self.q2_model.train()

        sample_len = 2 ** 13

        print(f"Training for {iterations} iterations with {len(new_observations)} new observations and replay buffer size {len(replay_buffer)}")

        for i in range(iterations):
            observations = replay_buffer.sample(sample_len) + new_observations

            states_0 = torch.tensor(np.array([obs.state for obs in observations]), dtype=torch.float32, device=self.device)
            states_1 = torch.tensor(np.array([obs.next_state for obs in observations]), dtype=torch.float32, device=self.device)
            actions_0 = torch.tensor([int(obs.action) for obs in observations], dtype=torch.long, device=self.device)
            dones_1 = torch.tensor([obs.next_state[-1] for obs in observations], dtype=torch.float32, device=self.device)

            actions_0_onehot = F.one_hot(actions_0, num_classes=self.num_actions).float()

            states_1_tile = states_1.unsqueeze(1).repeat(1, self.num_actions, 1)
            actions_1_onehot = F.one_hot(torch.arange(self.num_actions, device=self.device), num_classes=self.num_actions).unsqueeze(0).repeat(len(states_1), 1, 1)
            inputs_1 = torch.cat([states_1_tile, actions_1_onehot], dim=-1).view(-1, self.input_dim + self.num_actions)

            with torch.no_grad():
                q1_values_2 = self.q1_model(inputs_1).view(-1, self.num_actions)
                q2_values_2 = self.q2_model(inputs_1).view(-1, self.num_actions)

            # Use minimum of the two Q-values (SAC approach)
            min_q_values_2 = torch.min(q1_values_2, q2_values_2)
            p_rewards_2 = torch.max(min_q_values_2, dim=1)[0].view(-1, 1)

            rewards_1 = calculate_reward(states_1).view(-1, 1)
            future_rewards = self.discount_factor * p_rewards_2
            rewards = rewards_1 + future_rewards

            done_indicies = (dones_1 == 1.0).nonzero(as_tuple=True)[0]
            rewards[done_indicies] = rewards_1[done_indicies]

            # input is state + action one-hot encoded
            model_input = torch.cat([states_0, actions_0_onehot], dim=-1)

            # Train Q1 network
            self.q1_optimizer.zero_grad()
            q1_outputs = self.q1_model(model_input)
            q1_loss = self.criterion(q1_outputs, rewards)
            q1_loss.backward()
            self.q1_optimizer.step()
            
            # Train Q2 network
            self.q2_optimizer.zero_grad()
            q2_outputs = self.q2_model(model_input)
            q2_loss = self.criterion(q2_outputs, rewards)
            q2_loss.backward()
            self.q2_optimizer.step()

    def save(self):
        # Save Q1 model
        fd1, tmp_path1 = tempfile.mkstemp(prefix='.tmp_qlearning_q1_', suffix='.pt', dir=os.path.dirname(self.q1_model_path) or '.')
        os.close(fd1)
        # Save Q2 model
        fd2, tmp_path2 = tempfile.mkstemp(prefix='.tmp_qlearning_q2_', suffix='.pt', dir=os.path.dirname(self.q2_model_path) or '.')
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
        action_values = self.get_all_actions(obs).view(-1)

        # greedy action selection
        # action = torch.argmax(action_values).item()

        # stochastic action selection
        action_softmax = F.softmax(action_values, dim=0)
        action = torch.multinomial(action_softmax, num_samples=1).item()

        return int(action), action_values.cpu().numpy()

    def get_all_actions(self, obs):
        self.q1_model.eval()
        sensors = torch.tensor([obs], dtype=torch.float32, device=self.device)
        sensors_tile = sensors.unsqueeze(1).repeat(1, self.num_actions, 1)
        actions_arange = torch.arange(self.num_actions, device=self.device)
        actions_onehot = F.one_hot(actions_arange, num_classes=self.num_actions)
        actions_tile = actions_onehot.unsqueeze(0).repeat(len(sensors), 1, 1)
        inputs = torch.cat([sensors_tile, actions_tile], dim=-1).view(-1, self.input_dim + self.num_actions)
        with torch.no_grad():
            action_values = self.q1_model(inputs)

        return action_values