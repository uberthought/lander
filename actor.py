import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import tempfile

from observation import calculate_value



# PyTorch Actor Model
class ActorNet(nn.Module):
    def __init__(self, input_dim, num_actions, nodes, layers):
        super().__init__()
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.nodes = nodes
        self.layers = layers
        

        self.input = nn.Sequential(
            nn.Linear(self.input_dim, nodes),
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
            nn.Linear(nodes, num_actions),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = self.input(x)
        for skip_layer in self.skip_layers:
            skip = x
            x = skip_layer(x)
            x = x + skip
        x = self.output(x)
        return x


class ActorModel:
    def __init__(self, model_path="models/actor_model.pt"):
        self.model_path = model_path
        self.input_dim = 10
        self.num_actions = 4
        self.nodes = self.input_dim * 16
        self.layers = 4

        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = self._load_model() or self._create_model()
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.MSELoss()

    def set_critic_model(self, critic_model):
        self.critic_model = critic_model

    def _create_model(self):
        return ActorNet(self.input_dim, self.num_actions, self.nodes, self.layers)

    def _load_model(self):
        if os.path.exists(self.model_path):
            model = ActorNet(self.input_dim, self.num_actions, self.nodes, self.layers)
            model.load_state_dict(torch.load(self.model_path, map_location="cpu"))
            return model
        return None

    def train(self, observations):
        self.model.train()
        states_0 = torch.tensor(np.array([obs.state for obs in observations]), dtype=torch.float32, device=self.device)
        states_1 = torch.tensor(np.array([obs.next_state for obs in observations]), dtype=torch.float32, device=self.device)

        sensors_1_tiled = states_1.repeat_interleave(self.num_actions, dim=0)
        actions_onehot_tiled = torch.eye(self.num_actions, device=self.device).repeat(len(observations), 1)

        with torch.no_grad():
            predicted_rewards = self.critic_model.model(sensors_1_tiled, actions_onehot_tiled)
            predicted_rewards = predicted_rewards.view(len(observations), self.num_actions)
            predicted_actions = torch.argmax(predicted_rewards, dim=1)

        best_actions = F.one_hot(predicted_actions, num_classes=self.num_actions).float()

        for _ in range(4):
            self.optimizer.zero_grad()
            outputs = self.model(states_0)
            loss = self.criterion(outputs, best_actions)
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
        self.model.eval()
        sensors = torch.tensor([obs], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            prediction0 = self.model(sensors)[0]
        prediction_np = prediction0.cpu().numpy()
        action = np.random.choice(self.num_actions, p=prediction_np)
        return int(action), prediction_np