import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import tempfile

from observation import calculate_value

class SkipBlock(nn.Module):
    def __init__(self, nodes):
        super().__init__()
        self.block1 = nn.Linear(nodes, nodes)
        self.relu1 = nn.LeakyReLU()
        self.block2 = nn.Linear(nodes, nodes)
        self.relu2 = nn.LeakyReLU()
    def forward(self, x):
        s = x
        x = self.block1(x)
        x = self.relu1(x)
        x = self.block2(x)
        x = x + s
        x = self.relu2(x)
        return x

# PyTorch World Model
# input is the current state plus the action one-hot encoded
# output is the predicted next state
class WorldNet(nn.Module):
    def __init__(self, input_dim, num_actions, nodes, layers):
        super().__init__()
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.nodes = nodes
        self.layers = layers
        
        self.input = nn.Linear(self.input_dim + self.num_actions, nodes)
        self.skip_layers = nn.ModuleList([SkipBlock(nodes) for _ in range(layers)])
        self.output = nn.Linear(nodes, self.input_dim)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = self.input(x)
        for i in range(self.layers):
            x = self.skip_layers[i](x)
        x = self.output(x)
        return x


class WorldModel:
    def __init__(self, model_path="models/world_model.pt"):
        self.model_path = model_path
        self.input_dim = 9
        self.num_actions = 4
        self.nodes = (self.input_dim + self.num_actions) * 8
        self.layers = 24

        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        self.model = WorldNet(self.input_dim, self.num_actions, self.nodes, self.layers)
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, map_location="cpu"))
        
        self.model.to(self.device)
        
        self.optimizer = optim.AdamW(self.model.parameters())
        self.criterion = nn.MSELoss()
        
    def train(self, observations):
        self.model.train()

        actions = torch.tensor([int(obs.action) for obs in observations], dtype=torch.long, device=self.device)
        states_0 = torch.tensor(np.array([obs.prev_state for obs in observations]), dtype=torch.float32, device=self.device)
        states_1 = torch.tensor(np.array([obs.next_state for obs in observations]), dtype=torch.float32, device=self.device)

        delta = states_1 - states_0

        actions_hot = F.one_hot(actions, num_classes=self.num_actions).float()

        # train the network
        self.optimizer.zero_grad()
        outputs = self.model(states_0, actions_hot)
        loss = self.criterion(outputs, delta)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def save(self):
        fd1, tmp_path = tempfile.mkstemp(prefix='.tmp_world_', suffix='.pt', dir=os.path.dirname(self.model_path) or '.')
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


    def predict(self, state, action):
        self.model.eval()
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).reshape(1, -1)
            action_hot = torch.zeros((1, self.num_actions), dtype=torch.float32, device=self.device)
            action_hot[0, action] = 1.0
            delta = self.model(state_tensor, action_hot).cpu().numpy().flatten()
        return delta