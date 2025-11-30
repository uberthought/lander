import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import tempfile

from configuration import STATE_SIZE, ACTION_COUNT, NUM_ACTIONS, LAYER_COUNT, NODE_COUNT

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

# PyTorch World Model
# input is the current state plus the action one-hot encoded
# output is the predicted next state
class WorldNet(nn.Module):
    def __init__(self, input_dim, nodes, layers, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.nodes = nodes
        self.layers = layers
        
        self.input = nn.Linear(self.input_dim, nodes)
        self.skip_layers = nn.ModuleList([SkipBlock(nodes) for _ in range(layers)])
        self.output = nn.Sequential(
            nn.Linear(nodes, output_dim),
            nn.Tanh(),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = self.input(x)
        for i in range(self.layers):
            x = self.skip_layers[i](x)
        x = self.output(x)
        x = x / 10.0
        return x


class WorldModel:
    def __init__(self, model_path="models/world_model.pt"):
        self.model_path = model_path
        self.input_dim = STATE_SIZE
        self.num_actions = NUM_ACTIONS
        self.action_count = ACTION_COUNT
        self.nodes = NODE_COUNT
        self.layers = LAYER_COUNT

        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        net_input_dim = self.input_dim + self.num_actions * self.action_count
        self.model = WorldNet(net_input_dim, self.nodes, self.layers, self.input_dim)
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, map_location="cpu"))
        
        self.model.to(self.device)
        
        self.optimizer = optim.AdamW(self.model.parameters())
        self.criterion = nn.MSELoss()
        
    def train(self, observations):
        self.model.train()

        actions = torch.tensor(np.array([[int(a) for a in obs.actions] for obs in observations]), dtype=torch.long, device=self.device)
        states_0 = torch.tensor(np.array([obs.prev_state for obs in observations]), dtype=torch.float32, device=self.device)
        states_1 = torch.tensor(np.array([obs.next_state for obs in observations]), dtype=torch.float32, device=self.device)

        delta = states_1 - states_0

        actions_hot = F.one_hot(actions, num_classes=self.num_actions).float()
        actions_hot = actions_hot.view(actions_hot.size(0), -1)

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


    def predict(self, state, actions):
        self.model.eval()
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).reshape(1, -1)
            actions_tensor = torch.tensor(actions, dtype=torch.long, device=self.device).reshape(1, -1)
            actions_hot = F.one_hot(actions_tensor, num_classes=self.num_actions).float()
            actions_hot = actions_hot.view(1, -1)
            delta = self.model(state_tensor, actions_hot).cpu().numpy().flatten()
        return delta