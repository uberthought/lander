import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import tempfile

from configuration import STATE_SIZE, POSSIBLE_ACTIONS, LAYER_COUNT, NODE_COUNT, WORLD_LOSS_DELTA

STATE_PARAMETER_NAMES = ["x", "y", "vx", "vy", "angle", "vangle", "left_leg", "right_leg", "fuel", "sin_angle", "cos_angle"]


class SkipBlock(nn.Module):
    def __init__(self, nodes):
        super().__init__()
        self.block0 = nn.Linear(nodes, nodes)
        self.relu0 = nn.LeakyReLU()
        self.block1 = nn.Linear(nodes, nodes)
        self.relu1 = nn.LeakyReLU()
        self.block2 = nn.Linear(nodes, nodes)
        self.relu2 = nn.LeakyReLU()

    def forward(self, input):
        s = input
        x = input
        x = self.block0(x)
        x = self.relu0(x)
        x = self.block1(x)
        x = self.relu1(x)
        x = self.block2(x)
        x = x + s
        x = self.relu2(x)
        return x


# PyTorch World Model
# input is the current state plus the action one-hot encoded
# output is the predicted next-state delta
class WorldNet(nn.Module):
    def __init__(self, input_dim, nodes, layers, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.nodes = nodes
        self.layers = layers

        self.input = nn.Linear(self.input_dim, nodes)
        self.skip_layers = nn.ModuleList([SkipBlock(nodes) for _ in range(layers)])
        self.output = nn.Linear(nodes, output_dim)
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
        self.input_dim = STATE_SIZE
        self.possible_actions = POSSIBLE_ACTIONS
        self.nodes = NODE_COUNT
        self.layers = LAYER_COUNT

        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        net_input_dim = self.input_dim + self.possible_actions
        self.model = WorldNet(net_input_dim, self.nodes, self.layers, self.input_dim)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001)
        self.criterion = nn.HuberLoss(delta=WORLD_LOSS_DELTA, reduction='none')

        if os.path.exists(self.model_path):
            # checkpoint = torch.load(self.model_path, map_location="cpu")
            checkpoint = torch.load(self.model_path, map_location="mps" if torch.backends.mps.is_available() else "cpu")
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint and 'optimizer_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.to(self.device)
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print(f"Loaded model and optimizer state from {self.model_path}")
            else:
                # Backward compatibility: support old format (just state_dict)
                self.model.load_state_dict(checkpoint)
                self.model.to(self.device)
                print(f"Loaded model weights from {self.model_path} (no optimizer state)")
        else:
            print(f"No checkpoint found at {self.model_path}; starting with random weights.")
            self.model.to(self.device)

    def train(self, observations):
        if not observations:
            return None

        self.model.train()

        actions = torch.tensor(np.array([[int(a) for a in obs.actions] for obs in observations]), dtype=torch.long, device=self.device)
        states_0_np = np.array([obs.prev_state for obs in observations], dtype=np.float32)
        states_1_np = np.array([obs.next_state for obs in observations], dtype=np.float32)
        states_0 = torch.tensor(states_0_np, dtype=torch.float32, device=self.device)
        states_1 = torch.tensor(states_1_np, dtype=torch.float32, device=self.device)
        delta = states_1 - states_0

        actions_hot = F.one_hot(actions, num_classes=self.possible_actions).float()
        actions_hot = actions_hot.view(actions_hot.size(0), -1)

        self.optimizer.zero_grad()
        outputs = self.model(states_0, actions_hot)

        losses = self.criterion(outputs, delta)
        delta_var = delta.var(dim=0).clamp(min=1e-6)
        weights = (1.0 / delta_var)
        weights = weights / weights.mean()
        loss = (losses * weights).mean()
        loss.backward()

        self.optimizer.step()


    def save(self):
        fd1, tmp_path = tempfile.mkstemp(prefix='.tmp_world_', suffix='.pt', dir=os.path.dirname(self.model_path) or '.')
        os.close(fd1)

        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, tmp_path)
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

    def predict_batch(self, states, actions):
        self.model.eval()
        with torch.no_grad():
            states_np = np.array(states, dtype=np.float32)
            state_tensor = torch.tensor(states_np, dtype=torch.float32, device=self.device)
            actions_tensor = torch.tensor(np.array(actions), dtype=torch.long, device=self.device)
            actions_hot = F.one_hot(actions_tensor, num_classes=self.possible_actions).float()
            actions_hot = actions_hot.view(actions_hot.size(0), -1)
            result = self.model(state_tensor, actions_hot).cpu().numpy()
        return result

    def predict(self, state, actions):
        return self.predict_batch([state], [actions])[0]