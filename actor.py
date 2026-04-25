import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import tempfile

from configuration import STATE_SIZE, POSSIBLE_ACTIONS, LAYER_COUNT, NODE_COUNT
from observation import calculate_reward
from world import WorldModel

class SkipBlock(nn.Module):
    def __init__(self, nodes):
        super().__init__()
        self.block1 = nn.Linear(nodes, nodes)
        self.relu1 = nn.LeakyReLU()
        self.block2 = nn.Linear(nodes, nodes)
        self.relu2 = nn.LeakyReLU()
    def forward(self, input):
        s = input
        x = self.block1(input)
        x = self.relu1(x)
        x = self.block2(x)
        x = x + s
        x = self.relu2(x)
        return x

# input is the current state; output is logits over POSSIBLE_ACTIONS
class ActorNet(nn.Module):
    def __init__(self, input_dim, nodes, layers, output_dim):
        super().__init__()
        self.layers = layers

        self.input = nn.Linear(input_dim, nodes)
        self.skip_layers = nn.ModuleList([SkipBlock(nodes) for _ in range(layers)])
        self.output = nn.Sequential(
                    nn.Linear(nodes, nodes),
                    nn.LeakyReLU(),
                    nn.Linear(nodes, nodes),
                    nn.LeakyReLU(),
                    nn.Linear(nodes, nodes),
                    nn.LeakyReLU(),
                    nn.Linear(nodes, nodes),
                    nn.LeakyReLU(),
                    nn.Linear(nodes, output_dim)
                )

    def forward(self, state):
        x = self.input(state)
        for i in range(self.layers):
            x = self.skip_layers[i](x)
        return self.output(x)


class ActorModel:
    def __init__(self, world_model, model_path="models/actor_model.pt"):
        self.model_path = model_path
        self.possible_actions = POSSIBLE_ACTIONS
        self.world_model = world_model

        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        self.model = ActorNet(STATE_SIZE, NODE_COUNT, LAYER_COUNT, POSSIBLE_ACTIONS)
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, map_location="cpu"))
            print(f"Loaded model weights from {self.model_path}")
        else:
            print(f"No checkpoint found at {self.model_path}; starting with random weights.")
        self.model.to(self.device)

        self.optimizer = optim.AdamW(self.model.parameters())

    def train(self, observations):
        self.model.train()

        states_0 = torch.tensor(np.array([obs.prev_state for obs in observations]), dtype=torch.float32, device=self.device)
        states_1 = torch.tensor(np.array([obs.next_state for obs in observations]), dtype=torch.float32, device=self.device)
        states_all = torch.cat([states_0, states_1], dim=0)

        with torch.no_grad():
            B = states_all.size(0)
            A = self.possible_actions
            states_tiled = states_all.unsqueeze(1).repeat(1, A, 1).view(B * A, -1)
            actions_all = torch.arange(A, device=self.device)
            actions_onehot_tiled = F.one_hot(actions_all, num_classes=A).float().unsqueeze(0).repeat(B, 1, 1).view(B * A, -1)
            predicted_next_states = self.world_model.model(states_tiled, actions_onehot_tiled)
            predicted_rewards = calculate_reward(predicted_next_states).view(B, A)
            target_actions = torch.argmax(predicted_rewards, dim=1)

        self.optimizer.zero_grad()
        logits = self.model(states_all)
        loss = F.cross_entropy(logits, target_actions)
        loss.backward()
        self.optimizer.step()

    def save(self):
        fd1, tmp_path = tempfile.mkstemp(prefix='.tmp_actor_', suffix='.pt', dir=os.path.dirname(self.model_path) or '.')
        os.close(fd1)

        try:
            torch.save(self.model.state_dict(), tmp_path)
            os.replace(tmp_path, self.model_path)
        except (KeyboardInterrupt, Exception):
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    def get_best_action(self, state):
        self.model.eval()

        state_tensor = torch.tensor(np.array(state), dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(state_tensor)

        return torch.argmax(logits, dim=1).item()
