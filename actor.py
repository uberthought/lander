import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import tempfile

from configuration import STATE_SIZE, POSSIBLE_ACTIONS, NUM_ACTIONS, LAYER_COUNT, NODE_COUNT
from observation import calculate_reward

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

# PyTorch Actor Model
# input is the current state and a list of actions (one-hot encoded)
# output is the predicted reward for the given state and action
class ActorNet(nn.Module):
    def __init__(self, input_dim, nodes, layers, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.nodes = nodes
        self.layers = layers
        
        self.input = nn.Linear(self.input_dim, nodes)
        self.skip_layers = nn.ModuleList([SkipBlock(nodes) for _ in range(layers)])
        self.output = nn.Linear(nodes, output_dim)

    def forward(self, state):
        x = self.input(state)
        for i in range(self.layers):
            x = self.skip_layers[i](x)
        x = self.output(x)
        return x


class ActorModel:
    def __init__(self, model_path="models/actor_model.pt"):
        self.model_path = model_path
        self.input_dim = STATE_SIZE
        self.num_actions = NUM_ACTIONS
        self.possible_actions = POSSIBLE_ACTIONS
        self.nodes = NODE_COUNT
        self.layers = LAYER_COUNT
        self.discount_factor = 0.5

        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        net_input_dim = self.input_dim + self.num_actions * self.possible_actions
        self.model = ActorNet(net_input_dim, self.nodes, self.layers, 1)
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, map_location="cpu"))
        
        self.model.to(self.device)
        
        self.optimizer = optim.AdamW(self.model.parameters())
        self.criterion = nn.MSELoss()
        
    def train(self, observations):
        self.model.train()

        states_0 = torch.tensor(np.array([obs.prev_state for obs in observations]), dtype=torch.float32, device=self.device)
        states_1 = torch.tensor(np.array([obs.next_state for obs in observations]), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array([[int(a) for a in obs.actions] for obs in observations]), dtype=torch.long, device=self.device)
        dones = torch.tensor([obs.done for obs in observations], dtype=torch.float32, device=self.device)

        actions_onehot = F.one_hot(actions, num_classes=self.possible_actions).float()
        actions_onehot = actions_onehot.view(-1, self.num_actions * self.possible_actions)

        states_1_tile = states_1.unsqueeze(1).repeat(1, self.possible_actions ** self.num_actions, 1)
        
        # actions_1 needs to be a tensor of possible actions, so for example
        # if possible_actions=4 and num_actions=2 it should be [0,0], [0,1], [0,2], [0,3], [1,0], [1,1], ... [3,3]
        grids = torch.meshgrid(*[torch.arange(self.possible_actions, device=self.device) for _ in range(self.num_actions)], indexing='ij')
        actions_1 = torch.stack(grids, dim=-1).reshape(-1, self.num_actions)  # shape: (possible_actions**num_actions, num_actions)
 
        actions_1_onehot = F.one_hot(actions_1, num_classes=self.possible_actions).unsqueeze(0)
        actions_1_onehot = actions_1_onehot.view(-1, self.num_actions * self.possible_actions)
        actions_1_onehot = actions_1_onehot.repeat(states_1.size(0), 1, 1)
        inputs_1 = torch.cat([states_1_tile, actions_1_onehot], dim=-1)

        # compute future rewards
        with torch.no_grad():
            q_rewards = self.model(inputs_1)

        future_rewards = torch.max(q_rewards, dim=1)[0]

        not_done_1 = 1.0 - dones.view(-1, 1)
        future_rewards = future_rewards * not_done_1

        # compute current rewards
        prev_rewards = calculate_reward(states_0).view(-1, 1)
        current_rewards = calculate_reward(states_1).view(-1, 1)

        # set the rewards target
        # rewards = (1.0 - self.discount_factor) * current_rewards + self.discount_factor * future_rewards
        rewards = (current_rewards - prev_rewards) + self.discount_factor * future_rewards

        # optimize the model
        self.optimizer.zero_grad()
        inputs_0 = torch.cat([states_0, actions_onehot], dim=-1)
        outputs = self.model(inputs_0)
        loss = self.criterion(outputs, rewards)
        loss.backward()
        self.optimizer.step()

    def save(self):
        fd1, tmp_path = tempfile.mkstemp(prefix='.tmp_actor_', suffix='.pt', dir=os.path.dirname(self.model_path) or '.')
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


    def get_best_actions(self, states):
        self.model.eval()
        
        states_tensor = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        
        grids = torch.meshgrid(*[torch.arange(self.possible_actions, device=self.device) for _ in range(self.num_actions)], indexing='ij')
        actions_1 = torch.stack(grids, dim=-1).reshape(-1, self.num_actions)  # shape: (possible_actions**num_actions, num_actions)
        actions_1_onehot = F.one_hot(actions_1, num_classes=self.possible_actions).unsqueeze(0)
        actions_1_onehot = actions_1_onehot.view(-1, self.num_actions * self.possible_actions)

        actions_1_onehot = actions_1_onehot.repeat(states_tensor.size(0), 1, 1)
        states_expanded = states_tensor.unsqueeze(1).repeat(1, self.possible_actions ** self.num_actions, 1)
        inputs_1 = torch.cat([states_expanded, actions_1_onehot], dim=-1)

        with torch.no_grad():
            predicted_rewards = self.model(inputs_1)
        predicted_rewards = predicted_rewards.view(states_tensor.size(0), self.possible_actions ** self.num_actions)

        best_action_index = torch.argmax(predicted_rewards, dim=1)
        best_actions = actions_1[best_action_index]
        
        return best_actions.cpu().numpy()