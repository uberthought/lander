import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import tempfile

from configuration import POSSIBLE_ACTIONS, LAYER_COUNT, NODE_COUNT, WORLD_LOSS_DELTA, BOOL_LOSS_WEIGHT, STATE_SIZE
from observation import clip_state


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
# output is the predicted next-state delta
class WorldNet(nn.Module):
    def __init__(self, action_dim, nodes, layers):
        super().__init__()
        self.layers = layers
        self.sensors_input = nn.Linear(STATE_SIZE, nodes)
        self.action_input = nn.Linear(action_dim, nodes)
        self.skip_layers = nn.ModuleList([SkipBlock(nodes) for _ in range(layers)])
        self.output = nn.Linear(nodes, STATE_SIZE)

    def forward(self, state, action):
        sensors_embed = self.sensors_input(state)
        action_embed = self.action_input(action)
        x = sensors_embed + action_embed
        for i in range(self.layers):
            x = self.skip_layers[i](x)
        x = self.output(x)
        return x
        

class WorldModel:
    def __init__(self, model_path="checkpoints/world_model.pt"):
        self.model_path = model_path

        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        self.model = WorldNet(POSSIBLE_ACTIONS, NODE_COUNT, LAYER_COUNT)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001)
        # Output is interpreted as: [:, :6] continuous next-state delta, [:, 6:13] logits for boolean dims.
        self.cont_criterion = nn.HuberLoss(delta=WORLD_LOSS_DELTA)
        self.bool_criterion = nn.BCEWithLogitsLoss()

        if os.path.exists(self.model_path):
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
        self.model.train()
        actions = torch.tensor(np.array([int(obs.action) for obs in observations]), dtype=torch.long, device=self.device)
        states_0_full = torch.tensor(np.array([obs.prev_state for obs in observations]), dtype=torch.float32, device=self.device)
        states_1_full = torch.tensor(np.array([obs.next_state for obs in observations]), dtype=torch.float32, device=self.device)

        # Terminal/contact transitions produce unpredictable va spikes from collision
        # impulses the model can't see — filter them out for the continuous head only.
        # The bool head specifically needs these transitions to learn flips.
        leg_changed = (states_0_full[:, 6:8] != states_1_full[:, 6:8]).any(dim=1)
        done_transition = states_1_full[:, 8] > 0.5
        cont_mask = ~(leg_changed | done_transition)

        states_0_clipped = clip_state(states_0_full)
        states_1_clipped = clip_state(states_1_full)
        actions_hot = F.one_hot(actions, num_classes=POSSIBLE_ACTIONS).float()

        self.optimizer.zero_grad()
        prediction = self.model(states_0_clipped, actions_hot)

        bool_target = states_1_full[:, 6:13]
        bool_loss = self.bool_criterion(prediction[:, 6:13], bool_target)

        if cont_mask.any():
            cont_delta = (states_1_clipped[cont_mask, :6] - states_0_clipped[cont_mask, :6])
            cont_loss = self.cont_criterion(prediction[cont_mask, :6], cont_delta)
            loss = cont_loss + BOOL_LOSS_WEIGHT * bool_loss
        else:
            loss = BOOL_LOSS_WEIGHT * bool_loss

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
            states_np = clip_state(np.array(states, dtype=np.float32))
            state_tensor = torch.tensor(states_np, dtype=torch.float32, device=self.device)
            actions_tensor = torch.tensor(np.array(actions), dtype=torch.long, device=self.device)
            actions_hot = F.one_hot(actions_tensor, num_classes=POSSIBLE_ACTIONS).float()
            output = self.model(state_tensor, actions_hot)
            next_cont = state_tensor[:, :6] + output[:, :6]
            next_bool = torch.sigmoid(output[:, 6:13])
            return torch.cat([next_cont, next_bool], dim=-1).cpu().numpy()

    def predict(self, state, action):
        return self.predict_batch([state], [action])[0]