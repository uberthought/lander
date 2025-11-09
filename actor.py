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
        
        self.input = nn.Linear(self.input_dim, nodes)

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
            nn.Sigmoid()
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
        self.input_dim = 10
        self.num_actions = 4
        self.nodes = self.input_dim * self.num_actions * 4
        self.layers = 2

        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = self._load_model() or self._create_model()
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.MSELoss()

    def _create_model(self):
        return ActorNet(self.input_dim, self.num_actions, self.nodes, self.layers)

    def _load_model(self):
        if os.path.exists(self.model_path):
            model = ActorNet(self.input_dim, self.num_actions, self.nodes, self.layers)
            model.load_state_dict(torch.load(self.model_path, map_location="cpu"))
            return model
        return None



    # def train(self, observations):
    #     """Trains the actor model."""

    #     states_1 = tf.convert_to_tensor([obs.next_state for obs in observations], dtype=tf.float32)
    #     sensors_1 = states_1[:, :8]
    #     p_rewards_1 = self.model.predict(sensors_1, batch_size=2**14, verbose=0)

    #     # Vectorized Q-value update
    #     q_values_1 = tf.reduce_max(p_rewards_1, axis=1)
    #     q_values_1 = tf.reshape(q_values_1, (-1, 1))

    #     values_1 = calculate_value(states_1)
    #     values_1 = tf.reshape(values_1, (-1, 1))
    #     current_values = (1 - self.discount_factor) * values_1
    #     future_values = self.discount_factor * q_values_1

    #     dones_1 = tf.convert_to_tensor([obs.next_state[-1] for obs in observations], dtype=np.float32)
    #     dones = tf.convert_to_tensor(dones_1, dtype=tf.float32)
    #     dones = tf.reshape(dones, (-1, 1))
    #     not_dones = 1 - dones
    #     values = (current_values + future_values) * not_dones + values_1 * dones

    #     states_0 = tf.convert_to_tensor([obs.state for obs in observations], dtype=tf.float32)
    #     sensors_0 = states_0[:, :8]
    #     p_rewards_0 = self.model.predict(sensors_0, batch_size=2**14, verbose=0)
    #     actions_1 = tf.convert_to_tensor([obs.action for obs in observations], dtype=np.float32)
    #     indices = tf.stack([tf.range(tf.shape(actions_1)[0], dtype=tf.int32), 
    #                 tf.cast(actions_1, tf.int32)], axis=1)
    #     p_rewards_0 = tf.tensor_scatter_nd_update(p_rewards_0, indices, tf.squeeze(values))

    #     self.model.fit(sensors_0, p_rewards_0, batch_size=2**14, epochs=4, verbose=0)


    def train(self, observations):
        self.model.train()
        states_1 = torch.tensor(np.array([obs.next_state for obs in observations]), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            p_rewards_1 = self.model(states_1)

        # Vectorized Q-value update
        q_values_1, _ = torch.max(p_rewards_1, dim=1)
        q_values_1 = q_values_1.view(-1, 1)

        values_1 = calculate_value(states_1)
        values_1 = values_1.view(-1, 1)
        current_values = (1 - self.discount_factor) * values_1
        future_values = self.discount_factor * q_values_1

        dones_1 = torch.tensor([obs.next_state[-1] for obs in observations], dtype=torch.float32, device=self.device)
        dones = dones_1.view(-1, 1)
        not_dones = 1 - dones
        values = (current_values + future_values) * not_dones + values_1 * dones

        states_0 = torch.tensor(np.array([obs.state for obs in observations]), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            p_rewards_0 = self.model(states_0)
        actions_1 = torch.tensor([int(obs.action) for obs in observations], dtype=torch.long, device=self.device)
        indices = torch.stack([torch.arange(len(actions_1), device=self.device), actions_1], dim=1)
        p_rewards_0 = p_rewards_0.clone()
        p_rewards_0[indices[:,0], indices[:,1]] = values.squeeze()

        for _ in range(4):
            self.optimizer.zero_grad()
            outputs = self.model(states_0)
            loss = self.criterion(outputs, p_rewards_0)
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
            action_values = self.model(sensors)[0]
        # prediction_np = prediction0.cpu().numpy()

        # greedy action selection
        action1 = torch.argmax(action_values).item()

        # stochastic action selection
        action2 = torch.multinomial(action_values, num_samples=1).item()

        # 5% chance to explore
        action = action2 if np.random.rand() < 0.05 else action1
        
        return int(action), action_values.cpu().numpy()
