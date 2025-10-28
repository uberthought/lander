from tensorflow.keras.models import Model, load_model  # type: ignore
from tensorflow.keras.layers import Dense, Input  # type: ignore
from tensorflow.keras.optimizers import legacy as legacy_optimizers  # type: ignore
import tensorflow as tf
import numpy as np
import os
import tempfile

from observation import calculate_value


class ActorModel:
    def __init__(self, discount_factor=0.95, model_path="models/actor_model.h5"):
        self.model_path = model_path
        # input is state values (translation, rotation, deltas)
        self.input_dim = 8
        # output is 4 values (action probabilities for each action)
        self.num_actions = 4
        # discount factor for future rewards
        self.discount_factor = discount_factor
        self.nodes = self.input_dim * 16
        self.layers = 1

        self.model = self._load_model() or self._create_model()

    def _create_model(self):
        """Creates a new actor model."""

        input = Input(shape=(self.input_dim,))

        x = input
        x = Dense(self.nodes)(x)
        for _ in range(self.layers):
            skip = x
            x = Dense(self.nodes, activation='leaky_relu')(x)
            x = Dense(self.nodes, activation='leaky_relu')(x)
            x = Dense(self.nodes)(x)
            x = x + skip
        x = Dense(self.num_actions, activation='sigmoid')(x)
        output = x

        model = Model(inputs=input, outputs=output)
        optimizer = legacy_optimizers.Adam()
        model.compile(optimizer=optimizer, loss='mse')
        return model


    def _load_model(self):
        """Loads the actor model from the specified path."""
        if os.path.exists(self.model_path):
            return load_model(self.model_path)
        return None


    def train(self, observations):
        """Trains the actor model."""
        actions_1 = tf.convert_to_tensor([obs.action for obs in observations], dtype=np.float32)
        dones_1 = tf.convert_to_tensor([obs.next_state[-1] for obs in observations], dtype=np.float32)

        states_0 = tf.convert_to_tensor([obs.state for obs in observations], dtype=tf.float32)
        states_1 = tf.convert_to_tensor([obs.next_state for obs in observations], dtype=tf.float32)
        values_1 = calculate_value(states_1)
        values_1 = tf.reshape(values_1, (-1, 1))
        sensors_0 = states_0[:, :8]
        sensors_1 = states_1[:, :8]

        p_rewards_0 = self.model.predict(sensors_0, batch_size=2**14, verbose=0)
        p_rewards_1 = self.model.predict(sensors_1, batch_size=2**14, verbose=0)

        # Vectorized Q-value update
        q_values_1 = tf.reduce_max(p_rewards_1, axis=1)
        q_values_1 = tf.reshape(q_values_1, (-1, 1))

        dones = tf.convert_to_tensor(dones_1, dtype=tf.float32)
        dones = tf.reshape(dones, (-1, 1))
        not_dones = 1 - dones

        current_values = (1 - self.discount_factor) * values_1
        future_values = self.discount_factor * q_values_1

        values = (current_values + future_values) * not_dones + values_1 * dones

        indices = tf.stack([tf.range(tf.shape(actions_1)[0], dtype=tf.int32), 
                    tf.cast(actions_1, tf.int32)], axis=1)
        p_rewards_0 = tf.tensor_scatter_nd_update(p_rewards_0, indices, tf.squeeze(values))

        self.model.fit(sensors_0, p_rewards_0, batch_size=2**14, epochs=4, verbose=0)


    def save(self):
        # atomic save
        fd, tmp_path = tempfile.mkstemp(prefix='.tmp_actor_', suffix='.h5', dir=os.path.dirname(self.model_path) or '.')
        os.close(fd)
        try:
            self.model.save(tmp_path)
            os.replace(tmp_path, self.model_path)  # atomic on POSIX
        except KeyboardInterrupt:
            # If interrupted during save, leave old file untouched
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise


    def get_optimal_action(self, obs):
        """Gets the optimal action for a given observation."""

        sensors = obs[:8]
        sensors = tf.convert_to_tensor([sensors], dtype=tf.float32)

        prediction0 = self.model.predict(sensors, verbose=0)[0]
        prediction0 = tf.convert_to_tensor(prediction0, dtype=tf.float32)

        # greedy action selection
        action = tf.argmax(prediction0)
        action = int(action.numpy())

        return action, prediction0