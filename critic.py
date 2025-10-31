from tensorflow.keras.models import Model, load_model  # type: ignore
from tensorflow.keras.layers import Dense, Input  # type: ignore
from tensorflow.keras.optimizers import legacy as legacy_optimizers  # type: ignore
import tensorflow as tf
import numpy as np
import os
import tempfile

from observation import calculate_value


class CriticModel:
    def __init__(self, discount_factor=0.95, model_path="models/critic_model.h5"):
        self.model_path = model_path
        self.num_actions = 4
        # input is state values (translation, rotation, deltas, action one-hot)
        self.input_dim = 8
        # discount factor for future rewards
        self.discount_factor = discount_factor
        self.nodes = self.input_dim * 16
        self.layers = 1

        self.model = self._load_model() or self._create_model()

    def set_actor_model(self, actor_model):
        self.actor_model = actor_model

    def _create_model(self):
        """Creates a new critic model."""

        input_state = Input(shape=(self.input_dim,))
        input_action = Input(shape=(self.num_actions,))

        x = tf.concat([input_state, input_action], axis=-1)
        x = Dense(self.nodes)(x)
        for _ in range(self.layers):
            skip = x
            x = Dense(self.nodes, activation='leaky_relu')(x)
            x = Dense(self.nodes, activation='leaky_relu')(x)
            x = Dense(self.nodes)(x)
            x = x + skip
        x = Dense(1, activation='sigmoid')(x)
        output = x

        model = Model(inputs=[input_state, input_action], outputs=output)
        optimizer = legacy_optimizers.Adam()
        model.compile(optimizer=optimizer, loss='mse')
        return model


    def _load_model(self):
        """Loads the critic model from the specified path."""
        if os.path.exists(self.model_path):
            return load_model(self.model_path)
        return None


    def train(self, observations):
        """Trains the critic model."""
        
        # extract the actions, dones, states, and sensors from observations
        actions = tf.convert_to_tensor([obs.action for obs in observations], dtype=np.float32)
        dones = tf.convert_to_tensor([obs.next_state[-1] for obs in observations], dtype=np.float32)
        states_0 = tf.convert_to_tensor([obs.state for obs in observations], dtype=tf.float32)
        states_1 = tf.convert_to_tensor([obs.next_state for obs in observations], dtype=tf.float32)
        sensors_0 = states_0[:, :8]
        sensors_1 = states_1[:, :8]

        # one hot encode the actions
        actions_hot = tf.one_hot(actions.numpy(), self.num_actions)

        # calculate the values for the next states
        values_1 = calculate_value(states_1)
        values_1 = tf.reshape(values_1, (-1, 1))

        # get the predicted actions for the next states
        p_actions = self.actor_model.model.predict(sensors_1, batch_size=2**15, verbose=0)

        # convert predicted actions to one hot
        p_actions_hot = tf.one_hot(tf.argmax(p_actions, axis=1), self.num_actions)

        # get the predicted values for the next state-action pairs
        p_values_2 = self.model.predict([sensors_1, p_actions_hot], batch_size=2**15, verbose=0)
        p_values_2 = tf.reshape(p_values_2, (-1, 1))

        # compute the target values
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        dones = tf.reshape(dones, (-1, 1))
        not_dones = 1 - dones

        current_values = (1 - self.discount_factor) * values_1
        future_values = self.discount_factor * p_values_2

        values = (current_values + future_values) * not_dones + values_1 * dones

        indices = tf.where(tf.equal(dones, 1.0))
        values = tf.tensor_scatter_nd_update(values, indices, tf.gather_nd(values_1, indices))

        self.model.fit([sensors_0, actions_hot], values, batch_size=2**14, epochs=4, verbose=0)


    def save(self):
        # atomic save
        fd, tmp_path = tempfile.mkstemp(prefix='.tmp_critic_', suffix='.h5', dir=os.path.dirname(self.model_path) or '.')
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


    # def get_optimal_action(self, obs):
    #     """Gets the optimal action for a given observation."""

    #     sensors = obs[:8]
    #     sensors = tf.convert_to_tensor([sensors], dtype=tf.float32)

    #     prediction0 = self.model.predict(sensors, verbose=0)[0]
    #     prediction0 = tf.convert_to_tensor(prediction0, dtype=tf.float32)

    #     # greedy action selection
    #     action = tf.argmax(prediction0)
    #     action = int(action.numpy())

    #     return action, prediction0