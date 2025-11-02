from tensorflow.keras.models import Model, load_model  # type: ignore
from tensorflow.keras.layers import Dense, Input  # type: ignore
from tensorflow.keras.optimizers import legacy as legacy_optimizers  # type: ignore
import tensorflow as tf
import numpy as np
import os
import tempfile

from observation import calculate_value


class ActorModel:
    def __init__(self, model_path="models/actor_model.h5"):
        self.model_path = model_path
        # input is state values (translation, rotation, deltas, fuel)
        self.input_dim = 10
        # output is 4 values (action probabilities for each action)
        self.num_actions = 4
        self.nodes = self.input_dim * 16
        self.layers = 1

        self.model = self._load_model() or self._create_model()

    def set_critic_model(self, critic_model):
        self.critic_model = critic_model

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
        x = Dense(self.num_actions, activation='softmax')(x)
        output = x

        model = Model(inputs=input, outputs=output)
        optimizer = legacy_optimizers.Adam()
        model.compile(optimizer=optimizer, loss='categorical_crossentropy')
        return model


    def _load_model(self):
        """Loads the actor model from the specified path."""
        if os.path.exists(self.model_path):
            return load_model(self.model_path)
        return None


    def train(self, observations):
        """Trains the actor model."""

        states_0 = tf.convert_to_tensor([obs.state for obs in observations], dtype=tf.float32)
        states_1 = tf.convert_to_tensor([obs.next_state for obs in observations], dtype=tf.float32)
        sensors_1_tiled = tf.repeat(states_1, self.num_actions, axis=0)  # shape: (num_obs * OUTPUT_DIM, sensor_dim)

        actions_onehot_tiled = tf.one_hot(tf.tile(tf.range(self.num_actions), [len(observations)]), self.num_actions) # shape: (num_obs * OUTPUT_DIM, OUTPUT_DIM)

        predicted_rewards = self.critic_model.model.predict([sensors_1_tiled, actions_onehot_tiled], batch_size=2**14, verbose=0)
        predicted_rewards = tf.reshape(predicted_rewards, (len(observations), self.num_actions))

        predicted_actions = tf.argmax(predicted_rewards, axis=1)
        best_actions = tf.one_hot(predicted_actions, self.num_actions)

        self.model.fit(states_0, best_actions, batch_size=2**15, epochs=4, verbose=0)


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

        sensors = tf.convert_to_tensor([obs], dtype=tf.float32)

        prediction0 = self.model.predict(sensors, verbose=0)[0]
        prediction0 = tf.convert_to_tensor(prediction0, dtype=tf.float32)

        # greedy action selection
        action = tf.argmax(prediction0)
        action = int(action.numpy())

        return action, prediction0