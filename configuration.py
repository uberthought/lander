STATE_SIZE = 13 # Size of the state vector: 8 from env + done + 4 one-hot prev action
CONTINUOUS_STATE_DIM = 6 # Continuous sensor dims (x, y, vx, vy, angle, vangle)
POSSIBLE_ACTIONS = 4 # Number of possible actions per step

LAYER_COUNT = 16 # Number of layers in the model
NODE_COUNT = 64 # Number of nodes per layer

WORLD_LOSS_DELTA = 1.0 # Huber loss delta for WorldModel

DISCOUNT_FACTOR = 0.997 # Actor Q-learning discount factor (gamma)
TAU = 0.05 # Polyak averaging rate for ActorModel.target_model
ACTION_SHARPENING = 100 # Softmax temperature multiplier in get_best_action sharpening trick