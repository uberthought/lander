STATE_SIZE = 11 # Size of the state vector: 8 from env + fuel + sin(angle) + cos(angle)
POSSIBLE_ACTIONS = 4 # Number of possible actions per step

LAYER_COUNT = 8 # Number of layers in the model
NODE_COUNT = 64 # Number of nodes per layer
WORLD_LOSS_DELTA = 1.0 # Huber loss transition point for world-model regression