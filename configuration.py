STATE_SIZE = 9 # Size of the state vector: 8 from env + done
POSSIBLE_ACTIONS = 4 # Number of possible actions per step

LAYER_COUNT = 32 # Number of layers in the model
NODE_COUNT = 64 # Number of nodes per layer
WORLD_LOSS_DELTA = 1.0 # Huber loss transition point for world-model regression