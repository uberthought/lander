STATE_SIZE = 11 # Size of the state vector: 8 from env + fuel + sin(angle) + cos(angle)
POSSIBLE_ACTIONS = 4 # Number of possible actions per step

NUM_ACTIONS = 1 # Total number of actions in each observation

LAYER_COUNT = 4 # Number of layers in the model
NODE_COUNT = 128 # Number of nodes per layer
WORLD_LOSS_DELTA = 1.0 # Huber loss transition point for world-model regression
WORLD_LOSS_WEIGHTS = (1.0, 1.0, 16.0, 2.0, 1.0, 4.0, 1.0, 1.0, 1.0, 1.0, 1.0) # vx raised to compensate norm_factor=4; vy and vangle retain original emphasis