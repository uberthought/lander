STATE_SIZE = 13 # Size of the state vector: 8 from env + done + 4 one-hot prev action
CONTINUOUS_STATE_DIM = 6 # Continuous sensor dims (x, y, vx, vy, angle, vangle)
POSSIBLE_ACTIONS = 4 # Number of possible actions per step

LAYER_COUNT = 32 # Number of layers in the model
NODE_COUNT = 256 # Number of nodes per layer

WORLD_LOSS_DELTA = 1.0 # Huber loss delta for WorldModel
BOOL_LOSS_WEIGHT = 1.0 # Weight of the BCE term vs Huber in WorldModel total loss

DISCOUNT_FACTOR = 0.97 # Actor Q-learning discount factor (gamma)
TAU = 0.05 # Polyak averaging rate for ActorModel.target_model
ACTION_SHARPENING = 100 # Softmax temperature multiplier in get_best_action sharpening trick

# --- Checkpoint paths ---
ACTOR_MODEL_PATH = "checkpoints/actor_model.pt"          # actor trained against real transitions
ACTOR_WORLD_MODEL_PATH = "checkpoints/actor_model_world.pt"  # actor trained against world-model rollouts
WORLD_MODEL_PATH = "checkpoints/world_model.pt"          # world (dynamics) model

# --- Training / buffer sizes ---
VALIDATION_SAMPLE_SIZE = 2 ** 10     # held-out-ish sample drawn for live SNR/metric reporting
RECENT_BUFFER_SIZE = 40000           # deque(maxlen=...) of recent on-policy transitions
REPLAY_SAMPLE_MULTIPLIER = 4         # main-buffer samples drawn per recent sample each train cycle
WORLD_LR = 0.001                     # AdamW learning rate for WorldModel

# --- World-model imaginary rollout ---
DONE_THRESHOLD = 0.9 # Predicted done-flag value above which an imagined episode terminates