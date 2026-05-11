import os
from typing import List, Tuple
import random
import numpy as np

from configuration import STATE_SIZE

class ReplayBuffer:
    """Memory-mapped cyclic buffer with automatic disk persistence.

    Stores observations as structured arrays in a memory-mapped file.
    Each observation should be the collection with (prev_state, action, next_state).
    """

    def __init__(self, maxlen: int | None = None, filename: str | None = None, compress: bool = True):
        self.max_episode = 0
        self.maxlen = maxlen or 2**22 # 
        self.filename = filename or "data/replay_buffer.dat"
        self.compress = compress  # kept for compatibility, not used with memmap
        self.state_shape = (STATE_SIZE,)
        self.action_shape = (1,)
        
        # Create directory if needed
        os.makedirs(os.path.dirname(os.path.abspath(self.filename)) or '.', exist_ok=True)
        
        # Metadata file to track state
        self.meta_file = self.filename + ".meta.npz"
        
        # Initialize or load existing buffer
        if os.path.exists(self.filename) and os.path.exists(self.meta_file):
            self._load_metadata()
            self._open_memmap(mode='r+')
        else:
            self.size = 0
            self.write_pos = 0
            self.max_episode = 0
            if self.state_shape is None:
                raise ValueError("state_shape must be provided for new buffer")
            self._create_memmap()
    
    def _create_memmap(self):
        """Create a new memory-mapped file with the appropriate dtype."""
        # Define structured dtype for observations
        # Adjust state_dim based on your state representation
        state_dim = np.prod(self.state_shape) if isinstance(self.state_shape, tuple) else self.state_shape
        
        dtype = np.dtype([
            ('prev_state', np.float32, self.state_shape),
            ('action', np.float32, self.action_shape),
            ('next_state', np.float32, self.state_shape),
        ])
        
        self.buffer = np.memmap(
            self.filename,
            dtype=dtype,
            mode='w+',
            shape=(self.maxlen,)
        )
        self._save_metadata()
    
    def _open_memmap(self, mode='r+'):
        """Open existing memory-mapped file."""
        dtype = np.dtype([
            ('prev_state', np.float32, self.state_shape),
            ('action', np.float32, self.action_shape),
            ('next_state', np.float32, self.state_shape),
        ])
        
        self.buffer = np.memmap(
            self.filename,
            dtype=dtype,
            mode=mode,
            shape=(self.maxlen,)
        )
        pass
    
    def _save_metadata(self):
        """Save buffer metadata (size, write position, state_shape) atomically.
        
        Writes to a temporary file first, then atomically renames it to prevent
        partial reads by concurrent processes.
        """
        temp_file = self.meta_file + ".tmp"
        np.savez(temp_file,
                 size=self.size,
                 write_pos=self.write_pos,
                 state_shape=self.state_shape,
                 max_episode=self.max_episode)
        os.replace(temp_file + ".npz", self.meta_file)
    
    def _load_metadata(self):
        """Load buffer metadata."""
        meta = np.load(self.meta_file)
        self.size = int(meta['size'])
        self.write_pos = int(meta['write_pos'])
        self.state_shape = tuple(meta['state_shape'])
        self.max_episode = int(meta['max_episode']) if 'max_episode' in meta else 0

    
    def _get_valid_indices(self):
        """Get indices of valid data accounting for circular wrap.
        
        Returns array of indices where valid data is stored.
        """
        if self.size < self.maxlen:
            # Buffer not full yet, valid data is at [0, size)
            return np.arange(self.size)
        else:
            # Buffer is full, oldest data starts at write_pos
            return np.arange(self.write_pos, self.write_pos + self.maxlen) % self.maxlen

    # --- core API ---
    def add(self, observation):
        """Add an observation to the buffer.
        
        observation should be a tuple/namedtuple with (prev_state, action, next_state)
        or have those attributes.
        """
        # Extract fields from observation
        if hasattr(observation, '_fields'):  # namedtuple
            prev_state = observation.prev_state
            action = observation.action
            next_state = observation.next_state
        elif isinstance(observation, (tuple, list)):
            prev_state, action, next_state = observation[:3]
        else:
            raise ValueError("Observation must be tuple/namedtuple with (prev_state, action, next_state)")

        self.buffer[self.write_pos] = (prev_state, action, next_state)

        
        # Update circular buffer pointers
        self.write_pos = (self.write_pos + 1) % self.maxlen
        self.size = min(self.size + 1, self.maxlen)
        
        # Always flush and save metadata after every write to ensure write_pos and size are accurate
        self.buffer.flush()
        self._save_metadata()

    # alias for compatibility with existing code that used a raw deque
    append = add

    def extend(self, observations):
        for o in observations:
            self.add(o)

    def sample(self, k: int) -> List:
        """Sample k random observations from the buffer."""
        k = min(k, self.size)
        if k == 0:
            return []
        
        valid_indices = self._get_valid_indices()
        sampled_indices = np.random.choice(valid_indices, size=k, replace=False)
        
        # Return as list of tuples for compatibility
        samples = []
        for i in sampled_indices:
            obs = self.buffer[i]
            # Create a simple object to mimic namedtuple behavior
            class Observation:
                def __init__(self, prev_state, action, next_state):
                    self.prev_state = prev_state
                    self.action = action
                    self.next_state = next_state
            samples.append(Observation(obs['prev_state'], obs['action'], obs['next_state']))

        return samples

    def sample_weighted(self, k: int, alpha: float = 1.0) -> List:
        """Sample k observations with replacement, weighted toward the most recent.

        Weight for an item with recency rank r (0 = most recent) is 1 / (r + 1)**alpha.
        """
        if k == 0 or self.size == 0:
            return []

        valid_indices = self._get_valid_indices()  # ordered oldest -> newest
        n = len(valid_indices)
        ranks = np.arange(n - 1, -1, -1, dtype=np.float64)  # newest gets rank 0
        weights = 1.0 / (ranks + 1.0) ** alpha
        weights /= weights.sum()

        sampled_indices = np.random.choice(valid_indices, size=k, replace=True, p=weights)

        samples = []
        for i in sampled_indices:
            obs = self.buffer[i]
            class Observation:
                def __init__(self, prev_state, action, next_state):
                    self.prev_state = prev_state
                    self.action = action
                    self.next_state = next_state
            samples.append(Observation(obs['prev_state'], obs['action'], obs['next_state']))

        return samples

    def __len__(self):
        return self.size

    def __iter__(self):
        """Iterate over valid observations in the buffer."""
        valid_indices = self._get_valid_indices()
        for i in valid_indices:
            obs = self.buffer[i]
            class Observation:
                def __init__(self, prev_state, action, next_state):
                    self.prev_state = prev_state
                    self.action = action
                    self.next_state = next_state
            yield Observation(obs['prev_state'], obs['action'], obs['next_state'])

    # --- persistence ---
    def save(self, path: str | None = None):
        """Flush buffer to disk (data is already memory-mapped).
        
        This method is kept for API compatibility but simply ensures
        data is synced to disk.
        """
        self.buffer.flush()
        self._save_metadata()

    def load(self, path: str | None = None):
        """Load is automatic with memmap - this is a no-op for compatibility."""
        pass

    # --- utilities ---
    def to_arrays(self):
        """Return (prev_states, actions, next_states) as numpy arrays."""
        if self.size == 0:
            return (
                np.empty((0,) + self.state_shape, dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,) + self.state_shape, dtype=np.float32),
            )

        valid_indices = self._get_valid_indices()
        valid_data = self.buffer[valid_indices]

        return valid_data['prev_state'], valid_data['action'], valid_data['next_state']

    def shrink(self, keep_last: int):
        """Keep only the most recent N observations (in-place)."""
        if keep_last >= self.size:
            return
        
        # Get the most recent keep_last indices from valid data
        valid_indices = self._get_valid_indices()
        recent_indices = valid_indices[-keep_last:]
        
        # Copy recent data to beginning of buffer
        self.buffer[:keep_last] = self.buffer[recent_indices]
        
        self.size = keep_last
        self.write_pos = keep_last % self.maxlen
        self.buffer.flush()
        self._save_metadata()

    def maybe_autosave(self, every: int, episode: int):
        """Autosave every N episodes (call from training loop)."""
        if every > 0 and episode % every == 0:
            try:
                self.save()
            except Exception as e:
                print(f"ReplayBuffer autosave failed: {e}")
    def increment_episode(self):
        """Increment the cached maximum episode number."""
        self.max_episode += 1

    def close(self):
        """Explicitly close and flush the memory-mapped file."""
        if hasattr(self, 'buffer'):
            self.buffer.flush()
            self._save_metadata()
            del self.buffer
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()
