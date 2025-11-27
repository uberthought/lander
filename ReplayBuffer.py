import os
from typing import List, Tuple
import random
import numpy as np


class ReplayBuffer:
    """Memory-mapped cyclic buffer with automatic disk persistence.

    Stores observations as structured arrays in a memory-mapped file.
    Each observation should be the collection with (state, action, next_state).
    """

    def __init__(self, maxlen: int | None = None, filename: str | None = None, compress: bool = True):
        self.max_episode = 0
        self.maxlen = maxlen or 2**22 # 
        self.filename = filename or "replay_buffer.dat"
        self.compress = compress  # kept for compatibility, not used with memmap
        self.state_shape = (10,)
        
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
            ('state', np.float32, self.state_shape),
            ('action', np.float32),
            ('next_state', np.float32, self.state_shape),
            ('episode', np.float32),
            ('time', np.float32),
            ('done', np.float32)
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
            ('state', np.float32, self.state_shape),
            ('action', np.float32),
            ('next_state', np.float32, self.state_shape),
            ('episode', np.float32),
            ('time', np.float32),
            ('done', np.float32)
        ])
        
        self.buffer = np.memmap(
            self.filename,
            dtype=dtype,
            mode=mode,
            shape=(self.maxlen,)
        )
        # Set cached max episode after loading
        if self.size > 0:
            valid_indices = self._get_valid_indices()
            self.max_episode = int(np.max(self.buffer[valid_indices]['episode']))
        else:
            self.max_episode = 0
    
    def _save_metadata(self):
        """Save buffer metadata (size, write position, state_shape) atomically.
        
        Writes to a temporary file first, then atomically renames it to prevent
        partial reads by concurrent processes.
        """
        temp_file = self.meta_file + ".tmp"
        np.savez(temp_file, 
                 size=self.size, 
                 write_pos=self.write_pos,
                 state_shape=self.state_shape)
        os.replace(temp_file + ".npz", self.meta_file)
    
    def _load_metadata(self):
        """Load buffer metadata."""
        meta = np.load(self.meta_file)
        self.size = int(meta['size'])
        self.write_pos = int(meta['write_pos'])
        self.state_shape = tuple(meta['state_shape'])
        self.max_episode = 0  # will be set after memmap is opened

    
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
        
        observation should be a tuple/namedtuple with (state, action, next_state)
        or have those attributes.
        """
        # Extract fields from observation
        if hasattr(observation, '_fields'):  # namedtuple
            state = observation.state
            action = observation.action
            next_state = observation.next_state
            episode = observation.episode
            time = observation.time
            done = observation.done
        elif isinstance(observation, (tuple, list)):
            state, action, next_state, episode, time, done = observation[:6]
        else:
            raise ValueError("Observation must be tuple/namedtuple with (state, action, next_state, episode, time, done)")

        # Write to buffer
        # Always use cached max episode for new entries
        episode_to_store = self.max_episode
        self.buffer[self.write_pos] = (state, action, next_state, episode_to_store, time, done)

        
        # Update circular buffer pointers
        self.write_pos = (self.write_pos + 1) % self.maxlen
        self.size = min(self.size + 1, self.maxlen)
        
        # Flush to disk periodically (every 100 writes)
        if self.size % 100 == 0:
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
                def __init__(self, state, action, next_state, episode, time, done):
                    self.state = state
                    self.action = action
                    self.next_state = next_state
                    self.episode = episode
                    self.time = time
                    self.done = done
            samples.append(Observation(obs['state'], obs['action'], obs['next_state'], obs['episode'], obs['time'], obs['done']))

        return samples

    def __len__(self):
        return self.size

    def __iter__(self):
        """Iterate over valid observations in the buffer."""
        valid_indices = self._get_valid_indices()
        for i in valid_indices:
            obs = self.buffer[i]
            class Observation:
                def __init__(self, state, action, next_state, episode, time, done):
                    self.state = state
                    self.action = action
                    self.next_state = next_state
                    self.episode = episode
                    self.time = time
                    self.done = done
            yield Observation(obs['state'], obs['action'], obs['next_state'], obs['episode'], obs['time'], obs['done'])

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
        """Return (states, actions, next_states, episodes, times, dones) as numpy arrays."""
        if self.size == 0:
            return (
                np.empty((0,) + self.state_shape, dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,) + self.state_shape, dtype=np.float32),
                np.empty((0,), dtype=np.int32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.bool_),
            )
        
        valid_indices = self._get_valid_indices()
        valid_data = self.buffer[valid_indices]

        states = valid_data['state']
        actions = valid_data['action']
        next_states = valid_data['next_state']
        episodes = valid_data['episode']
        times = valid_data['time']
        dones = valid_data['done']
    
        return states, actions, next_states, episodes, times, dones

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
