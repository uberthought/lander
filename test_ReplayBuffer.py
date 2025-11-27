import os
import numpy as np
import tempfile
import shutil
from ReplayBuffer import ReplayBuffer

def make_obs(state, action, next_state, time, done):
    # Use dummy values for episode, will be overwritten by buffer
    return (state, action, next_state, 0.0, time, done)

def test_save_and_load():
    filename = 'test_replay_buffer.dat'
    buf = ReplayBuffer(maxlen=10, filename=filename)
    buf.increment_episode()
    state = np.zeros((10,), dtype=np.float32)
    next_state = np.ones((10,), dtype=np.float32)
    for t in range(5):
        buf.add(make_obs(state + t, float(t), next_state + t, float(t), float(t % 2)))
    max_ep = buf.max_episode
    assert max_ep == 1, f"Expected max_episode 1, got {max_ep}"
    buf.save()
    del buf
    # Reload
    buf2 = ReplayBuffer(maxlen=10, filename=filename)
    assert buf2.max_episode == 1, f"Reloaded max_episode should be 1, got {buf2.max_episode}"
    arrs = buf2.to_arrays()
    assert arrs[0].shape[0] == 5, f"Expected 5 states, got {arrs[0].shape[0]}"
    assert np.allclose(arrs[0][0], state), "State mismatch after reload"
    assert np.allclose(arrs[2][0], next_state), "Next state mismatch after reload"

    # remove test files
    os.remove(filename)
    os.remove(filename + ".meta.npz")

def test_increment_episode():
    filename = 'test_replay_buffer2.dat'
    buf = ReplayBuffer(maxlen=10, filename=filename)
    assert buf.max_episode == 0
    buf.increment_episode()
    assert buf.max_episode == 1
    buf.increment_episode()
    assert buf.max_episode == 2

if __name__ == "__main__":
    test_save_and_load()
    test_increment_episode()
    print("All ReplayBuffer tests passed.")
