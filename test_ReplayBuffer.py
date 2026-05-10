import os
import numpy as np
import tempfile
from ReplayBuffer import ReplayBuffer
from configuration import STATE_SIZE

def make_obs(state, actions, next_state, time, done):
    # Use dummy values for episode, will be overwritten by buffer
    return (state, actions, next_state, 0.0, time, done)

def test_save_and_load():
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, 'test_replay_buffer.dat')
        buf = ReplayBuffer(maxlen=10, filename=filename)
        buf.increment_episode()
        state = np.zeros((STATE_SIZE,), dtype=np.float32)
        next_state = np.ones((STATE_SIZE,), dtype=np.float32)
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
        buf2.close()

def test_increment_episode():
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, 'test_replay_buffer2.dat')
        buf = ReplayBuffer(maxlen=10, filename=filename)
        assert buf.max_episode == 0
        buf.increment_episode()
        assert buf.max_episode == 1
        buf.increment_episode()
        assert buf.max_episode == 2
        buf.close()

def test_sample2_sequential():
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, 'test_replay_buffer3.dat')
        buf = ReplayBuffer(maxlen=20, filename=filename)
        buf.increment_episode()
        state = np.zeros((STATE_SIZE,), dtype=np.float32)
        next_state = np.ones((STATE_SIZE,), dtype=np.float32)

        # Add 15 observations with sequential time values
        for t in range(15):
            buf.add(make_obs(state + t, float(t), next_state + t, float(t), float(t % 2)))

        # make sure we have enough data
        assert len(buf) == 15, f"Expected buffer length 15, got {len(buf)}"

        # Sample a sequence of 5 observations
        samples = buf.sample2(5)
        assert len(samples) == 5, f"Expected 5 samples, got {len(samples)}"

        # Check that action values are sequential
        for i in range(1, len(samples)):
            assert samples[i].action == samples[i-1].action + 1, \
                f"Actions should be sequential: {samples[i-1].action} -> {samples[i].action}"

        # Also verify states are sequential
        for i in range(1, len(samples)):
            expected_diff = samples[i].prev_state - samples[i-1].prev_state
            assert np.allclose(expected_diff, np.ones(STATE_SIZE, dtype=np.float32)), \
                f"States should be sequential with diff of 1"
        buf.close()

if __name__ == "__main__":
    test_save_and_load()
    test_increment_episode()
    test_sample2_sequential()
    print("All ReplayBuffer tests passed.")
