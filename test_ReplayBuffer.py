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

def test_sample_random():
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

        # Sample 5 random observations (without replacement)
        samples = buf.sample(5)
        assert len(samples) == 5, f"Expected 5 samples, got {len(samples)}"

        # Sampling is without replacement: all actions distinct
        actions = [float(s.action) for s in samples]
        assert len(set(actions)) == len(actions), \
            f"Expected no duplicate actions (sample without replacement), got {actions}"

        # Each action must come from the populated range [0, 14]
        for a in actions:
            assert 0.0 <= a <= 14.0, f"Action {a} out of populated range"

        # Per make_obs, prev_state was set to (state + t) and action to float(t),
        # so prev_state[0] must equal action for every sample — confirms fields aren't scrambled.
        for s in samples:
            assert np.isclose(s.prev_state[0], float(s.action)), \
                f"Field mismatch: prev_state[0]={s.prev_state[0]} action={s.action}"
        buf.close()

def test_sample_weighted_recency_bias():
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, 'test_replay_buffer4.dat')
        buf = ReplayBuffer(maxlen=100, filename=filename)
        buf.increment_episode()
        state = np.zeros((STATE_SIZE,), dtype=np.float32)
        next_state = np.ones((STATE_SIZE,), dtype=np.float32)

        n = 100
        for t in range(n):
            buf.add(make_obs(state + t, float(t), next_state + t, float(t), float(t % 2)))

        np.random.seed(0)
        samples = buf.sample_weighted(10000, alpha=1.0)
        assert len(samples) == 10000

        actions = np.array([float(s.action) for s in samples])
        # Most recent half (action >= 50) should dominate under 1/(rank+1) weighting.
        recent_frac = (actions >= 50).mean()
        assert recent_frac > 0.7, f"Expected recency bias; recent_frac={recent_frac}"

        # Field integrity: prev_state[0] must equal action for every sample.
        for s in samples[:50]:
            assert np.isclose(s.prev_state[0], float(s.action)), \
                f"Field mismatch: prev_state[0]={s.prev_state[0]} action={s.action}"
        buf.close()

if __name__ == "__main__":
    test_save_and_load()
    test_increment_episode()
    test_sample_random()
    test_sample_weighted_recency_bias()
    print("All ReplayBuffer tests passed.")
