"""Unit tests for Behavioral Score Diffusion components."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mbd.data.trajectory_dataset import TrajectoryDataset
from mbd.scorers.behavioral_score import BehavioralScorer, BSDConfig
from mbd.scorers.bandwidth_schedule import BandwidthSchedule


def make_dummy_dataset(n=50, H=10, Nu=2, Nx=4, seed=42):
    """Create a small dummy dataset for testing."""
    rng = jax.random.PRNGKey(seed)
    rng1, rng2, rng3 = jax.random.split(rng, 3)
    controls = jax.random.uniform(rng1, (n, H, Nu), minval=-1.0, maxval=1.0)
    states = jax.random.normal(rng2, (n, H + 1, Nx))
    rewards = jax.random.normal(rng3, (n,))
    return TrajectoryDataset.from_arrays(controls, states, rewards)


class TestTrajectoryDataset:

    def test_creation(self):
        ds = make_dummy_dataset(n=50, H=10, Nu=2, Nx=4)
        assert ds.n_trajectories == 50
        assert ds.horizon == 10
        assert ds.n_controls == 2
        assert ds.n_states == 4
        assert ds.controls.shape == (50, 10, 2)
        assert ds.states.shape == (50, 11, 4)
        assert ds.init_states.shape == (50, 4)
        assert ds.final_states.shape == (50, 4)
        assert ds.controls_flat.shape == (50, 20)

    def test_save_load(self, tmp_path):
        ds = make_dummy_dataset()
        path = str(tmp_path / "test_ds")
        ds.save(path)
        ds2 = TrajectoryDataset.load(path)
        assert ds2.n_trajectories == ds.n_trajectories
        np.testing.assert_allclose(
            np.asarray(ds.controls), np.asarray(ds2.controls), atol=1e-5
        )

    def test_subset(self):
        ds = make_dummy_dataset(n=50)
        idx = jnp.array([0, 5, 10, 15])
        sub = ds.subset(idx)
        assert sub.n_trajectories == 4
        np.testing.assert_allclose(
            np.asarray(sub.controls[0]),
            np.asarray(ds.controls[0]),
            atol=1e-6,
        )

    def test_stats(self):
        ds = make_dummy_dataset()
        stats = ds.stats()
        assert stats['n_trajectories'] == 50
        assert isinstance(stats['reward_mean'], float)


class TestBandwidthSchedule:

    def test_from_diffusion_schedule(self):
        sigmas = jnp.linspace(0.01, 1.0, 100)
        bw = BandwidthSchedule.from_diffusion_schedule(sigmas, nu_ctx=1.5)
        assert bw.sigma_max == pytest.approx(1.0, abs=0.01)
        assert bw.nu_ctx == 1.5

    def test_bandwidth_decreases_with_noise(self):
        bw = BandwidthSchedule(nu_ctx=1.0, nu_goal_0=2.0, gamma=0.5, sigma_max=1.0)
        _, _, nu_goal_high = bw.get_bandwidths(0.9)
        _, _, nu_goal_low = bw.get_bandwidths(0.1)
        assert nu_goal_high > nu_goal_low  # Broader at high noise

    def test_diffusion_bandwidth_tracks_sigma(self):
        bw = BandwidthSchedule(sigma_max=1.0)
        beta_diff, _, _ = bw.get_bandwidths(0.5)
        assert float(beta_diff) == pytest.approx(0.5, abs=1e-5)


class TestBehavioralScorer:

    def test_score_output_shapes(self):
        ds = make_dummy_dataset(n=50, H=10, Nu=2, Nx=4)
        bw = BandwidthSchedule(sigma_max=1.0)
        config = BSDConfig()
        scorer = BehavioralScorer(ds, config, bw)

        Y_t = jnp.zeros((10, 2))
        x_0 = jnp.zeros(4)
        x_goal = jnp.ones(4)
        sigma_t = jnp.array(0.5)

        Y0_hat, X_hat, eff_K, max_w = scorer.compute_score(
            Y_t, x_0, x_goal, sigma_t
        )

        assert Y0_hat.shape == (10, 2)
        assert X_hat.shape == (11, 4)
        assert float(eff_K) > 0
        assert 0 <= float(max_w) <= 1

    def test_jit_compiled_produces_valid_output(self):
        """JIT version should produce valid behavioral scores with correct shapes."""
        ds = make_dummy_dataset(n=50, H=10, Nu=2, Nx=4)
        bw = BandwidthSchedule(sigma_max=1.0)
        config = BSDConfig()
        scorer = BehavioralScorer(ds, config, bw)

        Y_t = jnp.zeros((10, 2))
        x_0 = jnp.zeros(4)
        x_goal = jnp.ones(4)
        sigma_t = jnp.array(0.5)

        compute_jit = scorer.compute_score_jit()
        Y0_jit, X_jit, ek_jit, mw_jit = compute_jit(
            Y_t, x_0, x_goal, sigma_t
        )

        # Correct shapes
        assert Y0_jit.shape == (10, 2)
        assert X_jit.shape == (11, 4)

        # Valid range (convex combination of [-1,1] data)
        assert jnp.all(Y0_jit >= -1.1)  # small tolerance
        assert jnp.all(Y0_jit <= 1.1)

        # Positive effective K
        assert float(ek_jit) > 0

        # Weight in [0,1]
        assert 0 <= float(mw_jit) <= 1

        # Deterministic: running twice gives same result
        Y0_jit2, _, _, _ = compute_jit(Y_t, x_0, x_goal, sigma_t)
        np.testing.assert_allclose(
            np.asarray(Y0_jit), np.asarray(Y0_jit2), atol=1e-6
        )

    def test_weights_sum_to_one(self):
        """Verify that kernel weights are properly normalized."""
        ds = make_dummy_dataset(n=50, H=10, Nu=2, Nx=4)
        bw = BandwidthSchedule(sigma_max=1.0)
        config = BSDConfig()
        scorer = BehavioralScorer(ds, config, bw)

        Y_t = jnp.zeros((10, 2))
        x_0 = jnp.zeros(4)
        x_goal = jnp.ones(4)

        # The estimate should be a convex combination of stored trajectories
        # => it should lie within the range of stored controls
        Y0_hat, _, _, _ = scorer.compute_score(Y_t, x_0, x_goal, jnp.array(0.5))
        assert jnp.all(Y0_hat >= -1.0 - 0.01)  # Small tolerance
        assert jnp.all(Y0_hat <= 1.0 + 0.01)

    def test_high_noise_broad_weights(self):
        """At high noise, effective K should be large (many trajectories contribute)."""
        ds = make_dummy_dataset(n=100, H=10, Nu=2, Nx=4)
        bw = BandwidthSchedule(sigma_max=1.0)
        config = BSDConfig()
        scorer = BehavioralScorer(ds, config, bw)

        Y_t = jnp.zeros((10, 2))
        x_0 = jnp.zeros(4)
        x_goal = jnp.ones(4)

        _, _, eff_K_high, _ = scorer.compute_score(Y_t, x_0, x_goal, jnp.array(0.9))
        _, _, eff_K_low, _ = scorer.compute_score(Y_t, x_0, x_goal, jnp.array(0.05))

        # At high noise, more trajectories should contribute
        assert float(eff_K_high) > float(eff_K_low)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
