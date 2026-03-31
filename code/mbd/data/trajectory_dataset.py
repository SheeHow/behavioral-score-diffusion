"""
Trajectory dataset storage and retrieval for Behavioral Score Diffusion.

Stores (control, state) trajectory pairs collected from system rollouts.
Provides efficient retrieval via flattened control vectors for kernel scoring.
"""

import jax
import jax.numpy as jnp
import numpy as np
import os
import logging
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrajectoryDataset:
    """Stores trajectory data for kernel-based behavioral score estimation.

    All arrays are JAX arrays for GPU-accelerated kernel computation.

    Attributes:
        controls: [N, H, Nu] control sequences (normalized to [-1, 1])
        states: [N, H+1, Nx] state sequences (including initial state at index 0)
        init_states: [N, Nx] initial states of each trajectory
        final_states: [N, Nx] final states of each trajectory
        rewards: [N] total reward for each trajectory
        controls_flat: [N, H*Nu] flattened controls for distance computation
        n_trajectories: int number of stored trajectories
        horizon: int planning horizon H
        n_controls: int control dimension Nu
        n_states: int state dimension Nx
    """
    controls: jnp.ndarray       # [N, H, Nu]
    states: jnp.ndarray         # [N, H+1, Nx]
    init_states: jnp.ndarray    # [N, Nx]
    final_states: jnp.ndarray   # [N, Nx]
    rewards: jnp.ndarray        # [N]
    controls_flat: jnp.ndarray  # [N, H*Nu]
    n_trajectories: int
    horizon: int
    n_controls: int
    n_states: int

    @classmethod
    def from_arrays(cls, controls: jnp.ndarray, states: jnp.ndarray,
                    rewards: Optional[jnp.ndarray] = None):
        """Create dataset from control and state arrays.

        Args:
            controls: [N, H, Nu] control sequences
            states: [N, H+1, Nx] state sequences (first entry is initial state)
            rewards: [N] optional rewards (zeros if not provided)
        """
        N, H, Nu = controls.shape
        Nx = states.shape[-1]

        if rewards is None:
            rewards = jnp.zeros(N)

        controls = jnp.asarray(controls)
        states = jnp.asarray(states)
        rewards = jnp.asarray(rewards)

        init_states = states[:, 0, :]     # [N, Nx]
        final_states = states[:, -1, :]   # [N, Nx]
        controls_flat = controls.reshape(N, -1)  # [N, H*Nu]

        logging.info(
            f"TrajectoryDataset created: N={N}, H={H}, Nu={Nu}, Nx={Nx}"
        )

        return cls(
            controls=controls,
            states=states,
            init_states=init_states,
            final_states=final_states,
            rewards=rewards,
            controls_flat=controls_flat,
            n_trajectories=N,
            horizon=H,
            n_controls=Nu,
            n_states=Nx,
        )

    def save(self, path: str):
        """Save dataset to disk as numpy arrays."""
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, "controls.npy"), np.asarray(self.controls))
        np.save(os.path.join(path, "states.npy"), np.asarray(self.states))
        np.save(os.path.join(path, "rewards.npy"), np.asarray(self.rewards))
        logging.info(f"TrajectoryDataset saved to {path} ({self.n_trajectories} trajectories)")

    @classmethod
    def load(cls, path: str):
        """Load dataset from disk."""
        controls = jnp.array(np.load(os.path.join(path, "controls.npy")))
        states = jnp.array(np.load(os.path.join(path, "states.npy")))
        rewards = jnp.array(np.load(os.path.join(path, "rewards.npy")))
        logging.info(f"TrajectoryDataset loaded from {path}")
        return cls.from_arrays(controls, states, rewards)

    def subset(self, indices: jnp.ndarray):
        """Return a new dataset containing only the specified trajectory indices."""
        return TrajectoryDataset.from_arrays(
            controls=self.controls[indices],
            states=self.states[indices],
            rewards=self.rewards[indices],
        )

    def stats(self) -> dict:
        """Return summary statistics of the dataset."""
        return {
            "n_trajectories": self.n_trajectories,
            "horizon": self.horizon,
            "n_controls": self.n_controls,
            "n_states": self.n_states,
            "reward_mean": float(jnp.mean(self.rewards)),
            "reward_std": float(jnp.std(self.rewards)),
            "reward_min": float(jnp.min(self.rewards)),
            "reward_max": float(jnp.max(self.rewards)),
            "control_range": [
                float(jnp.min(self.controls)),
                float(jnp.max(self.controls)),
            ],
        }
