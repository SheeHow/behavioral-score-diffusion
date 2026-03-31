"""
Behavioral Score Diffusion (BSD): Kernel-based score estimation from trajectory data.

This module implements the core BSD algorithm: computing the diffusion score
function as a Nadaraya-Watson kernel regression over stored trajectory data.
Instead of rolling out candidate trajectories through an analytical dynamics
model (as in MBD), BSD retrieves relevant trajectories from a dataset and
computes a kernel-weighted average.

The diffusion noise schedule controls kernel bandwidths, creating multi-scale
nonlinear trajectory interpolation:
  - High noise → broad kernel → global behavioral averaging
  - Low noise  → narrow kernel → local nonlinear dynamics

References:
    - Pan et al., "Model-Based Diffusion for Trajectory Optimization", NeurIPS 2024
    - Kim et al., "Safe Model Predictive Diffusion with Shielding", ICRA 2026
    - Yang et al., "Training-free score-based diffusion...", arXiv 2026
    - Veiga et al., "Kernel-Smoothed Scores for Denoising Diffusion", arXiv 2025
"""

import jax
import jax.numpy as jnp
from functools import partial
from dataclasses import dataclass
import logging

from .bandwidth_schedule import BandwidthSchedule
from ..data.trajectory_dataset import TrajectoryDataset


@dataclass
class BSDConfig:
    """Configuration for Behavioral Score Diffusion.

    Attributes:
        nu_ctx: Context kernel bandwidth (initial state similarity).
        nu_goal_0: Initial goal kernel bandwidth (at maximum noise).
        gamma: Goal bandwidth decay exponent (controls multi-scale goal conditioning).
        min_effective_weight: Minimum total weight before falling back to uniform.
        steering_weight: Weight for steering cost (same as MBD baseline).
        dim_scale_diff: If True, scale diffusion bandwidth by d^dim_scale_exp to
            correct for curse of dimensionality in the high-dimensional control space.
        dim_scale_exp: Exponent for dimension scaling: beta *= d^exp.
            0.5 = sqrt(d) (aggressive), 0.25 = d^(1/4) (moderate). Default 0.25.
        diff_bandwidth_mult: Additional multiplier on diffusion kernel bandwidth.
        reward_temp: Reward temperature for kernel weighting. Higher values give
            stronger preference to high-reward trajectories. 0 = no reward weighting.
    """
    nu_ctx: float = 2.0
    nu_goal_0: float = 3.0
    gamma: float = 0.5
    min_effective_weight: float = 1e-10
    steering_weight: float = 0.01
    dim_scale_diff: bool = True
    dim_scale_exp: float = 0.50
    diff_bandwidth_mult: float = 1.0
    reward_temp: float = 10.0


class BehavioralScorer:
    """Kernel-based behavioral score estimator for diffusion planning.

    Replaces MBD's dynamics-rollout + reward-weighting with kernel-weighted
    retrieval from a trajectory dataset. The score function is:

        E[Y_0 | Y_t] ≈ Σ_j w_j * u_j

    where w_j = softmax(log_w_diff + log_w_ctx + log_w_goal) and u_j are
    stored control sequences.
    """

    def __init__(self, dataset: TrajectoryDataset, bsd_config: BSDConfig,
                 bandwidth_schedule: BandwidthSchedule):
        self.dataset = dataset
        self.config = bsd_config
        self.bw = bandwidth_schedule

        # Pre-store dataset arrays as module attributes for JIT access
        self._controls = dataset.controls           # [N, H, Nu]
        self._controls_flat = dataset.controls_flat  # [N, H*Nu]
        self._states = dataset.states               # [N, H+1, Nx]
        self._init_states = dataset.init_states     # [N, Nx]
        self._final_states = dataset.final_states   # [N, Nx]

        # Normalize rewards to [0, 1] for stable reward weighting
        r = dataset.rewards
        r_min, r_max = jnp.min(r), jnp.max(r)
        r_range = jnp.maximum(r_max - r_min, 1e-8)
        self._rewards_normalized = (r - r_min) / r_range  # [N] in [0, 1]

        logging.info(
            f"BehavioralScorer initialized: {dataset.n_trajectories} trajectories, "
            f"H={dataset.horizon}, Nu={dataset.n_controls}, Nx={dataset.n_states}, "
            f"reward range=[{float(r_min):.3f}, {float(r_max):.3f}]"
        )

    def compute_score(self, Y_t, x_0, x_goal, sigma_t):
        """Compute the behavioral score: the denoised trajectory estimate.

        This is the core BSD operation. It replaces MBD's:
            1. Sample candidates from N(Ybar, sigma)
            2. Rollout each through dynamics → get states
            3. Compute rewards → softmax weights
            4. Weighted average → denoised estimate

        With:
            1. Compute triple-kernel weights against all stored trajectories
            2. Kernel-weighted average of stored controls → denoised estimate
            3. Kernel-weighted average of stored states → state estimate

        Args:
            Y_t: [H, Nu] Current noisy control sequence.
            x_0: [Nx] Current (initial) state.
            x_goal: [Nx] Goal state.
            sigma_t: Current diffusion noise level (scalar).

        Returns:
            Y0_hat: [H, Nu] Denoised control estimate.
            X_hat: [H+1, Nx] Estimated state trajectory.
            diagnostics: dict with effective_K, max_weight, etc.
        """
        Y_t_flat = Y_t.flatten()  # [H*Nu]

        # Get bandwidths for current noise level
        beta_diff, nu_ctx, nu_goal = self.bw.get_bandwidths(sigma_t)

        # Scale diffusion bandwidth by d^exp for high-dimensional controls
        d = float(self._controls_flat.shape[1])
        if self.config.dim_scale_diff:
            beta_diff = beta_diff * (d ** self.config.dim_scale_exp) * self.config.diff_bandwidth_mult
        else:
            beta_diff = beta_diff * self.config.diff_bandwidth_mult

        # --- Kernel 1: Diffusion proximity ---
        # How similar is each stored trajectory to the current noisy trajectory?
        diff_sq = jnp.sum(
            (self._controls_flat - Y_t_flat[None, :]) ** 2, axis=1
        )  # [N]
        log_w_diff = -diff_sq / (2.0 * beta_diff ** 2)

        # --- Kernel 2: Context (initial state) similarity ---
        # How similar is each stored trajectory's initial state to current state?
        ctx_sq = jnp.sum(
            (self._init_states - x_0[None, :]) ** 2, axis=1
        )  # [N]
        log_w_ctx = -ctx_sq / (2.0 * nu_ctx ** 2)

        # --- Kernel 3: Goal relevance ---
        # How close does each stored trajectory get to the goal?
        goal_sq = jnp.sum(
            (self._final_states - x_goal[None, :]) ** 2, axis=1
        )  # [N]
        log_w_goal = -goal_sq / (2.0 * nu_goal ** 2)

        # --- Kernel 4: Reward weighting ---
        # Prefer high-reward trajectories (analogous to MBD's reward softmax)
        log_w_rew = self.config.reward_temp * self._rewards_normalized

        # --- Combined weights (log-space for numerical stability) ---
        log_w = log_w_diff + log_w_ctx + log_w_goal + log_w_rew
        # Shift for numerical stability before exp
        log_w_shifted = log_w - jnp.max(log_w)
        w = jnp.exp(log_w_shifted)
        w_sum = jnp.sum(w) + 1e-10
        w_normalized = w / w_sum

        # --- Kernel-weighted denoised estimates ---
        # Control estimate: E[u | Y_t, x_0, x_goal]
        Y0_hat = jnp.einsum('n,nhc->hc', w_normalized, self._controls)  # [H, Nu]

        # State estimate (co-retrieval): E[x | Y_t, x_0, x_goal]
        X_hat = jnp.einsum('n,nhc->hc', w_normalized, self._states)  # [H+1, Nx]

        # --- Diagnostics ---
        effective_K = 1.0 / (jnp.sum(w_normalized ** 2) + 1e-10)
        max_weight = jnp.max(w_normalized)

        return Y0_hat, X_hat, effective_K, max_weight

    def compute_score_jit(self):
        """Return a JIT-compiled version of compute_score.

        The JIT function captures dataset arrays as closed-over constants.
        """
        # Capture immutable data as closure variables
        controls = self._controls
        controls_flat = self._controls_flat
        states = self._states
        init_states = self._init_states
        final_states = self._final_states
        rewards_norm = self._rewards_normalized
        nu_ctx_val = self.config.nu_ctx
        nu_goal_0_val = self.config.nu_goal_0
        gamma_val = self.config.gamma
        sigma_max_val = self.bw.sigma_max
        dim_scale = self.config.dim_scale_diff
        dim_exp = self.config.dim_scale_exp
        diff_mult = self.config.diff_bandwidth_mult
        reward_temp = self.config.reward_temp
        control_dim = float(self._controls_flat.shape[1])

        @jax.jit
        def _compute_score(Y_t, x_0, x_goal, sigma_t):
            Y_t_flat = Y_t.flatten()

            # Bandwidths
            beta_diff = jnp.maximum(sigma_t, 1e-6)
            # Scale bandwidth by d^exp to correct for high dimensionality.
            # In d dimensions, E[||x-y||^2] ~ d*sigma^2 for Gaussian data.
            # exp=0.5 → sqrt(d) (aggressive), exp=0.25 → d^(1/4) (moderate).
            beta_diff = jnp.where(
                dim_scale,
                beta_diff * jnp.power(control_dim, dim_exp) * diff_mult,
                beta_diff * diff_mult,
            )
            nu_goal = nu_goal_0_val * jnp.power(
                jnp.maximum(sigma_t / sigma_max_val, 1e-6), gamma_val
            )

            # Kernel 1: diffusion proximity
            diff_sq = jnp.sum(
                (controls_flat - Y_t_flat[None, :]) ** 2, axis=1
            )
            log_w_diff = -diff_sq / (2.0 * beta_diff ** 2)

            # Kernel 2: context similarity
            ctx_sq = jnp.sum(
                (init_states - x_0[None, :]) ** 2, axis=1
            )
            log_w_ctx = -ctx_sq / (2.0 * nu_ctx_val ** 2)

            # Kernel 3: goal relevance
            goal_sq = jnp.sum(
                (final_states - x_goal[None, :]) ** 2, axis=1
            )
            log_w_goal = -goal_sq / (2.0 * nu_goal ** 2)

            # Kernel 4: reward weighting
            log_w_rew = reward_temp * rewards_norm

            # Combined weights
            log_w = log_w_diff + log_w_ctx + log_w_goal + log_w_rew
            log_w_shifted = log_w - jnp.max(log_w)
            w = jnp.exp(log_w_shifted)
            w_sum = jnp.sum(w) + 1e-10
            w_norm = w / w_sum

            # Kernel-weighted estimates
            Y0_hat = jnp.einsum('n,nhc->hc', w_norm, controls)
            X_hat = jnp.einsum('n,nhc->hc', w_norm, states)

            effective_K = 1.0 / (jnp.sum(w_norm ** 2) + 1e-10)
            max_weight = jnp.max(w_norm)

            return Y0_hat, X_hat, effective_K, max_weight

        return _compute_score

    def compute_weights_jit(self):
        """Return a JIT-compiled function that computes kernel weights only.

        Used by multi-sample BSD to get sampling probabilities over the dataset.
        Returns (w_normalized, effective_K) without computing weighted averages.
        """
        controls_flat = self._controls_flat
        init_states = self._init_states
        final_states = self._final_states
        rewards_norm = self._rewards_normalized
        nu_ctx_val = self.config.nu_ctx
        nu_goal_0_val = self.config.nu_goal_0
        gamma_val = self.config.gamma
        sigma_max_val = self.bw.sigma_max
        dim_scale = self.config.dim_scale_diff
        dim_exp = self.config.dim_scale_exp
        diff_mult = self.config.diff_bandwidth_mult
        reward_temp = self.config.reward_temp
        control_dim = float(self._controls_flat.shape[1])

        @jax.jit
        def _compute_weights(Y_t, x_0, x_goal, sigma_t):
            Y_t_flat = Y_t.flatten()

            beta_diff = jnp.maximum(sigma_t, 1e-6)
            beta_diff = jnp.where(
                dim_scale,
                beta_diff * jnp.power(control_dim, dim_exp) * diff_mult,
                beta_diff * diff_mult,
            )
            nu_goal = nu_goal_0_val * jnp.power(
                jnp.maximum(sigma_t / sigma_max_val, 1e-6), gamma_val
            )

            diff_sq = jnp.sum(
                (controls_flat - Y_t_flat[None, :]) ** 2, axis=1
            )
            log_w_diff = -diff_sq / (2.0 * beta_diff ** 2)

            ctx_sq = jnp.sum(
                (init_states - x_0[None, :]) ** 2, axis=1
            )
            log_w_ctx = -ctx_sq / (2.0 * nu_ctx_val ** 2)

            goal_sq = jnp.sum(
                (final_states - x_goal[None, :]) ** 2, axis=1
            )
            log_w_goal = -goal_sq / (2.0 * nu_goal ** 2)

            log_w_rew = reward_temp * rewards_norm

            log_w = log_w_diff + log_w_ctx + log_w_goal + log_w_rew
            log_w_shifted = log_w - jnp.max(log_w)
            w = jnp.exp(log_w_shifted)
            w_sum = jnp.sum(w) + 1e-10
            w_norm = w / w_sum

            effective_K = 1.0 / (jnp.sum(w_norm ** 2) + 1e-10)

            return w_norm, effective_K

        return _compute_weights
