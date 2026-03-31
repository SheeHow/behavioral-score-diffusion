"""
Diffusion-coupled bandwidth schedule for Behavioral Score Diffusion.

The key insight: kernel bandwidths should co-vary with the diffusion noise
schedule. At high noise (early denoising), broad kernels capture global
behavioral structure. At low noise (late denoising), narrow kernels capture
local nonlinear dynamics.
"""

import jax.numpy as jnp
from dataclasses import dataclass


@dataclass
class BandwidthSchedule:
    """Manages the diffusion-coupled bandwidth schedule for BSD.

    The diffusion kernel bandwidth tracks the noise level:
        beta_diff(t) = sigma_t  (matches diffusion noise)

    The goal kernel bandwidth narrows as denoising progresses:
        nu_goal(t) = nu_goal_0 * (sigma_t / sigma_max) ^ gamma

    The context kernel bandwidth is fixed:
        nu_ctx = constant (set by state-space scale)
    """

    nu_ctx: float = 1.0        # Context (initial state) bandwidth — fixed
    nu_goal_0: float = 2.0     # Goal bandwidth at maximum noise
    gamma: float = 0.5         # Goal bandwidth decay exponent
    sigma_max: float = 1.0     # Maximum noise level (set from diffusion schedule)

    def get_bandwidths(self, sigma_t: float):
        """Compute all kernel bandwidths for a given noise level.

        Args:
            sigma_t: Current diffusion noise level.

        Returns:
            Tuple of (beta_diff, nu_ctx, nu_goal) bandwidths.
        """
        beta_diff = jnp.maximum(sigma_t, 1e-6)  # Diffusion bandwidth = noise level
        nu_ctx = self.nu_ctx
        nu_goal = self.nu_goal_0 * jnp.power(
            jnp.maximum(sigma_t / self.sigma_max, 1e-6), self.gamma
        )
        return beta_diff, nu_ctx, nu_goal

    @classmethod
    def from_diffusion_schedule(cls, sigmas: jnp.ndarray, nu_ctx: float = 1.0,
                                 nu_goal_0: float = 2.0, gamma: float = 0.5):
        """Create a bandwidth schedule from a diffusion noise schedule.

        Args:
            sigmas: [T] noise levels from diffusion schedule.
            nu_ctx: Context kernel bandwidth.
            nu_goal_0: Initial goal bandwidth.
            gamma: Goal bandwidth decay exponent.
        """
        sigma_max = float(jnp.max(sigmas))
        return cls(
            nu_ctx=nu_ctx,
            nu_goal_0=nu_goal_0,
            gamma=gamma,
            sigma_max=sigma_max,
        )
