"""
Behavioral Score Diffusion (BSD) Planner.

Drop-in replacement for run_diffusion() that uses kernel-based behavioral
score estimation instead of dynamics-model-based rollout + reward weighting.

The core change is in the reverse diffusion loop:
  MBD:  sample candidates → rollout through dynamics → reward-weight → denoise
  BSD:  kernel-weight against trajectory data → denoise

Everything else (diffusion schedule, shielded rollout for safety, post-processing)
is preserved from the original Safe-MPD implementation.
"""

from functools import partial
import os
import jax
from jax import numpy as jnp
from dataclasses import dataclass
from tqdm import tqdm
import time
import logging

import mbd
from mbd.utils import rollout_us, rollout_us_with_terminal
from mbd.scorers.behavioral_score import BehavioralScorer, BSDConfig
from mbd.scorers.bandwidth_schedule import BandwidthSchedule
from mbd.data.trajectory_dataset import TrajectoryDataset
from mbd.planners.mbd_planner import MBDConfig, _jit_function_cache


def run_diffusion_bsd(args=None, env=None, dataset=None, bsd_config=None):
    """Run the BSD (Behavioral Score Diffusion) planner.

    Multi-sample variant: at each diffusion step, samples K candidate control
    sequences from the kernel-weighted data distribution, evaluates each through
    the shielded dynamics, and takes a reward-weighted average (like MBD).

    This combines data-driven proposal (kernel) with exploration-exploitation
    (multi-sample + reward weighting), replacing MBD's model-based rollout
    with data-based retrieval.

    Args:
        args: MBDConfig or dict with diffusion parameters.
        env: Environment object (used for shielding, reward eval, rendering).
        dataset: TrajectoryDataset with collected trajectory data.
        bsd_config: BSDConfig (optional, uses defaults if None).

    Returns:
        rew_final: Final reward value.
        Y0: Final action sequence [H, Nu].
        trajectory_states: State trajectory [H+1, Nx].
        timing_info: Dict with timing breakdown.
    """
    from mbd.planners.mbd_planner import dict_to_config_obj

    if isinstance(args, dict):
        args = dict_to_config_obj(args)
    elif args is None:
        raise ValueError("args parameter is required")

    if dataset is None:
        raise ValueError("dataset parameter is required for BSD planner")

    if bsd_config is None:
        bsd_config = BSDConfig()

    logging.info("=== BSD PLANNER (multi-sample) ===")
    total_start_time = time.time()

    rng = jax.random.PRNGKey(seed=args.seed)
    Nu = env.action_size
    Nsample = args.Nsample  # Number of candidate samples per step (same as MBD)

    # --- Set up JIT functions for post-processing and shielding ---
    cache_key = f"{type(env).__name__}_env_funcs"
    if cache_key in _jit_function_cache:
        step_env_jit, rollout_us_jit, rollout_us_with_terminal_jit = _jit_function_cache[cache_key]
    else:
        step_env_jit = jax.jit(env.step)
        rollout_us_jit = jax.jit(partial(rollout_us, step_env_jit))
        rollout_us_with_terminal_jit = jax.jit(partial(rollout_us_with_terminal, step_env_jit, env))
        _jit_function_cache[cache_key] = (step_env_jit, rollout_us_jit, rollout_us_with_terminal_jit)

    rng, rng_reset = jax.random.split(rng)
    state_init = env.reset(rng_reset)

    # --- Set up diffusion schedule (identical to MBD) ---
    betas = jnp.linspace(args.beta0, args.betaT, args.Ndiffuse)
    alphas = 1.0 - betas
    alphas_bar = jnp.cumprod(alphas)
    sigmas = jnp.sqrt(1 - alphas_bar)

    # --- Set up BSD scorer (for kernel weight computation) ---
    bw_schedule = BandwidthSchedule.from_diffusion_schedule(
        sigmas,
        nu_ctx=bsd_config.nu_ctx,
        nu_goal_0=bsd_config.nu_goal_0,
        gamma=bsd_config.gamma,
    )

    scorer = BehavioralScorer(dataset, bsd_config, bw_schedule)
    compute_weights_jit = scorer.compute_weights_jit()

    # Warm up JIT
    warmup_start = time.time()
    dummy_Y = jnp.zeros([args.Hsample, Nu])
    x_0 = state_init.pipeline_state
    x_goal = jnp.array(env.xg)
    _ = compute_weights_jit(dummy_Y, x_0, x_goal, sigmas[-1])
    # Also warm up vmap rollout
    dummy_Y0s = jnp.zeros([Nsample, args.Hsample, Nu])
    _ = jax.vmap(rollout_us_with_terminal_jit, in_axes=(None, 0))(
        state_init, dummy_Y0s
    )
    warmup_time = time.time() - warmup_start
    logging.info(f"BSD JIT warmup: {warmup_time:.2f}s")

    # Pre-fetch dataset controls for sampling
    data_controls = dataset.controls  # [N, H, Nu]

    # --- BSD Reverse Diffusion Loop ---
    pure_diffusion_start = time.time()

    Y_t = jnp.zeros([args.Hsample, Nu])
    x_0 = state_init.pipeline_state
    x_goal = jnp.array(env.xg)

    effective_Ks = []
    mean_rewards = []

    with tqdm(range(args.Ndiffuse - 1, 0, -1), desc="BSD Diffusing") as pbar:
        for idx, i in enumerate(pbar):
            sigma_t = sigmas[i]

            # --- Step 1: Compute kernel weights over dataset ---
            w_norm, eff_K = compute_weights_jit(Y_t, x_0, x_goal, sigma_t)
            effective_Ks.append(float(eff_K))

            # --- Step 2: Sample K candidates from kernel distribution ---
            rng, rng_sample, rng_noise = jax.random.split(rng, 3)

            # Sample indices from kernel weights (categorical distribution)
            sample_indices = jax.random.choice(
                rng_sample, dataset.n_trajectories,
                shape=(Nsample,), p=w_norm, replace=True
            )
            # Retrieve corresponding control sequences
            Y0s = data_controls[sample_indices]  # [Nsample, H, Nu]

            # Add diffusion noise (same as MBD's sampling around Ybar)
            eps = jax.random.normal(rng_noise, Y0s.shape)
            Y0s = Y0s + eps * sigmas[i] / jnp.sqrt(alphas_bar[i - 1])
            Y0s = jnp.clip(Y0s, -1.0, 1.0)

            # --- Step 3: Evaluate all candidates through shielded rollout ---
            rews, _, Y0s_applied = jax.vmap(
                rollout_us_with_terminal_jit, in_axes=(None, 0)
            )(state_init, Y0s)
            Y0s_effective = Y0s_applied

            # --- Step 4: Reward-weighted average (same as MBD) ---
            rew_std = jnp.maximum(rews.std(), 1e-4)
            rew_mean = rews.mean()
            logp = (rews - rew_mean) / rew_std / args.temp_sample
            weights = jax.nn.softmax(logp)
            Ybar = jnp.einsum("n,nij->ij", weights, Y0s_effective)

            mean_rewards.append(float(rew_mean))

            # --- DDPM-style reverse step ---
            Yi = Y_t * jnp.sqrt(alphas_bar[i])
            score = (1.0 / (1.0 - alphas_bar[i])) * (
                -Yi + jnp.sqrt(alphas_bar[i]) * Ybar
            )
            Yim1 = (1.0 / jnp.sqrt(alphas[i])) * (
                Yi + (1.0 - alphas_bar[i]) * score
            )
            Ybar_im1 = Yim1 / jnp.sqrt(alphas_bar[i - 1])

            # Add noise for next step (except at final step)
            rng, rng_step_noise = jax.random.split(rng)
            if i > 1:
                noise = jax.random.normal(rng_step_noise, Y_t.shape)
                sigma_cond = jnp.sqrt(
                    (1 - alphas[i]) * (1 - alphas_bar[i - 1]) / (1 - alphas_bar[i])
                )
                Y_t = Ybar_im1 + sigma_cond * noise
            else:
                Y_t = Ybar_im1

            pbar.set_postfix({
                "rew": f"{float(rew_mean):.2f}",
                "effK": f"{float(eff_K):.0f}",
            })

    Y0 = Y_t  # Final denoised control sequence
    pure_diffusion_time = time.time() - pure_diffusion_start

    # --- Post-processing (identical to MBD) ---
    post_start = time.time()

    # Get shielded final trajectory
    _, _, Y0_applied = rollout_us_with_terminal_jit(state_init, Y0)
    Y0 = Y0_applied

    # Compute trajectory states
    trajectory_length = Y0.shape[0] + 1
    xs = jnp.zeros((trajectory_length, state_init.pipeline_state.shape[0]))
    xs = xs.at[0].set(state_init.pipeline_state)

    state = state_init
    for t in range(Y0.shape[0]):
        action = jnp.array(Y0[t])
        state = step_env_jit(state, action)
        xs = xs.at[t + 1].set(state.pipeline_state)

    trajectory_states = xs
    post_time = time.time() - post_start

    # --- Final reward ---
    rew_final, _, _ = rollout_us_with_terminal_jit(state_init, Y0)

    # --- Timing ---
    total_time = time.time() - total_start_time
    timing_info = {
        'total_time': total_time,
        'compilation_time': warmup_time,
        'warmup_time': warmup_time,
        'pure_diffusion_time': pure_diffusion_time,
        'post_processing_time': post_time,
        'overhead_time': total_time - pure_diffusion_time,
        # BSD-specific diagnostics
        'bsd_effective_K_mean': float(jnp.mean(jnp.array(effective_Ks))),
        'bsd_effective_K_min': float(jnp.min(jnp.array(effective_Ks))),
        'bsd_mean_reward_final': float(mean_rewards[-1]) if mean_rewards else 0.0,
        'dataset_size': dataset.n_trajectories,
    }

    logging.info(f"\n=== BSD TIMING REPORT ===")
    logging.info(f"Total time:              {total_time:.3f}s")
    logging.info(f"Pure diffusion time:     {pure_diffusion_time:.3f}s")
    logging.info(f"Effective K (mean):      {timing_info['bsd_effective_K_mean']:.1f}")
    logging.info(f"Effective K (min):       {timing_info['bsd_effective_K_min']:.1f}")
    logging.info(f"Mean reward (final):     {timing_info['bsd_mean_reward_final']:.3f}")
    logging.info(f"Dataset size:            {dataset.n_trajectories}")
    logging.info(f"Nsample per step:        {Nsample}")
    logging.info(f"Final reward:            {float(rew_final):.3e}")
    logging.info(f"=========================")

    return rew_final, Y0, trajectory_states, timing_info


def run_diffusion_bsd_fixed_bandwidth(args=None, env=None, dataset=None,
                                       bsd_config=None, fixed_sigma=None):
    """BSD with fixed kernel bandwidth (ablation: no multi-scale coupling).

    Same multi-sample architecture as run_diffusion_bsd, but the kernel
    bandwidth is held constant (at the broadest sigma) instead of narrowing
    with the noise schedule. Tests whether the multi-scale property helps.

    Args:
        Same as run_diffusion_bsd, plus:
        fixed_sigma: Fixed sigma for kernel bandwidth (defaults to max sigma).
    """
    from mbd.planners.mbd_planner import dict_to_config_obj

    if isinstance(args, dict):
        args = dict_to_config_obj(args)
    elif args is None:
        raise ValueError("args parameter is required")

    if dataset is None:
        raise ValueError("dataset parameter is required for BSD planner")

    if bsd_config is None:
        bsd_config = BSDConfig()

    betas = jnp.linspace(args.beta0, args.betaT, args.Ndiffuse)
    alphas = 1.0 - betas
    alphas_bar = jnp.cumprod(alphas)
    sigmas = jnp.sqrt(1 - alphas_bar)

    if fixed_sigma is None:
        fixed_sigma = float(sigmas[-1])

    logging.info(f"=== BSD FIXED BANDWIDTH (sigma={fixed_sigma:.4f}) ===")
    total_start_time = time.time()

    rng = jax.random.PRNGKey(seed=args.seed)
    Nu = env.action_size
    Nsample = args.Nsample

    cache_key = f"{type(env).__name__}_env_funcs"
    if cache_key in _jit_function_cache:
        step_env_jit, rollout_us_jit, rollout_us_with_terminal_jit = _jit_function_cache[cache_key]
    else:
        step_env_jit = jax.jit(env.step)
        rollout_us_jit = jax.jit(partial(rollout_us, step_env_jit))
        rollout_us_with_terminal_jit = jax.jit(partial(rollout_us_with_terminal, step_env_jit, env))
        _jit_function_cache[cache_key] = (step_env_jit, rollout_us_jit, rollout_us_with_terminal_jit)

    rng, rng_reset = jax.random.split(rng)
    state_init = env.reset(rng_reset)

    # Fixed config: gamma=0 (goal bandwidth also fixed)
    fixed_config = BSDConfig(
        nu_ctx=bsd_config.nu_ctx,
        nu_goal_0=bsd_config.nu_goal_0,
        gamma=0.0,
        min_effective_weight=bsd_config.min_effective_weight,
        steering_weight=bsd_config.steering_weight,
        dim_scale_diff=bsd_config.dim_scale_diff,
        dim_scale_exp=bsd_config.dim_scale_exp,
        diff_bandwidth_mult=bsd_config.diff_bandwidth_mult,
        reward_temp=bsd_config.reward_temp,
    )

    bw_schedule = BandwidthSchedule.from_diffusion_schedule(
        sigmas, nu_ctx=fixed_config.nu_ctx, nu_goal_0=fixed_config.nu_goal_0,
        gamma=fixed_config.gamma,
    )

    scorer = BehavioralScorer(dataset, fixed_config, bw_schedule)
    compute_weights_jit = scorer.compute_weights_jit()

    # Warm up
    warmup_start = time.time()
    dummy_Y = jnp.zeros([args.Hsample, Nu])
    x_0 = state_init.pipeline_state
    x_goal = jnp.array(env.xg)
    _ = compute_weights_jit(dummy_Y, x_0, x_goal, jnp.float32(fixed_sigma))
    dummy_Y0s = jnp.zeros([Nsample, args.Hsample, Nu])
    _ = jax.vmap(rollout_us_with_terminal_jit, in_axes=(None, 0))(
        state_init, dummy_Y0s
    )
    warmup_time = time.time() - warmup_start

    data_controls = dataset.controls
    sigma_t_fixed = jnp.float32(fixed_sigma)

    pure_diffusion_start = time.time()

    Y_t = jnp.zeros([args.Hsample, Nu])
    x_0 = state_init.pipeline_state
    x_goal = jnp.array(env.xg)

    effective_Ks = []
    mean_rewards = []

    with tqdm(range(args.Ndiffuse - 1, 0, -1), desc="BSD-fix Diffusing") as pbar:
        for idx, i in enumerate(pbar):
            # Use FIXED sigma for kernel weights
            w_norm, eff_K = compute_weights_jit(
                Y_t, x_0, x_goal, sigma_t_fixed
            )
            effective_Ks.append(float(eff_K))

            rng, rng_sample, rng_noise = jax.random.split(rng, 3)

            sample_indices = jax.random.choice(
                rng_sample, dataset.n_trajectories,
                shape=(Nsample,), p=w_norm, replace=True
            )
            Y0s = data_controls[sample_indices]

            eps = jax.random.normal(rng_noise, Y0s.shape)
            Y0s = Y0s + eps * sigmas[i] / jnp.sqrt(alphas_bar[i - 1])
            Y0s = jnp.clip(Y0s, -1.0, 1.0)

            rews, _, Y0s_applied = jax.vmap(
                rollout_us_with_terminal_jit, in_axes=(None, 0)
            )(state_init, Y0s)
            Y0s_effective = Y0s_applied

            rew_std = jnp.maximum(rews.std(), 1e-4)
            rew_mean = rews.mean()
            logp = (rews - rew_mean) / rew_std / args.temp_sample
            weights = jax.nn.softmax(logp)
            Ybar = jnp.einsum("n,nij->ij", weights, Y0s_effective)

            mean_rewards.append(float(rew_mean))

            Yi = Y_t * jnp.sqrt(alphas_bar[i])
            score = (1.0 / (1.0 - alphas_bar[i])) * (
                -Yi + jnp.sqrt(alphas_bar[i]) * Ybar
            )
            Yim1 = (1.0 / jnp.sqrt(alphas[i])) * (
                Yi + (1.0 - alphas_bar[i]) * score
            )
            Ybar_im1 = Yim1 / jnp.sqrt(alphas_bar[i - 1])

            rng, rng_step_noise = jax.random.split(rng)
            if i > 1:
                noise = jax.random.normal(rng_step_noise, Y_t.shape)
                sigma_cond = jnp.sqrt(
                    (1 - alphas[i]) * (1 - alphas_bar[i - 1]) / (1 - alphas_bar[i])
                )
                Y_t = Ybar_im1 + sigma_cond * noise
            else:
                Y_t = Ybar_im1

            pbar.set_postfix({
                "rew": f"{float(rew_mean):.2f}",
                "effK": f"{float(eff_K):.0f}",
            })

    Y0 = Y_t
    pure_diffusion_time = time.time() - pure_diffusion_start

    _, _, Y0_applied = rollout_us_with_terminal_jit(state_init, Y0)
    Y0 = Y0_applied

    trajectory_length = Y0.shape[0] + 1
    xs = jnp.zeros((trajectory_length, state_init.pipeline_state.shape[0]))
    xs = xs.at[0].set(state_init.pipeline_state)

    state = state_init
    for t in range(Y0.shape[0]):
        state = step_env_jit(state, jnp.array(Y0[t]))
        xs = xs.at[t + 1].set(state.pipeline_state)

    trajectory_states = xs
    rew_final, _, _ = rollout_us_with_terminal_jit(state_init, Y0)

    total_time = time.time() - total_start_time
    timing_info = {
        'total_time': total_time,
        'compilation_time': warmup_time,
        'warmup_time': warmup_time,
        'pure_diffusion_time': pure_diffusion_time,
        'post_processing_time': total_time - pure_diffusion_time - warmup_time,
        'overhead_time': total_time - pure_diffusion_time,
        'bsd_effective_K_mean': float(jnp.mean(jnp.array(effective_Ks))),
        'bsd_effective_K_min': float(jnp.min(jnp.array(effective_Ks))),
        'bsd_mean_reward_final': float(mean_rewards[-1]) if mean_rewards else 0.0,
        'dataset_size': dataset.n_trajectories,
        'bsd_fixed_sigma': fixed_sigma,
    }

    return rew_final, Y0, trajectory_states, timing_info


def run_nearest_neighbor(args=None, env=None, dataset=None):
    """Pure nearest-neighbor retrieval (ablation: no diffusion at all).

    Finds the trajectory in the dataset most similar to the current
    (initial_state, goal_state) context and returns it directly.
    No denoising, no iterative refinement.
    """
    from mbd.planners.mbd_planner import dict_to_config_obj, _jit_function_cache

    if isinstance(args, dict):
        args = dict_to_config_obj(args)

    total_start_time = time.time()

    rng = jax.random.PRNGKey(seed=args.seed)
    rng, rng_reset = jax.random.split(rng)
    state_init = env.reset(rng_reset)

    # JIT functions for post-processing
    cache_key = f"{type(env).__name__}_env_funcs"
    if cache_key in _jit_function_cache:
        step_env_jit, _, rollout_us_with_terminal_jit = _jit_function_cache[cache_key]
    else:
        step_env_jit = jax.jit(env.step)
        rollout_us_with_terminal_jit = jax.jit(
            partial(rollout_us_with_terminal, step_env_jit, env)
        )
        _jit_function_cache[cache_key] = (
            step_env_jit,
            jax.jit(partial(rollout_us, step_env_jit)),
            rollout_us_with_terminal_jit,
        )

    x_0 = state_init.pipeline_state
    x_goal = jnp.array(env.xg)

    # Find nearest trajectory by context (init_state, goal proximity)
    ctx_dist = jnp.sum((dataset.init_states - x_0[None, :]) ** 2, axis=1)
    goal_dist = jnp.sum((dataset.final_states - x_goal[None, :]) ** 2, axis=1)
    combined_dist = ctx_dist + goal_dist
    best_idx = jnp.argmin(combined_dist)

    Y0 = dataset.controls[best_idx]  # [H, Nu]
    Y0 = jnp.clip(Y0, -1.0, 1.0)

    # Apply shielded rollout
    _, _, Y0 = rollout_us_with_terminal_jit(state_init, Y0)

    # Compute trajectory states
    trajectory_length = Y0.shape[0] + 1
    xs = jnp.zeros((trajectory_length, state_init.pipeline_state.shape[0]))
    xs = xs.at[0].set(state_init.pipeline_state)

    state = state_init
    for t in range(Y0.shape[0]):
        state = step_env_jit(state, jnp.array(Y0[t]))
        xs = xs.at[t + 1].set(state.pipeline_state)

    trajectory_states = xs
    rew_final, _, _ = rollout_us_with_terminal_jit(state_init, Y0)

    total_time = time.time() - total_start_time
    timing_info = {
        'total_time': total_time,
        'pure_diffusion_time': 0.0,  # No diffusion
        'post_processing_time': total_time,
        'dataset_size': dataset.n_trajectories,
        'method': 'nearest_neighbor',
    }

    return rew_final, Y0, trajectory_states, timing_info
