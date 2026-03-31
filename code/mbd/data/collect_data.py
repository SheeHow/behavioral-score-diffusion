"""
Collect trajectory data from Safe-MPD baseline runs for BSD training.

Runs the baseline MBD planner across randomized initial conditions and saves
the resulting (control, state, reward) trajectories as a TrajectoryDataset.
"""

import jax
import jax.numpy as jnp
import numpy as np
import logging
import time
from functools import partial
from tqdm import tqdm

from .trajectory_dataset import TrajectoryDataset


def collect_trajectories_from_baseline(
    run_diffusion_fn,
    env_factory,
    config,
    n_trajectories: int = 1000,
    randomize_init: bool = True,
    seed: int = 0,
    filter_failures: bool = True,
    min_reward: float = 0.0,
):
    """Collect trajectory data by running the MBD baseline planner.

    Args:
        run_diffusion_fn: The run_diffusion function from mbd_planner.
        env_factory: Callable that creates and returns a configured environment.
        config: MBDConfig (or dict) for the planner.
        n_trajectories: Number of trajectories to collect.
        randomize_init: Whether to randomize initial positions.
        seed: Base random seed.
        filter_failures: If True, skip trajectories with reward below min_reward.
        min_reward: Minimum reward threshold for keeping a trajectory.

    Returns:
        TrajectoryDataset with collected trajectories.
    """
    rng = jax.random.PRNGKey(seed)
    all_controls = []
    all_states = []
    all_rewards = []

    logging.info(f"Collecting {n_trajectories} trajectories from baseline planner...")
    start_time = time.time()

    n_attempts = 0
    max_attempts = n_trajectories * 3  # Allow up to 3x attempts for filtering

    pbar = tqdm(total=n_trajectories, desc="Collecting data")
    while len(all_controls) < n_trajectories and n_attempts < max_attempts:
        rng, rng_init, rng_config = jax.random.split(rng, 3)

        # Create environment (potentially with randomized init)
        env = env_factory()

        if randomize_init:
            # Randomize initial position within reasonable bounds
            dx_range = (-12.0, 12.0)
            dy_range = (1.0, 10.0)
            theta_range = (-0.5, 0.5)

            dx = float(jax.random.uniform(rng_init, minval=dx_range[0], maxval=dx_range[1]))
            dy = float(jax.random.uniform(
                jax.random.fold_in(rng_init, 1),
                minval=dy_range[0], maxval=dy_range[1],
            ))
            theta1 = float(jax.random.uniform(
                jax.random.fold_in(rng_init, 2),
                minval=theta_range[0], maxval=theta_range[1],
            ))
            theta2 = float(jax.random.uniform(
                jax.random.fold_in(rng_init, 3),
                minval=theta_range[0], maxval=theta_range[1],
            ))
            env.set_init_pos(dx=dx, dy=dy, theta1=theta1, theta2=theta2)

        # Override config for data collection (no rendering, deterministic seed)
        if isinstance(config, dict):
            trial_config = dict(config)
        else:
            trial_config = {k: getattr(config, k) for k in config.__dataclass_fields__}

        trial_config["render"] = False
        trial_config["show_animation"] = False
        trial_config["save_animation"] = False
        trial_config["save_denoising_animation"] = False
        trial_config["seed"] = seed + n_attempts

        n_attempts += 1

        try:
            rew_final, Y0, trajectory_states, _ = run_diffusion_fn(
                args=trial_config, env=env
            )

            rew_val = float(rew_final)

            # Filter out failed trajectories
            if filter_failures and rew_val < min_reward:
                continue

            all_controls.append(np.asarray(Y0))
            all_states.append(np.asarray(trajectory_states))
            all_rewards.append(rew_val)
            pbar.update(1)

        except Exception as e:
            logging.warning(f"Trial {n_attempts} failed: {e}")
            continue

    pbar.close()

    elapsed = time.time() - start_time
    n_collected = len(all_controls)
    logging.info(
        f"Collected {n_collected}/{n_trajectories} trajectories in {elapsed:.1f}s "
        f"({n_attempts} attempts, {elapsed / max(n_collected, 1):.2f}s per trajectory)"
    )

    if n_collected == 0:
        raise RuntimeError("No trajectories collected! Check environment config.")

    # Stack into arrays
    controls = jnp.array(np.stack(all_controls, axis=0))  # [N, H, Nu]
    states = jnp.array(np.stack(all_states, axis=0))      # [N, H+1, Nx]
    rewards = jnp.array(np.array(all_rewards))             # [N]

    return TrajectoryDataset.from_arrays(controls, states, rewards)
