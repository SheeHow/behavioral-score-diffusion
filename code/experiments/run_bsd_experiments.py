"""
BSD Experiment Runner

Runs all experimental conditions for the Behavioral Score Diffusion paper:
  1. MBD baseline (analytical dynamics)
  2. BSD (kernel behavioral score, adaptive bandwidth)
  3. BSD-fix (fixed bandwidth ablation)
  4. NN (nearest neighbor, no diffusion)

For each condition, evaluates across:
  - Multiple dynamical systems (Bicycle, TT2D, AccTT2D)
  - Multiple scenarios (parking, navigation)
  - 100 random initial conditions per (system, scenario, condition)

Outputs:
  - results/<system>/<condition>/trial_<i>.npz
  - results/summary.json
"""

import os
import sys
import json
import time
import logging
import argparse
import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm
from functools import partial

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import mbd
from mbd.planners.mbd_planner import MBDConfig, run_diffusion, clear_jit_cache
from mbd.planners.bsd_planner import (
    run_diffusion_bsd,
    run_diffusion_bsd_fixed_bandwidth,
    run_nearest_neighbor,
)
from mbd.scorers.behavioral_score import BSDConfig
from mbd.data.trajectory_dataset import TrajectoryDataset
from mbd.data.collect_data import collect_trajectories_from_baseline


def create_env(env_name, case="parking", config=None):
    """Create environment with given config."""
    if config is None:
        config = MBDConfig()

    return mbd.envs.get_env(
        env_name,
        case=case,
        dt=config.dt,
        H=config.Hsample,
        motion_preference=config.motion_preference,
        collision_penalty=config.collision_penalty,
        enable_shielded_rollout_collision=config.enable_shielded_rollout_collision,
        hitch_penalty=config.hitch_penalty,
        enable_shielded_rollout_hitch=config.enable_shielded_rollout_hitch,
        enable_projection=config.enable_projection,
        enable_guidance=config.enable_guidance,
        reward_threshold=config.reward_threshold,
        ref_reward_threshold=config.ref_reward_threshold,
        max_w_theta=config.max_w_theta,
        hitch_angle_weight=config.hitch_angle_weight,
        l1=config.l1, l2=config.l2, lh=config.lh,
        lf1=config.lf1, lr=config.lr, lf2=config.lf2, lr2=config.lr2,
        tractor_width=config.tractor_width, trailer_width=config.trailer_width,
        v_max=config.v_max, delta_max_deg=config.delta_max_deg,
        a_max=config.a_max, omega_max=config.omega_max,
        d_thr_factor=config.d_thr_factor, k_switch=config.k_switch,
        steering_weight=config.steering_weight,
        preference_penalty_weight=config.preference_penalty_weight,
        heading_reward_weight=config.heading_reward_weight,
        terminal_reward_threshold=config.terminal_reward_threshold,
        terminal_reward_weight=config.terminal_reward_weight,
        ref_pos_weight=config.ref_pos_weight,
        ref_theta1_weight=config.ref_theta1_weight,
        ref_theta2_weight=config.ref_theta2_weight,
        num_trailers=config.num_trailers,
    )


def run_single_trial(planner_fn, env, config, trial_seed, **planner_kwargs):
    """Run a single planning trial and return metrics.

    Args:
        planner_fn: Planning function (run_diffusion, run_diffusion_bsd, etc.)
        env: Environment.
        config: MBDConfig.
        trial_seed: Seed for this trial.
        **planner_kwargs: Extra kwargs for the planner (dataset, bsd_config, etc.)

    Returns:
        dict with metrics.
    """
    # Set seed
    trial_config = MBDConfig(
        **{k: getattr(config, k) for k in config.__dataclass_fields__}
    )
    trial_config.seed = trial_seed
    trial_config.render = False
    trial_config.show_animation = False
    trial_config.save_animation = False
    trial_config.save_denoising_animation = False

    try:
        rew_final, Y0, trajectory_states, timing_info = planner_fn(
            args=trial_config, env=env, **planner_kwargs
        )

        # Compute safety metrics
        states_np = np.asarray(trajectory_states)
        n_collisions = 0
        n_hitch_violations = 0

        for t in range(states_np.shape[0]):
            state_t = states_np[t]
            # Collision check
            collision = env.check_obstacle_collision(
                state_t, env.obs_circles, env.obs_rectangles
            )
            if collision:
                n_collisions += 1

            # Hitch violation check (skip for bicycle)
            if hasattr(env, 'check_hitch_violation'):
                hitch = env.check_hitch_violation(state_t)
                if hitch:
                    n_hitch_violations += 1

        # Goal distance
        final_state = states_np[-1]
        goal = np.asarray(env.xg)
        goal_dist = np.linalg.norm(final_state[:2] - goal[:2])

        # Success criterion: within 2.0m of goal with no collisions
        success = (goal_dist < 2.0) and (n_collisions == 0)

        return {
            'reward': float(rew_final),
            'goal_distance': float(goal_dist),
            'success': bool(success),
            'n_collisions': n_collisions,
            'n_hitch_violations': n_hitch_violations,
            'safe': n_collisions == 0,
            'planning_time_ms': float(timing_info.get('pure_diffusion_time', 0)) * 1000,
            'total_time_ms': float(timing_info.get('total_time', 0)) * 1000,
            **{k: v for k, v in timing_info.items() if k.startswith('bsd_')},
        }

    except Exception as e:
        logging.warning(f"Trial {trial_seed} failed: {e}")
        return {
            'reward': float('-inf'),
            'goal_distance': float('inf'),
            'success': False,
            'n_collisions': -1,
            'n_hitch_violations': -1,
            'safe': False,
            'planning_time_ms': 0.0,
            'total_time_ms': 0.0,
            'error': str(e),
        }


def run_experiment(
    env_name: str = "tt2d",
    case: str = "parking",
    n_trials: int = 100,
    n_data: int = 500,
    conditions: list = None,
    output_dir: str = "results",
    base_seed: int = 42,
    num_trailers: int = 1,
):
    """Run full experiment for one (system, scenario) combination.

    Args:
        env_name: Dynamical system ("kinematic_bicycle2d", "tt2d", "acc_tt2d", "n_trailer2d").
        case: Scenario ("parking" or "navigation").
        n_trials: Number of trials per condition.
        n_data: Number of trajectories in dataset.
        conditions: List of condition names to run.
        output_dir: Where to save results.
        base_seed: Base random seed.
        num_trailers: Number of trailers (relevant for n_trailer2d).
    """
    if conditions is None:
        conditions = ["mbd", "bsd", "bsd_fix", "nn"]

    config = MBDConfig(
        env_name=env_name, case=case, render=False,
        show_animation=False, save_animation=False,
        save_denoising_animation=False,
        motion_preference=1,  # Forward preference
        Nsample=20000,
        Hsample=50,
        Ndiffuse=100,
        enable_shielded_rollout_collision=True,
        enable_shielded_rollout_hitch=True,
        terminal_reward_weight=5.0,
        terminal_reward_threshold=20.0,
        temp_sample=0.00001,
        steering_weight=0.01,
        reward_threshold=50.0,
        k_switch=3.0,
        hitch_angle_weight=0.01,
        d_thr_factor=1.0,
        num_trailers=num_trailers,
    )

    exp_dir = os.path.join(output_dir, env_name, case)
    os.makedirs(exp_dir, exist_ok=True)

    # --- Step 1: Collect trajectory data (needed for BSD conditions) ---
    data_conditions = [c for c in conditions if c in ("bsd", "bsd_fix", "nn")]
    dataset = None
    data_path = os.path.join(exp_dir, "dataset")

    if data_conditions:
        if os.path.exists(os.path.join(data_path, "controls.npy")):
            logging.info(f"Loading existing dataset from {data_path}")
            dataset = TrajectoryDataset.load(data_path)
        else:
            logging.info(f"Collecting {n_data} trajectories for dataset...")

            def env_factory():
                env = create_env(env_name, case, config)
                env.set_goal_pos(
                    theta1=-jnp.pi / 2, theta2=-jnp.pi / 2
                )
                return env

            dataset = collect_trajectories_from_baseline(
                run_diffusion, env_factory, config,
                n_trajectories=n_data, seed=base_seed + 10000,
                filter_failures=True, min_reward=0.0,
            )
            dataset.save(data_path)
        logging.info(f"Dataset stats: {dataset.stats()}")

    bsd_config = BSDConfig()

    # --- Step 2: Run trials for each condition ---
    all_results = {}

    for condition in conditions:
        logging.info(f"\n{'='*60}")
        logging.info(f"Running condition: {condition} ({env_name}/{case})")
        logging.info(f"{'='*60}")

        cond_dir = os.path.join(exp_dir, condition)
        os.makedirs(cond_dir, exist_ok=True)

        results = []
        clear_jit_cache()

        for trial in tqdm(range(n_trials), desc=f"{condition}"):
            trial_seed = base_seed + trial

            # Create fresh env for each trial
            env = create_env(env_name, case, config)
            # Randomize init position
            rng = jax.random.PRNGKey(trial_seed)
            dx = float(jax.random.uniform(rng, minval=-12.0, maxval=12.0))
            dy = float(jax.random.uniform(
                jax.random.fold_in(rng, 1), minval=1.0, maxval=10.0
            ))
            theta1 = float(jax.random.uniform(
                jax.random.fold_in(rng, 2), minval=-0.5, maxval=0.5
            ))
            theta2 = float(jax.random.uniform(
                jax.random.fold_in(rng, 3), minval=-0.3, maxval=0.3
            ))
            env.set_init_pos(dx=dx, dy=dy, theta1=theta1, theta2=theta2)
            env.set_goal_pos(
                theta1=-jnp.pi / 2, theta2=-jnp.pi / 2
            )

            if condition == "mbd":
                metrics = run_single_trial(
                    run_diffusion, env, config, trial_seed
                )
            elif condition == "bsd":
                metrics = run_single_trial(
                    run_diffusion_bsd, env, config, trial_seed,
                    dataset=dataset, bsd_config=bsd_config
                )
            elif condition == "bsd_fix":
                metrics = run_single_trial(
                    run_diffusion_bsd_fixed_bandwidth, env, config, trial_seed,
                    dataset=dataset, bsd_config=bsd_config
                )
            elif condition == "nn":
                metrics = run_single_trial(
                    run_nearest_neighbor, env, config, trial_seed,
                    dataset=dataset
                )
            else:
                raise ValueError(f"Unknown condition: {condition}")

            metrics['trial'] = trial
            metrics['seed'] = trial_seed
            results.append(metrics)

            # Save individual trial
            np.savez(
                os.path.join(cond_dir, f"trial_{trial:04d}.npz"),
                **{k: np.array(v) for k, v in metrics.items()
                   if not isinstance(v, str)}
            )

        all_results[condition] = results

        # Compute and log summary
        successes = [r['success'] for r in results]
        safe_count = [r['safe'] for r in results]
        rewards = [r['reward'] for r in results if r['reward'] > float('-inf')]
        times = [r['planning_time_ms'] for r in results]

        logging.info(f"\n--- {condition} Summary ---")
        logging.info(f"Success rate: {sum(successes)/len(successes)*100:.1f}%")
        logging.info(f"Safety rate:  {sum(safe_count)/len(safe_count)*100:.1f}%")
        if rewards:
            logging.info(f"Reward:       {np.mean(rewards):.3f} ± {np.std(rewards):.3f}")
        logging.info(f"Plan time:    {np.mean(times):.0f} ± {np.std(times):.0f} ms")

    # --- Step 3: Save combined summary ---
    summary = {}
    for cond, results in all_results.items():
        successes = [r['success'] for r in results]
        safe_count = [r['safe'] for r in results]
        rewards = [r['reward'] for r in results if r['reward'] > float('-inf')]
        times = [r['planning_time_ms'] for r in results]

        summary[cond] = {
            'n_trials': len(results),
            'success_rate': sum(successes) / len(successes),
            'safety_rate': sum(safe_count) / len(safe_count),
            'reward_mean': float(np.mean(rewards)) if rewards else None,
            'reward_std': float(np.std(rewards)) if rewards else None,
            'planning_time_mean_ms': float(np.mean(times)),
            'planning_time_std_ms': float(np.std(times)),
        }

    with open(os.path.join(exp_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    logging.info(f"\nResults saved to {exp_dir}")
    return all_results, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BSD Experiments")
    parser.add_argument("--env", default="tt2d", choices=["kinematic_bicycle2d", "tt2d", "acc_tt2d", "n_trailer2d"])
    parser.add_argument("--case", default="parking", choices=["parking", "navigation"])
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--n-data", type=int, default=500)
    parser.add_argument("--conditions", nargs="+", default=["mbd", "bsd", "bsd_fix", "nn"])
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger('jax').setLevel(logging.WARNING)

    run_experiment(
        env_name=args.env,
        case=args.case,
        n_trials=args.n_trials,
        n_data=args.n_data,
        conditions=args.conditions,
        output_dir=args.output_dir,
        base_seed=args.seed,
    )
