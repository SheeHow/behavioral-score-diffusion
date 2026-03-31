"""
Master experiment runner: runs all 4 systems sequentially.

Runs each system one at a time to avoid GPU memory contention.
Saves results to the project experiment results directory.
"""

import os
import sys
import json
import time
import logging
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from run_bsd_experiments import run_experiment

RESULTS_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', '..', '..', 'results'
)

# System configurations: (env_name, num_trailers, label)
SYSTEMS = [
    ("kinematic_bicycle2d", 1, "Bicycle"),
    ("tt2d", 1, "TT2D"),
    ("acc_tt2d", 1, "AccTT2D"),
    ("tt2d", 2, "NTrailer2"),  # tt2d with num_trailers=2
]

CONDITIONS = ["mbd", "bsd", "bsd_fix", "nn"]


def run_all(
    n_trials: int = 50,
    n_data: int = 200,
    case: str = "parking",
    base_seed: int = 42,
    output_dir: str = None,
    systems: list = None,
):
    if output_dir is None:
        output_dir = os.path.abspath(RESULTS_DIR)

    os.makedirs(output_dir, exist_ok=True)

    if systems is None:
        systems = SYSTEMS

    all_summaries = {}
    total_start = time.time()

    for env_name, num_trailers, label in systems:
        sys_start = time.time()
        logging.info(f"\n{'#'*70}")
        logging.info(f"# SYSTEM: {label} (env={env_name}, trailers={num_trailers})")
        logging.info(f"{'#'*70}")

        # For n_trailer, we need to patch the config's num_trailers
        # The run_experiment function creates MBDConfig internally,
        # and NTrailer is activated via env_name="tt2d" + num_trailers>=2
        # or env_name="n_trailer2d". Let's use the n_trailer2d name for clarity.
        effective_env_name = env_name
        if num_trailers >= 2:
            effective_env_name = "n_trailer2d"

        try:
            _, summary = run_experiment(
                env_name=effective_env_name,
                case=case,
                n_trials=n_trials,
                n_data=n_data,
                conditions=CONDITIONS,
                output_dir=output_dir,
                base_seed=base_seed,
                num_trailers=num_trailers,
            )
            all_summaries[label] = summary

            sys_elapsed = time.time() - sys_start
            logging.info(f"\n{label} completed in {sys_elapsed:.0f}s ({sys_elapsed/60:.1f}min)")

        except Exception as e:
            logging.error(f"SYSTEM {label} FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_summaries[label] = {"error": str(e)}

    # Save combined summary
    total_elapsed = time.time() - total_start
    combined = {
        "metadata": {
            "n_trials": n_trials,
            "n_data": n_data,
            "case": case,
            "seed": base_seed,
            "total_time_s": total_elapsed,
            "conditions": CONDITIONS,
        },
        "results": all_summaries,
    }

    combined_path = os.path.join(output_dir, "combined_summary.json")
    with open(combined_path, 'w') as f:
        json.dump(combined, f, indent=2)

    logging.info(f"\n{'='*70}")
    logging.info(f"ALL EXPERIMENTS COMPLETED in {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")
    logging.info(f"Combined summary: {combined_path}")
    logging.info(f"{'='*70}")

    # Print results table
    print("\n" + "="*90)
    print(f"{'System':<15} {'Condition':<10} {'Success%':>10} {'Safety%':>10} "
          f"{'Reward':>12} {'PlanTime(ms)':>14}")
    print("-"*90)
    for label, summary in all_summaries.items():
        if isinstance(summary, dict) and "error" in summary:
            print(f"{label:<15} ERROR: {summary['error']}")
            continue
        for cond in CONDITIONS:
            if cond in summary:
                s = summary[cond]
                rew_str = f"{s['reward_mean']:.3f}+-{s['reward_std']:.3f}" if s.get('reward_mean') else "N/A"
                print(f"{label:<15} {cond:<10} {s['success_rate']*100:>9.1f}% "
                      f"{s['safety_rate']*100:>9.1f}% {rew_str:>12} "
                      f"{s['planning_time_mean_ms']:>10.0f}+-{s['planning_time_std_ms']:.0f}")
    print("="*90)

    return all_summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all BSD experiments")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--n-data", type=int, default=200)
    parser.add_argument("--case", default="parking")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--systems", nargs="+", default=None,
                       help="Which systems to run: bicycle tt2d acc_tt2d ntrailer")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logging.getLogger('jax').setLevel(logging.WARNING)

    systems = None
    if args.systems:
        name_map = {
            "bicycle": ("kinematic_bicycle2d", 1, "Bicycle"),
            "tt2d": ("tt2d", 1, "TT2D"),
            "acc_tt2d": ("acc_tt2d", 1, "AccTT2D"),
            "ntrailer": ("tt2d", 2, "NTrailer2"),
        }
        systems = [name_map[s] for s in args.systems if s in name_map]

    run_all(
        n_trials=args.n_trials,
        n_data=args.n_data,
        case=args.case,
        base_seed=args.seed,
        output_dir=args.output_dir,
        systems=systems,
    )
