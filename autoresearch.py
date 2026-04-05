#!/usr/bin/env python
"""Autonomous experiment loop for NHS EAD forecasting.

Inspired by Karpathy's autoresearch: https://github.com/karpathy/autoresearch

This script can run in three modes:
1. Interactive: Propose experiments for human approval
2. Batch: Run a predefined set of configs
3. Grid: Generate and run grid search experiments

The core loop:
1. Read current best from results.tsv
2. Propose next experiment
3. Run experiment
4. Evaluate: keep or discard based on improvement
5. Update research log
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from run_experiment import run_experiment, RESULTS_FILE, CONFIGS_DIR, EXPERIMENTS_DIR


def get_best_result() -> dict | None:
    """Get the best result from results.tsv."""
    if not RESULTS_FILE.exists():
        return None

    df = pd.read_csv(RESULTS_FILE, sep="\t")
    if len(df) == 0:
        return None

    kept = df[df["keep"] == "yes"]
    if len(kept) == 0:
        return None

    best_idx = kept["cv_mse_1_5"].astype(float).idxmin()
    best = kept.loc[best_idx]

    return {
        "run_id": int(best["run_id"]),
        "cv_mse_1_5": float(best["cv_mse_1_5"]),
        "cv_mse_6_10": float(best["cv_mse_6_10"]),
        "description": best["description"],
    }


def print_status() -> None:
    """Print current experiment status."""
    print("\n" + "=" * 70)
    print("NHS EAD FORECASTING - AUTORESEARCH")
    print("=" * 70)

    if not RESULTS_FILE.exists():
        print("\nNo experiments run yet. Starting fresh.")
        return

    df = pd.read_csv(RESULTS_FILE, sep="\t")
    print(f"\nTotal experiments: {len(df)}")
    print(f"Kept experiments: {len(df[df['keep'] == 'yes'])}")

    best = get_best_result()
    if best:
        print(f"\nCurrent Best:")
        print(f"  Run #{best['run_id']:03d}: {best['description']}")
        print(f"  MSE 1-5:  {best['cv_mse_1_5']:.6f}")
        print(f"  MSE 6-10: {best['cv_mse_6_10']:.6f}")

    print("\nRecent experiments:")
    recent = df.tail(5)
    for _, row in recent.iterrows():
        keep_str = "[KEEP]" if row["keep"] == "yes" else "[    ]"
        print(f"  #{int(row['run_id']):03d} {keep_str} MSE={float(row['cv_mse_1_5']):.4f} - {row['description'][:40]}")


def append_to_experiments_md(result: dict, config: dict) -> None:
    """Append experiment result to EXPERIMENTS.md."""
    experiments_md = EXPERIMENTS_DIR / "EXPERIMENTS.md"

    if not experiments_md.exists():
        return

    with open(experiments_md, "a") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        keep_str = "IMPROVED" if result["keep"] else "no improvement"
        f.write(f"\n- [{timestamp}] Run #{result['run_id']:03d}: "
                f"{config['experiment']['description']} - "
                f"MSE 1-5: {result['metrics']['cv_mse_1_5']:.6f} ({keep_str})")


def run_batch(config_paths: list[Path], verbose: bool = True) -> None:
    """Run a batch of experiments from config files."""
    print_status()

    for i, config_path in enumerate(config_paths):
        print(f"\n>>> Batch experiment {i+1}/{len(config_paths)}: {config_path.name}")

        config = yaml.safe_load(open(config_path))
        result = run_experiment(config_path, verbose=verbose)
        append_to_experiments_md(result, config)

        print_status()


def run_interactive(max_experiments: int = 10, verbose: bool = True) -> None:
    """Run experiments interactively, asking for approval.

    This mode shows the current status and waits for user to:
    1. Enter a config file path to run
    2. Type 'list' to see available configs
    3. Type 'status' to see current results
    4. Type 'quit' to exit
    """
    experiments_run = 0

    while experiments_run < max_experiments:
        print_status()

        print("\n" + "-" * 40)
        print("Commands:")
        print("  <config.yaml>  - Run experiment from config")
        print("  list           - List available configs")
        print("  status         - Show current status")
        print("  quit           - Exit")
        print("-" * 40)

        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("Exiting...")
            break

        if user_input.lower() == "status":
            continue

        if user_input.lower() == "list":
            configs = list(CONFIGS_DIR.glob("*.yaml"))
            print("\nAvailable configs:")
            for cfg in sorted(configs):
                print(f"  {cfg.name}")
            continue

        config_path = Path(user_input)
        if not config_path.is_absolute():
            if not config_path.exists():
                config_path = CONFIGS_DIR / config_path

        if not config_path.exists():
            print(f"Config not found: {config_path}")
            continue

        print(f"\nRunning experiment: {config_path.name}")
        confirm = input("Proceed? [Y/n] ").strip().lower()
        if confirm and confirm != "y":
            print("Skipped.")
            continue

        config = yaml.safe_load(open(config_path))
        result = run_experiment(config_path, verbose=verbose)
        append_to_experiments_md(result, config)
        experiments_run += 1


def generate_grid_configs(
    base_config_path: Path,
    param_grid: dict,
    output_dir: Path,
) -> list[Path]:
    """Generate grid search configs from a base config.

    Args:
        base_config_path: Path to base YAML config.
        param_grid: Dict of param paths to lists of values.
                   e.g., {"hyperparameters.train_window": [60, 90, 120]}
        output_dir: Where to save generated configs.

    Returns:
        List of generated config paths.
    """
    import itertools

    base_config = yaml.safe_load(open(base_config_path))

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))

    output_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    for i, combo in enumerate(combinations):
        config = base_config.copy()

        param_strs = []
        for key, val in zip(keys, combo):
            parts = key.split(".")
            target = config
            for part in parts[:-1]:
                target = target[part]
            target[parts[-1]] = val
            param_strs.append(f"{parts[-1]}={val}")

        config["experiment"]["name"] = f"grid_{i+1:03d}"
        config["experiment"]["description"] = f"Grid: {', '.join(param_strs)}"

        output_path = output_dir / f"grid_{i+1:03d}.yaml"
        with open(output_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        generated.append(output_path)

    return generated


def run_grid(
    base_config: Path,
    param_grid: dict,
    verbose: bool = True,
) -> None:
    """Run grid search from a base config.

    Args:
        base_config: Path to base config YAML.
        param_grid: Dict of parameters to grid search.
        verbose: Whether to print progress.
    """
    grid_dir = CONFIGS_DIR / "grid"
    configs = generate_grid_configs(base_config, param_grid, grid_dir)

    print(f"\nGenerated {len(configs)} grid configs in {grid_dir}")
    print("Configs:")
    for cfg in configs:
        print(f"  {cfg.name}")

    confirm = input("\nRun all? [Y/n] ").strip().lower()
    if confirm and confirm != "y":
        print("Aborted.")
        return

    run_batch(configs, verbose=verbose)


def main():
    parser = argparse.ArgumentParser(
        description="Autonomous experiment loop for NHS EAD forecasting"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    batch_parser = subparsers.add_parser("batch", help="Run batch of experiments")
    batch_parser.add_argument(
        "configs",
        nargs="+",
        type=str,
        help="Config files to run",
    )
    batch_parser.add_argument("--quiet", action="store_true", help="Suppress output")

    interactive_parser = subparsers.add_parser("interactive", help="Run interactively")
    interactive_parser.add_argument(
        "--max-experiments",
        type=int,
        default=10,
        help="Max experiments to run",
    )
    interactive_parser.add_argument("--quiet", action="store_true", help="Suppress output")

    grid_parser = subparsers.add_parser("grid", help="Run grid search")
    grid_parser.add_argument("base_config", type=str, help="Base config file")
    grid_parser.add_argument(
        "--train-window",
        nargs="+",
        type=int,
        help="Train window values to try",
    )
    grid_parser.add_argument(
        "--model",
        nargs="+",
        type=str,
        help="Model types to try",
    )
    grid_parser.add_argument("--quiet", action="store_true", help="Suppress output")

    status_parser = subparsers.add_parser("status", help="Show current status")

    baselines_parser = subparsers.add_parser("baselines", help="Run all baseline experiments")
    baselines_parser.add_argument("--quiet", action="store_true", help="Suppress output")

    args = parser.parse_args()

    if args.command == "status" or args.command is None:
        print_status()
        return

    if args.command == "batch":
        configs = [Path(c) for c in args.configs]
        run_batch(configs, verbose=not args.quiet)

    elif args.command == "interactive":
        run_interactive(
            max_experiments=args.max_experiments,
            verbose=not args.quiet,
        )

    elif args.command == "grid":
        param_grid = {}
        if args.train_window:
            param_grid["hyperparameters.train_window"] = args.train_window
        if args.model:
            param_grid["model.type"] = args.model

        if not param_grid:
            print("Error: No grid parameters specified. Use --train-window or --model")
            sys.exit(1)

        run_grid(
            Path(args.base_config),
            param_grid,
            verbose=not args.quiet,
        )

    elif args.command == "baselines":
        baseline_configs = sorted(CONFIGS_DIR.glob("baseline_*.yaml"))
        if not baseline_configs:
            print("No baseline configs found in experiments/configs/")
            print("Create baseline_*.yaml files first.")
            sys.exit(1)
        run_batch(baseline_configs, verbose=not args.quiet)


if __name__ == "__main__":
    main()
