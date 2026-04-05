#!/usr/bin/env python
"""Single experiment runner for NHS EAD forecasting."""

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.pipeline import prepare_data, engineer_features, get_feature_columns, run_rolling_forecast
from src.preprocessing import select_upstream_features
from src.evaluation import compute_overall_mse
from src.models import ProphetModel


EXPERIMENTS_DIR = Path(__file__).parent / "experiments"
RESULTS_FILE = EXPERIMENTS_DIR / "results.tsv"
CONFIGS_DIR = EXPERIMENTS_DIR / "configs"
RUNS_DIR = EXPERIMENTS_DIR / "runs"
DATA_PATH = Path(__file__).parent / "data" / "turingAI_forecasting_challenge_dataset.csv.zip"

TARGET_COL = "estimated_avoidable_deaths"
HORIZON = 10
TARGET_LAG = 3


def run_prophet_rolling_forecast(
    df: pd.DataFrame,
    model_kwargs: dict,
    start_date: str,
    end_date: str,
    horizon: int = HORIZON,
    target_lag: int = TARGET_LAG,
    verbose: bool = True,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Run rolling window forecast with Prophet.

    Prophet requires special handling due to the 3-day target lag constraint:
    - At forecast origin t (origin_idx), we want to predict days t through t+9
    - But we only know y values through t-3 (target_lag=3)
    - Train Prophet on data through t-3
    - Forecast (target_lag - 1 + horizon) = 12 steps ahead: t-2, t-1, t, t+1, ..., t+9
    - Take steps starting from index (target_lag - 1) as predictions for t through t+9
    """
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    dates = pd.to_datetime(df["midday_day"])

    start_idx = int(df[dates >= pd.Timestamp(start_date)].index.min())
    end_idx = int(df[dates <= pd.Timestamp(end_date)].index.max())

    n_forecasts = end_idx - start_idx + 1
    if n_forecasts <= 0:
        raise ValueError("No forecasts possible with given date range.")

    pred_matrix = np.zeros((n_forecasts, horizon))
    actual_matrix = np.zeros((n_forecasts, horizon))

    for i, origin_idx in enumerate(range(start_idx, end_idx + 1)):
        if verbose and i % 50 == 0:
            print(f"Prophet forecast {i+1}/{n_forecasts}")

        train_end_idx = origin_idx - target_lag
        if train_end_idx < 30:
            continue

        train_dates = dates.iloc[:train_end_idx + 1]
        train_y = df[TARGET_COL].iloc[:train_end_idx + 1].values

        model = ProphetModel(seed=seed, **model_kwargs)
        model.fit_prophet(train_dates, train_y)

        future_start = dates.iloc[train_end_idx] + pd.Timedelta(days=1)
        n_future = (target_lag - 1) + horizon
        future_dates = pd.date_range(
            start=future_start,
            periods=n_future,
            freq="D",
        )

        preds = model.predict_prophet(future_dates)
        preds_for_horizon = preds[(target_lag - 1):]

        test_start_idx = origin_idx
        test_end_idx = min(origin_idx + horizon, len(df))
        test_data = df.iloc[test_start_idx:test_end_idx]
        actual_horizon = len(test_data)

        pred_matrix[i, :actual_horizon] = preds_for_horizon[:actual_horizon]
        actual_matrix[i, :actual_horizon] = test_data[TARGET_COL].values[:actual_horizon]

        if actual_horizon < horizon:
            pred_matrix[i, actual_horizon:] = preds_for_horizon[-1]
            actual_matrix[i, actual_horizon:] = np.nan

    return pred_matrix, actual_matrix


def get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return "unknown"


def get_next_run_id() -> int:
    """Get next sequential run ID from results.tsv."""
    if not RESULTS_FILE.exists():
        return 1
    df = pd.read_csv(RESULTS_FILE, sep="\t")
    if len(df) == 0:
        return 1
    return int(df["run_id"].max()) + 1


def load_config(config_path: Path) -> dict:
    """Load experiment configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_single_fold(
    df: pd.DataFrame,
    config: dict,
    start_date: str,
    end_date: str,
    verbose: bool = True,
) -> tuple[float, float, float]:
    """Run a single fold of the experiment.

    Returns:
        Tuple of (mse_1_5, mse_6_10, runtime_sec).
    """
    model_type = config["model"]["type"]
    phase = config["model"]["phase"]
    train_window = config["hyperparameters"].get("train_window", 90)
    seed = config.get("seed", 42)

    feature_cols_for_rolling = None
    if phase == "enhanced":
        key_features = [
            c for c in df.columns
            if any(
                kw in c.lower()
                for kw in ["patients_in_a_e", "dta", "breach", "ambulance"]
            )
        ]
        feature_cols_for_rolling = key_features[:10]

    exog_lag_keywords = ["dta_", "ambulance_handover", "nctr_", "4hr_breach", "opel_"]
    exog_lag_cols = None
    if phase == "ar_exog_lags":
        exog_lag_cols = [
            c for c in df.columns
            if any(kw in c.lower() for kw in exog_lag_keywords)
            and "lag" not in c.lower()
        ]
    elif phase in ["ar_upstream", "ar_upstream_calendar"]:
        exog_lag_cols = select_upstream_features(df)

    use_calendar = phase in ["ar_calendar", "ar_upstream_calendar"]
    use_dow = phase in ["ar", "enhanced", "ar_calendar", "ar_exog_lags", "ar_upstream", "ar_domain_pca", "ar_upstream_calendar"]
    use_domain_pca = phase == "ar_domain_pca"

    df_eng = engineer_features(
        df,
        use_target_lags=True,
        use_rolling=(phase == "enhanced"),
        use_dow=use_dow,
        use_calendar=use_calendar,
        exog_lag_cols=exog_lag_cols,
        feature_cols=feature_cols_for_rolling,
        use_domain_pca=use_domain_pca,
    )

    if phase == "naive":
        feature_cols = [f"{TARGET_COL}_lag3"]
    else:
        feature_cols = get_feature_columns(df_eng)

    y_range = df_eng[TARGET_COL].dropna()
    clip_range = (0.0, y_range.max() * 1.5)

    model_kwargs = {}
    if model_type != "naive":
        model_kwargs["clip_range"] = clip_range
        if "alphas" in config["hyperparameters"]:
            model_kwargs["alphas"] = config["hyperparameters"]["alphas"]
        if "l1_ratio" in config["hyperparameters"]:
            model_kwargs["l1_ratio"] = config["hyperparameters"]["l1_ratio"]
        if model_type == "gradientboosting":
            for param in ["n_estimators", "max_depth", "learning_rate",
                          "min_samples_split", "min_samples_leaf"]:
                if param in config["hyperparameters"]:
                    model_kwargs[param] = config["hyperparameters"][param]
        if model_type == "xgboost":
            for param in ["n_estimators", "max_depth", "learning_rate",
                          "subsample", "colsample_bytree", "reg_alpha", "reg_lambda"]:
                if param in config["hyperparameters"]:
                    model_kwargs[param] = config["hyperparameters"][param]
        if model_type == "mlp":
            for param in ["hidden_sizes", "learning_rate", "epochs",
                          "batch_size", "dropout"]:
                if param in config["hyperparameters"]:
                    model_kwargs[param] = config["hyperparameters"][param]
        if model_type == "prophet":
            model_kwargs = {}
            for param in ["yearly_seasonality", "weekly_seasonality", "daily_seasonality",
                          "seasonality_mode", "changepoint_prior_scale", "seasonality_prior_scale"]:
                if param in config["hyperparameters"]:
                    model_kwargs[param] = config["hyperparameters"][param]

    start_time = time.time()

    if model_type == "prophet":
        pred_matrix, actual_matrix = run_prophet_rolling_forecast(
            df=df,
            model_kwargs=model_kwargs,
            start_date=start_date,
            end_date=end_date,
            verbose=verbose,
            seed=seed,
        )
    else:
        pred_matrix, actual_matrix, _ = run_rolling_forecast(
            df=df_eng,
            model_type=model_type,
            train_window=train_window,
            feature_cols=feature_cols,
            model_kwargs=model_kwargs,
            start_date=start_date,
            end_date=end_date,
            verbose=verbose,
            seed=seed,
        )

    runtime = time.time() - start_time
    mse_1_5, mse_6_10 = compute_overall_mse(pred_matrix, actual_matrix)

    return mse_1_5, mse_6_10, runtime


def run_cv_experiment(
    df: pd.DataFrame,
    config: dict,
    verbose: bool = True,
) -> dict:
    """Run cross-validation experiment.

    CV folds (expanding window):
    Fold 1: Train [Mar 2023 - Dec 2024] -> Validate [Jan - Mar 2025]
    Fold 2: Train [Mar 2023 - Mar 2025] -> Validate [Apr - Jun 2025]
    Fold 3: Train [Mar 2023 - Jun 2025] -> Validate [Jul - Sep 2025]
    """
    cv_folds = [
        ("2025-01-01", "2025-03-31"),
        ("2025-04-01", "2025-06-30"),
        ("2025-07-01", "2025-09-30"),
    ]

    n_folds = config["evaluation"].get("cv_folds", 3)
    cv_folds = cv_folds[:n_folds]

    mse_1_5_list = []
    mse_6_10_list = []
    runtimes = []

    for i, (start_date, end_date) in enumerate(cv_folds):
        if verbose:
            print(f"\n=== Fold {i+1}/{n_folds}: {start_date} to {end_date} ===")

        mse_1_5, mse_6_10, runtime = run_single_fold(
            df=df,
            config=config,
            start_date=start_date,
            end_date=end_date,
            verbose=verbose,
        )

        mse_1_5_list.append(mse_1_5)
        mse_6_10_list.append(mse_6_10)
        runtimes.append(runtime)

        if verbose:
            print(f"Fold {i+1} MSE 1-5: {mse_1_5:.6f}, MSE 6-10: {mse_6_10:.6f}")

    return {
        "cv_mse_1_5": float(np.mean(mse_1_5_list)),
        "cv_mse_6_10": float(np.mean(mse_6_10_list)),
        "cv_std_1_5": float(np.std(mse_1_5_list)),
        "cv_std_6_10": float(np.std(mse_6_10_list)),
        "fold_mse_1_5": mse_1_5_list,
        "fold_mse_6_10": mse_6_10_list,
        "fold_runtimes": runtimes,
        "total_runtime": sum(runtimes),
    }


def run_holdout_experiment(
    df: pd.DataFrame,
    config: dict,
    verbose: bool = True,
) -> dict:
    """Run holdout experiment (faster, for quick iteration)."""
    start_date = config["evaluation"].get("start_date", "2025-07-01")
    end_date = config["evaluation"].get("end_date", "2025-09-30")

    if verbose:
        print(f"\n=== Holdout: {start_date} to {end_date} ===")

    mse_1_5, mse_6_10, runtime = run_single_fold(
        df=df,
        config=config,
        start_date=start_date,
        end_date=end_date,
        verbose=verbose,
    )

    return {
        "cv_mse_1_5": mse_1_5,
        "cv_mse_6_10": mse_6_10,
        "cv_std_1_5": 0.0,
        "cv_std_6_10": 0.0,
        "total_runtime": runtime,
    }


def append_to_results(
    run_id: int,
    commit: str,
    metrics: dict,
    config: dict,
    keep: bool = True,
) -> None:
    """Append experiment result to results.tsv."""
    timestamp = datetime.now().isoformat(timespec="seconds")
    description = config["experiment"].get("description", config["experiment"]["name"])

    row = {
        "timestamp": timestamp,
        "run_id": f"{run_id:03d}",
        "commit": commit,
        "cv_mse_1_5": f"{metrics['cv_mse_1_5']:.6f}",
        "cv_mse_6_10": f"{metrics['cv_mse_6_10']:.6f}",
        "cv_std": f"{(metrics['cv_std_1_5'] + metrics['cv_std_6_10']) / 2:.6f}",
        "runtime_sec": f"{metrics['total_runtime']:.0f}",
        "keep": "yes" if keep else "no",
        "description": description,
    }

    df = pd.DataFrame([row])

    if not RESULTS_FILE.exists():
        df.to_csv(RESULTS_FILE, sep="\t", index=False)
    else:
        df.to_csv(RESULTS_FILE, sep="\t", index=False, mode="a", header=False)


def save_run_artifacts(
    run_id: int,
    config: dict,
    metrics: dict,
    description: str,
) -> Path:
    """Save run artifacts to runs directory."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    run_dir = RUNS_DIR / f"{date_str}_{run_id:03d}_{description.replace(' ', '_')[:30]}"
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return run_dir


def run_experiment(config_path: Path, verbose: bool = True, keep: bool | None = None) -> dict:
    """Run a complete experiment from a config file.

    Args:
        config_path: Path to YAML config file.
        verbose: Whether to print progress.
        keep: Whether to mark as "keep" in results. If None, auto-determine based on improvement.

    Returns:
        Dictionary with experiment results.
    """
    config = load_config(config_path)
    run_id = get_next_run_id()
    commit = get_git_commit()

    if verbose:
        print(f"\n{'='*60}")
        print(f"Experiment: {config['experiment']['name']}")
        print(f"Run ID: {run_id:03d}")
        print(f"Commit: {commit}")
        print(f"{'='*60}")

    print("Loading and preparing data...")
    df = prepare_data(DATA_PATH)

    eval_method = config["evaluation"].get("method", "cv")

    if eval_method == "cv":
        metrics = run_cv_experiment(df, config, verbose=verbose)
    else:
        metrics = run_holdout_experiment(df, config, verbose=verbose)

    if keep is None:
        if RESULTS_FILE.exists():
            results_df = pd.read_csv(RESULTS_FILE, sep="\t")
            if len(results_df) > 0:
                best_mse = results_df[results_df["keep"] == "yes"]["cv_mse_1_5"].astype(float).min()
                keep = metrics["cv_mse_1_5"] <= best_mse
            else:
                keep = True
        else:
            keep = True

    append_to_results(run_id, commit, metrics, config, keep=keep)

    description = config["experiment"].get("description", config["experiment"]["name"])
    run_dir = save_run_artifacts(run_id, config, metrics, description)

    if verbose:
        print(f"\n{'='*60}")
        print("RESULTS:")
        print(f"  CV MSE (1-5):  {metrics['cv_mse_1_5']:.6f}")
        print(f"  CV MSE (6-10): {metrics['cv_mse_6_10']:.6f}")
        if metrics.get("cv_std_1_5", 0) > 0:
            print(f"  CV Std (1-5):  {metrics['cv_std_1_5']:.6f}")
            print(f"  CV Std (6-10): {metrics['cv_std_6_10']:.6f}")
        print(f"  Runtime:       {metrics['total_runtime']:.0f}s")
        print(f"  Keep:          {'yes' if keep else 'no'}")
        print(f"  Artifacts:     {run_dir}")
        print(f"{'='*60}")

    return {
        "run_id": run_id,
        "metrics": metrics,
        "keep": keep,
        "run_dir": str(run_dir),
    }


def main():
    parser = argparse.ArgumentParser(description="Run a single NHS EAD forecasting experiment")
    parser.add_argument(
        "config",
        type=str,
        help="Path to experiment config YAML file",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Force mark as 'keep' regardless of improvement",
    )
    parser.add_argument(
        "--no-keep",
        action="store_true",
        help="Force mark as 'no keep' regardless of improvement",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path

    keep = None
    if args.keep:
        keep = True
    elif args.no_keep:
        keep = False

    run_experiment(config_path, verbose=not args.quiet, keep=keep)


if __name__ == "__main__":
    main()
