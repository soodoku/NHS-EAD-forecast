"""Main forecasting pipeline for NHS EAD forecasting."""

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from .data_loader import load_data, split_target_features
from .evaluation import (
    compute_horizon_mse,
    compute_overall_mse,
    create_mse_summary_df,
    create_pred_matrix_df,
)
from .models import ForecastModel, get_model
from .preprocessing import (
    create_bank_holiday_features,
    create_day_of_week_features,
    create_domain_pca_features,
    create_exogenous_lag_features,
    create_lag_features,
    create_rolling_features,
    merge_target_features,
    preprocess_features,
    preprocess_target,
    select_upstream_features,
)


TARGET_COL = "estimated_avoidable_deaths"
HORIZON = 10
TARGET_LAG = 3
TRAIN_WINDOW = 90


def prepare_data(zip_path: str | Path) -> pd.DataFrame:
    """Load and preprocess all data.

    Args:
        zip_path: Path to the data zip file.

    Returns:
        Merged and preprocessed DataFrame.
    """
    print("Loading data...")
    raw_df = load_data(zip_path)
    target_df, features_df = split_target_features(raw_df)

    print("Preprocessing target...")
    target_clean = preprocess_target(target_df)

    print("Preprocessing features...")
    features_clean = preprocess_features(features_df)

    print("Merging target and features...")
    merged = merge_target_features(target_clean, features_clean)

    return merged


def engineer_features(
    df: pd.DataFrame,
    use_target_lags: bool = True,
    use_rolling: bool = True,
    use_dow: bool = True,
    use_calendar: bool = False,
    exog_lag_cols: list[str] | None = None,
    feature_cols: list[str] | None = None,
    use_domain_pca: bool = False,
) -> pd.DataFrame:
    """Create engineered features.

    Args:
        df: Merged DataFrame with target and features.
        use_target_lags: Whether to add target lag features.
        use_rolling: Whether to add rolling features.
        use_dow: Whether to add day-of-week features.
        use_calendar: Whether to add bank holiday features.
        exog_lag_cols: Exogenous columns to create lag features for.
        feature_cols: Specific feature columns to create rolling features for.
                      If None, uses a default set of key ED metrics.
        use_domain_pca: Whether to add domain PCA features.

    Returns:
        DataFrame with engineered features.
    """
    df = df.copy()

    if use_target_lags:
        df = create_lag_features(df, TARGET_COL, lags=[3, 4, 5, 6, 7])
        df = create_rolling_features(df, [f"{TARGET_COL}_lag3"], windows=[7])

    if use_rolling and feature_cols:
        df = create_rolling_features(df, feature_cols, windows=[7])

    if use_dow:
        df = create_day_of_week_features(df)

    if use_calendar:
        df = create_bank_holiday_features(df)

    if exog_lag_cols:
        df = create_exogenous_lag_features(df, exog_lag_cols, lags=[1, 2])

    if use_domain_pca:
        df = create_domain_pca_features(df)

    return df


def get_feature_columns(df: pd.DataFrame, exclude_patterns: list[str] | None = None) -> list[str]:
    """Get list of feature columns excluding target and date.

    Args:
        df: DataFrame to extract columns from.
        exclude_patterns: Additional patterns to exclude (substring match).

    Returns:
        List of feature column names.
    """
    if exclude_patterns is None:
        exclude_patterns = []
    exclude = ["midday_day", TARGET_COL]
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if any(p in c for p in exclude_patterns):
            continue
        cols.append(c)
    return cols


def run_rolling_forecast(
    df: pd.DataFrame,
    model_type: str = "ridge",
    train_window: int = TRAIN_WINDOW,
    horizon: int = HORIZON,
    feature_cols: list[str] | None = None,
    model_kwargs: dict | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    verbose: bool = True,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[pd.Timestamp]]:
    """Run rolling window forecast.

    Args:
        df: DataFrame with engineered features.
        model_type: Type of model to use ('naive', 'ridge', 'elasticnet').
        train_window: Number of days in training window.
        horizon: Forecast horizon (days).
        feature_cols: Columns to use as features. If None, all except target/date.
        model_kwargs: Additional keyword arguments for model.
        start_date: Start date for forecasting (origin date). If None, starts after train_window.
        end_date: End date for forecasting (last origin date).
        verbose: Whether to print progress.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (pred_matrix, actual_matrix, origin_dates) where matrices are (n_forecasts, horizon).
    """
    df = df.dropna().reset_index(drop=True)

    if feature_cols is None:
        feature_cols = get_feature_columns(df)

    if model_kwargs is None:
        model_kwargs = {}

    dates = df["midday_day"]

    if start_date is not None:
        start_idx = df[dates >= pd.Timestamp(start_date)].index.min()
    else:
        start_idx = train_window

    if end_date is not None:
        end_idx = df[dates <= pd.Timestamp(end_date)].index.max()
    else:
        end_idx = len(df) - horizon

    n_forecasts = end_idx - start_idx + 1
    if n_forecasts <= 0:
        raise ValueError(f"No forecasts possible with given date range. start_idx={start_idx}, end_idx={end_idx}")

    pred_matrix = np.zeros((n_forecasts, horizon))
    actual_matrix = np.zeros((n_forecasts, horizon))
    origin_dates = []

    target_lag_cols = [c for c in feature_cols if TARGET_COL in c and "lag" in c]

    for i, origin_idx in enumerate(range(start_idx, end_idx + 1)):
        if verbose and i % 50 == 0:
            print(f"Forecast {i+1}/{n_forecasts}")

        train_start = max(0, origin_idx - train_window)
        train_end = origin_idx
        test_start = origin_idx
        test_end = min(origin_idx + horizon, len(df))

        train_data = df.iloc[train_start:train_end]
        test_data = df.iloc[test_start:test_end].copy()

        if len(test_data) < horizon:
            actual_horizon = len(test_data)
        else:
            actual_horizon = horizon

        valid_features = [c for c in feature_cols if c in train_data.columns]
        valid_features = [c for c in valid_features if train_data[c].std() > 1e-10]

        if len(valid_features) == 0:
            raise ValueError("No valid features with non-zero variance in training data.")

        origin_row = df.iloc[origin_idx - 1]
        for lag_col in target_lag_cols:
            if lag_col in valid_features:
                test_data[lag_col] = origin_row[lag_col]

        X_train = train_data[valid_features].values
        y_train = train_data[TARGET_COL].values
        X_test = test_data[valid_features].values
        y_test = test_data[TARGET_COL].values

        model_kwargs_with_seed = {**model_kwargs, "seed": seed}
        model = get_model(model_type, **model_kwargs_with_seed)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        pred_matrix[i, :actual_horizon] = preds[:actual_horizon]
        actual_matrix[i, :actual_horizon] = y_test[:actual_horizon]

        if actual_horizon < horizon:
            pred_matrix[i, actual_horizon:] = preds[-1]
            actual_matrix[i, actual_horizon:] = np.nan

        origin_dates.append(dates.iloc[origin_idx])

    return pred_matrix, actual_matrix, origin_dates


EXOG_LAG_KEYWORDS = ["dta_", "ambulance_handover", "nctr_", "4hr_breach", "opel_"]


def run_forecast_pipeline(
    data_path: str | Path,
    output_dir: str | Path,
    model_type: str = "ridge",
    phase: Literal[
        "naive", "ar", "enhanced", "ar_calendar", "ar_exog_lags", "ar_upstream", "ar_domain_pca",
        "ar_upstream_calendar"
    ] = "ar",
    train_window: int = TRAIN_WINDOW,
    start_date: str | None = None,
    end_date: str | None = None,
    verbose: bool = True,
    seed: int = 42,
) -> tuple[float, float]:
    """Run the complete forecasting pipeline.

    Args:
        data_path: Path to the data zip file.
        output_dir: Directory to save outputs.
        model_type: Type of model ('naive', 'ridge', 'elasticnet').
        phase: Feature engineering phase ('naive', 'ar', 'enhanced', 'ar_calendar',
               'ar_exog_lags', 'ar_upstream', 'ar_domain_pca').
        train_window: Training window size.
        start_date: Start date for forecasting origin.
        end_date: End date for forecasting origin.
        verbose: Print progress.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (overall_mse_1_5, overall_mse_6_10).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = prepare_data(data_path)

    if verbose:
        print(f"Data loaded: {len(df)} days from {df['midday_day'].min()} to {df['midday_day'].max()}")

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

    exog_lag_cols = None
    if phase == "ar_exog_lags":
        exog_lag_cols = [
            c for c in df.columns
            if any(kw in c.lower() for kw in EXOG_LAG_KEYWORDS)
            and "lag" not in c.lower()
        ]
        if verbose:
            print(f"Selected {len(exog_lag_cols)} exogenous columns for lagging")
    elif phase in ["ar_upstream", "ar_upstream_calendar"]:
        exog_lag_cols = select_upstream_features(df)
        if verbose:
            print(f"Selected {len(exog_lag_cols)} upstream pressure columns for lagging")

    if verbose:
        print(f"Engineering features (phase={phase})...")

    use_calendar = phase in ["ar_calendar", "ar_upstream_calendar"]
    use_dow = phase in ["ar", "enhanced", "ar_calendar", "ar_exog_lags", "ar_upstream", "ar_domain_pca", "ar_upstream_calendar"]
    use_domain_pca = phase == "ar_domain_pca"

    df = engineer_features(
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
        feature_cols = get_feature_columns(df)

    y_range = df[TARGET_COL].dropna()
    clip_range = (0.0, y_range.max() * 1.5)
    model_kwargs = {"clip_range": clip_range} if model_type != "naive" else {}

    if verbose:
        print(f"Running rolling forecast with {len(feature_cols)} features...")

    pred_matrix, actual_matrix, _ = run_rolling_forecast(
        df=df,
        model_type=model_type,
        train_window=train_window,
        feature_cols=feature_cols,
        model_kwargs=model_kwargs,
        start_date=start_date,
        end_date=end_date,
        verbose=verbose,
        seed=seed,
    )

    mse_1_5, mse_6_10 = compute_horizon_mse(pred_matrix, actual_matrix)
    overall_1_5, overall_6_10 = compute_overall_mse(pred_matrix, actual_matrix)

    if verbose:
        print(f"\nOverall MSE (1-5 days): {overall_1_5:.6f}")
        print(f"Overall MSE (6-10 days): {overall_6_10:.6f}")

    pred_df = create_pred_matrix_df(pred_matrix)
    mse_df = create_mse_summary_df(mse_1_5, mse_6_10)

    pred_df.to_csv(output_dir / "pred_matrix.csv", index=False)
    mse_df.to_csv(output_dir / "mse_summary.csv", index=False)

    if verbose:
        print(f"\nOutputs saved to {output_dir}")
        print(f"  - pred_matrix.csv ({len(pred_df)} forecasts)")
        print(f"  - mse_summary.csv")

    return overall_1_5, overall_6_10
