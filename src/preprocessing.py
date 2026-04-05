"""Preprocessing utilities for NHS EAD forecasting."""

import datetime

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def apply_midday_threshold(df: pd.DataFrame) -> pd.DataFrame:
    """Apply midday threshold: times <= 12:00 map to same day, > 12:00 to next day.

    Args:
        df: DataFrame with 'dt' datetime column.

    Returns:
        DataFrame with new 'midday_day' column (date).
    """
    df = df.copy()
    df["date"] = df["dt"].dt.date
    df["time"] = df["dt"].dt.time
    midday = pd.Timestamp("12:00:00").time()
    df["midday_day"] = df.apply(
        lambda row: row["date"]
        if row["time"] <= midday
        else (pd.Timestamp(row["date"]) + pd.Timedelta(days=1)).date(),
        axis=1,
    )
    df["midday_day"] = pd.to_datetime(df["midday_day"])
    return df


def aggregate_to_daily(df: pd.DataFrame, value_col: str = "value") -> pd.DataFrame:
    """Aggregate data to daily resolution by taking mean per metric per day.

    Args:
        df: DataFrame with 'midday_day' and 'metric_name' columns.
        value_col: Name of the value column.

    Returns:
        Wide-format DataFrame with midday_day as index, metrics as columns.
    """
    daily = (
        df.groupby(["midday_day", "metric_name"])[value_col]
        .mean()
        .reset_index()
        .pivot(index="midday_day", columns="metric_name", values=value_col)
    )
    daily = daily.reset_index()
    daily.columns.name = None
    return daily


def clean_column_names(df: pd.DataFrame, exclude_cols: list[str] | None = None) -> pd.DataFrame:
    """Clean and standardize column names, ensuring uniqueness.

    Args:
        df: DataFrame with columns to clean.
        exclude_cols: Columns to leave unchanged.

    Returns:
        DataFrame with cleaned, unique column names.
    """
    if exclude_cols is None:
        exclude_cols = []
    df = df.copy()

    new_names = []
    seen = {}
    for col in df.columns:
        if col in exclude_cols:
            new_names.append(col)
            continue
        new_name = col.lower()
        new_name = new_name.replace("%", "pct")
        new_name = "".join(c if c.isalnum() or c == "_" else "_" for c in new_name)
        new_name = "_".join(filter(None, new_name.split("_")))

        if new_name in seen:
            seen[new_name] += 1
            new_name = f"{new_name}_{seen[new_name]}"
        else:
            seen[new_name] = 0
        new_names.append(new_name)

    df.columns = new_names
    return df


def impute_missing(df: pd.DataFrame, exclude_cols: list[str] | None = None) -> pd.DataFrame:
    """Impute missing values using linear interpolation, then forward/backward fill.

    Args:
        df: DataFrame with potential missing values.
        exclude_cols: Columns to skip imputation.

    Returns:
        DataFrame with imputed values.
    """
    if exclude_cols is None:
        exclude_cols = []
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_impute = [c for c in numeric_cols if c not in exclude_cols]
    for col in cols_to_impute:
        df[col] = df[col].interpolate(method="linear", limit_direction="both")
        df[col] = df[col].ffill().bfill()
    return df


def create_rolling_features(
    df: pd.DataFrame, cols: list[str], windows: list[int] | None = None
) -> pd.DataFrame:
    """Create rolling mean and std features for specified columns.

    Args:
        df: DataFrame sorted by date.
        cols: Columns to create rolling features for.
        windows: List of window sizes (default: [7]).

    Returns:
        DataFrame with added rolling features.
    """
    if windows is None:
        windows = [7]
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        for w in windows:
            df[f"{col}_roll_mean_{w}"] = df[col].rolling(window=w, min_periods=1).mean()
            df[f"{col}_roll_std_{w}"] = df[col].rolling(window=w, min_periods=1).std()
    return df


def create_lag_features(df: pd.DataFrame, col: str, lags: list[int]) -> pd.DataFrame:
    """Create lag features for a column.

    Args:
        df: DataFrame sorted by date.
        col: Column to create lags for.
        lags: List of lag values.

    Returns:
        DataFrame with added lag features.
    """
    df = df.copy()
    for lag in lags:
        df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df


def create_day_of_week_features(df: pd.DataFrame, date_col: str = "midday_day") -> pd.DataFrame:
    """Create day-of-week one-hot encoded features.

    Args:
        df: DataFrame with date column.
        date_col: Name of date column.

    Returns:
        DataFrame with added day-of-week features.
    """
    df = df.copy()
    dow = pd.to_datetime(df[date_col]).dt.dayofweek
    for i in range(7):
        df[f"dow_{i}"] = (dow == i).astype(int)
    return df


def preprocess_target(target_df: pd.DataFrame, filter_dummy: bool = True) -> pd.DataFrame:
    """Preprocess target (outcome) data.

    Target may have multiple readings per day - aggregate to daily mean.

    Args:
        target_df: Target DataFrame from split_target_features.
        filter_dummy: Whether to filter out -9999 dummy values (assessment period).

    Returns:
        DataFrame with columns: midday_day, estimated_avoidable_deaths.
    """
    df = target_df.copy()

    if filter_dummy:
        df = df[df["value"] != -9999]

    df["midday_day"] = pd.to_datetime(df["dt"], format="mixed").dt.normalize()

    df = (
        df.groupby("midday_day")["value"]
        .mean()
        .reset_index()
        .rename(columns={"value": "estimated_avoidable_deaths"})
    )
    df = df.sort_values("midday_day").reset_index(drop=True)
    return df


def preprocess_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess feature data.

    Args:
        features_df: Features DataFrame from split_target_features.

    Returns:
        Wide-format daily DataFrame with cleaned column names.
    """
    df = apply_midday_threshold(features_df)
    df = aggregate_to_daily(df)
    df = clean_column_names(df, exclude_cols=["midday_day"])
    df = impute_missing(df, exclude_cols=["midday_day"])
    df = df.sort_values("midday_day").reset_index(drop=True)
    return df


def merge_target_features(target_df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
    """Merge target and features on midday_day.

    Args:
        target_df: Preprocessed target DataFrame.
        features_df: Preprocessed features DataFrame.

    Returns:
        Merged DataFrame.
    """
    merged = pd.merge(target_df, features_df, on="midday_day", how="inner")
    return merged.sort_values("midday_day").reset_index(drop=True)


def get_uk_bank_holidays(start_year: int = 2020, end_year: int = 2030) -> set[datetime.date]:
    """Generate UK (England & Wales) bank holidays for a range of years.

    Args:
        start_year: First year to include.
        end_year: Last year to include.

    Returns:
        Set of date objects representing bank holidays.
    """
    holidays = set()

    for year in range(start_year, end_year + 1):
        holidays.add(datetime.date(year, 1, 1))

        a = year % 19
        b = year // 100
        c = year % 100
        d = b // 4
        e = b % 4
        f = (b + 8) // 25
        g = (b - f + 1) // 3
        h = (19 * a + b - d - g + 15) % 30
        i = c // 4
        k = c % 4
        l = (32 + 2 * e + 2 * i - h - k) % 7
        m = (a + 11 * h + 22 * l) // 451
        month = (h + l - 7 * m + 114) // 31
        day = ((h + l - 7 * m + 114) % 31) + 1
        easter = datetime.date(year, month, day)

        good_friday = easter - datetime.timedelta(days=2)
        easter_monday = easter + datetime.timedelta(days=1)
        holidays.add(good_friday)
        holidays.add(easter_monday)

        may_first = datetime.date(year, 5, 1)
        days_until_monday = (7 - may_first.weekday()) % 7
        early_may = may_first + datetime.timedelta(days=days_until_monday)
        holidays.add(early_may)

        may_31 = datetime.date(year, 5, 31)
        days_since_monday = may_31.weekday()
        spring_bank = may_31 - datetime.timedelta(days=days_since_monday)
        holidays.add(spring_bank)

        aug_31 = datetime.date(year, 8, 31)
        days_since_monday = aug_31.weekday()
        summer_bank = aug_31 - datetime.timedelta(days=days_since_monday)
        holidays.add(summer_bank)

        christmas = datetime.date(year, 12, 25)
        boxing = datetime.date(year, 12, 26)
        if christmas.weekday() == 5:
            holidays.add(datetime.date(year, 12, 27))
            holidays.add(datetime.date(year, 12, 28))
        elif christmas.weekday() == 6:
            holidays.add(datetime.date(year, 12, 26))
            holidays.add(datetime.date(year, 12, 27))
        else:
            holidays.add(christmas)
            if boxing.weekday() == 5:
                holidays.add(datetime.date(year, 12, 28))
            elif boxing.weekday() == 6:
                holidays.add(datetime.date(year, 12, 28))
            else:
                holidays.add(boxing)

    return holidays


def create_bank_holiday_features(
    df: pd.DataFrame, date_col: str = "midday_day", window: int = 3
) -> pd.DataFrame:
    """Add UK bank holiday features.

    Args:
        df: DataFrame with date column.
        date_col: Name of date column.
        window: Days before/after holiday to flag proximity.

    Returns:
        DataFrame with added bank holiday features.
    """
    df = df.copy()

    dates = pd.to_datetime(df[date_col])
    min_year = dates.dt.year.min()
    max_year = dates.dt.year.max()
    bank_holidays = get_uk_bank_holidays(min_year - 1, max_year + 1)

    is_holiday = dates.dt.date.isin(bank_holidays).astype(int)
    df["is_bank_holiday"] = is_holiday

    days_to_holiday = []
    for d in dates:
        d_date = d.date()
        min_dist = window + 1
        for h in bank_holidays:
            dist = (h - d_date).days
            if abs(dist) <= window and abs(dist) < abs(min_dist):
                min_dist = dist
        if abs(min_dist) > window:
            min_dist = 0
        days_to_holiday.append(min_dist)

    df["days_to_holiday"] = days_to_holiday

    return df


def create_exogenous_lag_features(
    df: pd.DataFrame, cols: list[str], lags: list[int] | None = None
) -> pd.DataFrame:
    """Create lag features for exogenous columns.

    Args:
        df: DataFrame sorted by date.
        cols: Columns to create lags for.
        lags: List of lag values (default: [1, 2]).

    Returns:
        DataFrame with added lag features.
    """
    if lags is None:
        lags = [1, 2]

    valid_cols = [c for c in cols if c in df.columns]
    if not valid_cols:
        return df.copy()

    lag_data = {}
    for col in valid_cols:
        for lag in lags:
            lag_data[f"{col}_lag{lag}"] = df[col].shift(lag)

    lag_df = pd.DataFrame(lag_data, index=df.index)
    return pd.concat([df, lag_df], axis=1)


UPSTREAM_KEYWORDS = [
    "patients_in_a_e",
    "no_of_dtas",
    "g_a_bed_occupancy",
    "opel",
    "ambulance_queue",
    "handover_time_lost",
    "999_call",
    "4hr_breach",
]


def select_upstream_features(df: pd.DataFrame) -> list[str]:
    """Select causally-relevant upstream pressure features.

    These features represent system pressure (ambulance calls, bed occupancy, DTAs)
    that precede ED crowding and avoidable deaths.

    Args:
        df: DataFrame with feature columns.

    Returns:
        List of column names matching upstream pressure indicators.
    """
    selected = []
    for col in df.columns:
        col_lower = col.lower()
        if any(kw in col_lower for kw in UPSTREAM_KEYWORDS):
            if "lag" not in col_lower:
                selected.append(col)
    return selected


DOMAIN_KEYWORDS = {
    "ambulance": ["ambulance", "handover", "999_call", "category_c"],
    "ed": ["patients_in_a_e", "breach", "majors", "minors"],
    "flow": ["dta", "nctr", "discharge", "bed_occupancy"],
    "pressure": ["opel", "escalation"],
}


def create_domain_pca_features(
    df: pd.DataFrame,
    domain_keywords: dict[str, list[str]] | None = None,
    n_components: int = 1,
    lags: list[int] | None = None,
) -> pd.DataFrame:
    """Create PCA features for each domain, then lag them.

    Groups features by domain (ambulance, ED, flow, pressure), applies PCA to
    compress highly correlated features within each domain, and creates lagged
    versions for forecasting.

    Args:
        df: DataFrame with feature columns.
        domain_keywords: Dict mapping domain names to keyword lists.
                        If None, uses default DOMAIN_KEYWORDS.
        n_components: Number of PCA components to retain per domain.
        lags: List of lag values (default: [1, 2]).

    Returns:
        DataFrame with added domain PCA features and their lags.
    """
    if domain_keywords is None:
        domain_keywords = DOMAIN_KEYWORDS
    if lags is None:
        lags = [1, 2]

    df = df.copy()

    for domain, keywords in domain_keywords.items():
        cols = [
            c
            for c in df.columns
            if any(kw in c.lower() for kw in keywords) and "lag" not in c.lower()
        ]
        if len(cols) < 2:
            continue

        X = df[cols].fillna(df[cols].median())
        X_scaled = StandardScaler().fit_transform(X)
        pca = PCA(n_components=min(n_components, len(cols)))
        pcs = pca.fit_transform(X_scaled)

        for i in range(pcs.shape[1]):
            pc_col = f"{domain}_pc{i+1}"
            df[pc_col] = pcs[:, i]
            for lag in lags:
                df[f"{pc_col}_lag{lag}"] = df[pc_col].shift(lag)

    return df
