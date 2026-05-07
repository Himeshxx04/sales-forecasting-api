import numpy as np
import pandas as pd
import holidays as hd


# US holidays we'll flag - having a holiday in a week typically spikes beverage sales
_US_HOLIDAYS = hd.US()


def _has_holiday_in_week(week_end_date):
    # week_end_date is a Sunday; check the full Mon-Sun window
    for offset in range(7):
        day = week_end_date - pd.Timedelta(days=offset)
        if day in _US_HOLIDAYS:
            return 1
    return 0


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds all engineered features to the weekly resampled dataframe.
    Must be called AFTER resample_to_weekly() so dates are clean and uniform.

    Returns a new dataframe with added columns - original rows are preserved.
    Rows with NaN in lag/rolling cols (the first N rows per state) are dropped
    at the END, not here, because models need to decide their own cutoff.
    """
    df = df.sort_values(["State", "Date"]).copy()

    # --- time-based features ---
    df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)
    df["month"] = df["Date"].dt.month
    df["quarter"] = df["Date"].dt.quarter
    df["year"] = df["Date"].dt.year

    # holiday flag - precompute once, it's slow per-row
    holiday_cache = {}
    def holiday_flag(d):
        if d not in holiday_cache:
            holiday_cache[d] = _has_holiday_in_week(d)
        return holiday_cache[d]

    df["is_holiday_week"] = df["Date"].apply(holiday_flag)

    # --- lag features (per state so we never mix states) ---
    # t-1: last week's sales
    # t-7: 7 weeks ago (~2 months)
    # t-30: 30 weeks ago (~7 months, captures a half-year seasonal signal)
    for lag in [1, 7, 30]:
        df[f"lag_{lag}"] = df.groupby("State")["sales"].shift(lag)

    # --- rolling stats (shift by 1 first to avoid target leakage) ---
    # 4-week window (~1 month) - captures short-term momentum
    df["roll_mean_4w"] = df.groupby("State")["sales"].transform(
        lambda x: x.shift(1).rolling(4, min_periods=2).mean()
    )
    df["roll_std_4w"] = df.groupby("State")["sales"].transform(
        lambda x: x.shift(1).rolling(4, min_periods=2).std()
    )
    # 12-week window (~3 months) - captures medium-term trend
    df["roll_mean_12w"] = df.groupby("State")["sales"].transform(
        lambda x: x.shift(1).rolling(12, min_periods=4).mean()
    )

    return df


# the columns XGBoost and LSTM will actually use as input features
FEATURE_COLS = [
    "lag_1",
    "lag_7",
    "lag_30",
    "roll_mean_4w",
    "roll_std_4w",
    "roll_mean_12w",
    "week_of_year",
    "month",
    "quarter",
    "year",
    "is_holiday_week",
]
