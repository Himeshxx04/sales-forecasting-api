import numpy as np
import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data" / "sales_data.xlsx"


def load_raw(filepath=None):
    path = filepath or DATA_PATH
    df = pd.read_excel(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.rename(columns={"Total": "sales"})
    # only columns we care about
    df = df[["State", "Date", "sales"]].copy()
    return df


def resample_to_weekly(df):
    """
    Raw data has irregular gaps between dates (could be 3, 5, 7, 8 or 9 days).
    Resample each state independently to a clean W (week-ending Sunday) frequency.
    Where a week had no data we interpolate - time-based linear is fine here.
    """
    chunks = []
    for state, grp in df.groupby("State"):
        grp = grp.set_index("Date").sort_index()
        weekly = grp["sales"].resample("W").sum()

        # resample("W").sum() will put 0 for a completely empty week.
        # Replace those zeros with NaN then interpolate so we don't skew the series.
        weekly = weekly.replace(0, np.nan)
        weekly = weekly.interpolate(method="time").ffill().bfill()

        chunks.append(
            pd.DataFrame(
                {"State": state, "Date": weekly.index, "sales": weekly.values}
            )
        )

    result = pd.concat(chunks, ignore_index=True)
    return result


def get_series_for_state(df, state):
    s = df[df["State"] == state].sort_values("Date").copy()
    s = s.set_index("Date")["sales"]
    return s


def train_val_split(series, val_weeks=16):
    """
    Chronological split - last val_weeks become validation set.
    Never shuffle time series data; that leaks future info into training.
    """
    cutoff = len(series) - val_weeks
    return series.iloc[:cutoff], series.iloc[cutoff:]


def get_all_states(df):
    return sorted(df["State"].unique().tolist())


def prepare_data(filepath=None):
    df = load_raw(filepath)
    df = resample_to_weekly(df)
    return df
