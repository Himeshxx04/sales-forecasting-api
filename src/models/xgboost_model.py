import numpy as np
import pandas as pd
import xgboost as xgb
import holidays as hd

_US_HOLIDAYS = hd.US()

_FEATURE_COLS = [
    "lag_1", "lag_7", "lag_30",
    "roll_mean_4w", "roll_std_4w", "roll_mean_12w",
    "week_of_year", "month", "quarter", "year", "is_holiday_week",
]


def _build_df(series):
    df = pd.DataFrame({"sales": series.values}, index=series.index)

    df["week_of_year"] = df.index.isocalendar().week.astype(int)
    df["month"] = df.index.month
    df["quarter"] = df.index.quarter
    df["year"] = df.index.year
    df["is_holiday_week"] = df.index.map(
        lambda d: int(any((d - pd.Timedelta(days=i)) in _US_HOLIDAYS for i in range(7)))
    )

    for lag in [1, 7, 30]:
        df[f"lag_{lag}"] = df["sales"].shift(lag)

    df["roll_mean_4w"] = df["sales"].shift(1).rolling(4, min_periods=2).mean()
    df["roll_std_4w"] = df["sales"].shift(1).rolling(4, min_periods=2).std()
    df["roll_mean_12w"] = df["sales"].shift(1).rolling(12, min_periods=4).mean()

    return df


class XGBoostModel:
    name = "XGBoost"

    def __init__(self):
        self.model = None
        self._train_series = None

    def fit(self, series):
        self._train_series = series.copy()
        df = _build_df(series).dropna()

        X = df[_FEATURE_COLS].values
        y = df["sales"].values

        self.model = xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        )
        self.model.fit(X, y)
        return self

    def predict(self, n_periods):
        assert self.model is not None, "call fit() first"

        # recursive multi-step forecast:
        # each predicted week gets appended to history so the next step
        # can compute lag features from it.
        history = self._train_series.copy()
        last_date = history.index[-1]
        preds = []

        for i in range(n_periods):
            next_date = last_date + pd.Timedelta(weeks=i + 1)
            # append a NaN placeholder, build features, then fill with prediction
            temp = pd.concat([history, pd.Series([np.nan], index=[next_date])])
            df = _build_df(temp)
            row = df.loc[next_date, _FEATURE_COLS].values.reshape(1, -1)
            p = float(self.model.predict(row)[0])
            preds.append(max(p, 0))
            history = pd.concat([history, pd.Series([p], index=[next_date])])

        return np.array(preds)
