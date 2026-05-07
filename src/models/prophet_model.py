import numpy as np
import pandas as pd

# suppress the noisy Stan output
import logging
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

from prophet import Prophet


class ProphetModel:
    name = "Prophet"

    def __init__(self):
        self.model = None
        self._last_date = None

    def fit(self, series):
        df = pd.DataFrame({"ds": series.index, "y": series.values})

        self.model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,  # our data is already weekly-aggregated
            daily_seasonality=False,
            seasonality_mode="multiplicative",  # works better for sales with growing trend
            changepoint_prior_scale=0.05,
        )
        self.model.fit(df)
        self._last_date = series.index[-1]
        return self

    def predict(self, n_periods):
        assert self.model is not None, "call fit() first"
        future = self.model.make_future_dataframe(periods=n_periods, freq="W")
        forecast = self.model.predict(future)
        preds = forecast.tail(n_periods)["yhat"].values
        return np.maximum(preds, 0)
