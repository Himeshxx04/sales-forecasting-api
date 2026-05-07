import warnings
import numpy as np
from pmdarima import auto_arima


class ARIMAModel:
    name = "SARIMA"

    def __init__(self):
        self.model = None

    def fit(self, series):
        vals = series.values.astype(float)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # m=52 is the correct seasonal period for weekly data (yearly cycle),
            # but it can be unstable on some state series. We try it first, then
            # fall back to m=4 (quarterly), then plain ARIMA if both fail.
            for seasonal, m in [(True, 52), (True, 4), (False, 1)]:
                try:
                    self.model = auto_arima(
                        vals,
                        seasonal=seasonal,
                        m=m,
                        max_p=2, max_q=2, max_d=2,
                        max_P=1, max_Q=1, max_D=1,
                        stepwise=True,
                        suppress_warnings=True,
                        error_action="ignore",
                        information_criterion="aic",
                        n_jobs=1,   # single-threaded avoids forking issues on Windows
                    )
                    break
                except Exception:
                    continue

        return self

    def predict(self, n_periods):
        assert self.model is not None, "call fit() first"
        preds, _ = self.model.predict(n_periods=n_periods, return_conf_int=True)
        # sales can't be negative
        return np.maximum(preds, 0)
