import json
import joblib
import numpy as np
from pathlib import Path

from src.models.arima_model import ARIMAModel
from src.models.prophet_model import ProphetModel
from src.models.xgboost_model import XGBoostModel
from src.models.lstm_model import LSTMModel
from src.data_processing import train_val_split

MODELS_DIR = Path(__file__).parent.parent / "models_saved"
REGISTRY_PATH = MODELS_DIR / "registry.json"

VAL_WEEKS = 16  # ~4 months held out for comparison


def rmse(actual, predicted):
    return float(np.sqrt(np.mean((np.array(actual) - np.array(predicted)) ** 2)))


def _load_registry():
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH) as f:
            return json.load(f)
    return {}


def _save_registry(reg):
    MODELS_DIR.mkdir(exist_ok=True)
    with open(REGISTRY_PATH, "w") as f:
        json.dump(reg, f, indent=2)


def train_and_select(state, series, verbose=True):
    """
    Trains all 4 models on the training portion of `series`,
    evaluates on validation, saves the best model, updates the registry.
    Returns a dict with results for each model.
    """
    MODELS_DIR.mkdir(exist_ok=True)
    train, val = train_val_split(series, val_weeks=VAL_WEEKS)

    candidates = [ARIMAModel(), ProphetModel(), XGBoostModel(), LSTMModel()]
    results = {}

    for model in candidates:
        try:
            if verbose:
                print(f"  [{state}] training {model.name}...", flush=True)
            model.fit(train)
            preds = model.predict(len(val))
            score = rmse(val.values, preds)
            results[model.name] = {"rmse": round(score, 2), "model": model}
            if verbose:
                print(f"    -> RMSE: {score:,.0f}")
        except Exception as e:
            if verbose:
                print(f"    -> {model.name} failed: {e}")
            results[model.name] = {"rmse": float("inf"), "model": None}

    # pick the model with the lowest RMSE that actually trained successfully
    best_name = min(
        (k for k, v in results.items() if v["model"] is not None),
        key=lambda k: results[k]["rmse"],
    )
    best_model = results[best_name]["model"]

    # retrain best model on the full series before saving
    # (more data = better predictions in production)
    best_model.fit(series)
    save_path = MODELS_DIR / f"{state.replace(' ', '_')}.pkl"
    joblib.dump(best_model, save_path)

    # update registry
    reg = _load_registry()
    reg[state] = {
        "best_model": best_name,
        "scores": {k: v["rmse"] for k, v in results.items() if v["rmse"] != float("inf")},
    }
    _save_registry(reg)

    if verbose:
        print(f"  [{state}] winner: {best_name} (RMSE {results[best_name]['rmse']:,.0f})\n")

    return reg[state]


def load_model(state):
    path = MODELS_DIR / f"{state.replace(' ', '_')}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"No trained model found for '{state}'. Run train.py first.")
    return joblib.load(path)


def get_registry():
    return _load_registry()
