import pandas as pd
from fastapi import APIRouter, HTTPException

from api.schemas import ForecastRequest, ForecastResponse, WeeklyForecast, ModelInfo, HealthResponse
from src.model_selector import load_model, get_registry
from src.data_processing import prepare_data, get_all_states

router = APIRouter()

# in-memory caches — loaded once on first use, reused on every subsequent request
_df = None
_states = []
_model_cache = {}   # {state_name: model_object}


def get_df():
    global _df, _states
    if _df is None:
        _df = prepare_data()
        _states = get_all_states(_df)
    return _df, _states


def get_model(state):
    """Load model from disk on first call, return cached version after that."""
    if state not in _model_cache:
        _model_cache[state] = load_model(state)
    return _model_cache[state]


@router.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    registry = get_registry()
    _, states = get_df()
    return HealthResponse(
        status="ok",
        trained_states=len(registry),
        message=f"{len(registry)}/{len(states)} states have trained models ready.",
    )


@router.get("/states", tags=["Data"])
def list_states():
    """Returns all states available in the dataset."""
    _, states = get_df()
    return {"states": states, "count": len(states)}


@router.get("/models", tags=["Models"])
def list_models():
    """Returns the best model and RMSE scores for every trained state."""
    registry = get_registry()
    if not registry:
        raise HTTPException(status_code=503, detail="No models trained yet. Run train.py first.")
    return {
        "trained_states": len(registry),
        "results": registry,
    }


@router.get("/models/{state}", response_model=ModelInfo, tags=["Models"])
def model_info(state: str):
    """Returns which model won for a specific state and the RMSE comparison."""
    registry = get_registry()
    matched = next((k for k in registry if k.lower() == state.lower()), None)
    if not matched:
        raise HTTPException(
            status_code=404,
            detail=f"No trained model for '{state}'. Either it doesn't exist or training hasn't run yet.",
        )
    entry = registry[matched]
    return ModelInfo(
        state=matched,
        best_model=entry["best_model"],
        rmse_scores=entry["scores"],
    )


@router.post("/forecast", response_model=ForecastResponse, tags=["Forecast"])
def forecast(req: ForecastRequest):
    """
    Forecast the next N weeks of beverage sales for a given US state.
    The model used is the one that scored lowest RMSE on the validation set during training.
    """
    df, states = get_df()

    matched_state = next((s for s in states if s.lower() == req.state.lower()), None)
    if not matched_state:
        raise HTTPException(status_code=404, detail=f"State '{req.state}' not found in dataset.")

    try:
        model = get_model(matched_state)
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail=f"Model for '{matched_state}' not trained yet. Run train.py first.",
        )

    preds = model.predict(req.weeks)

    last_date = df[df["State"] == matched_state]["Date"].max()
    forecast_dates = [last_date + pd.Timedelta(weeks=i + 1) for i in range(req.weeks)]

    registry = get_registry()
    model_name = registry.get(matched_state, {}).get("best_model", model.name)

    return ForecastResponse(
        state=matched_state,
        model_used=model_name,
        weeks_ahead=req.weeks,
        forecast=[
            WeeklyForecast(
                week=i + 1,
                date=d.strftime("%Y-%m-%d"),
                predicted_sales=round(float(p), 2),
            )
            for i, (d, p) in enumerate(zip(forecast_dates, preds))
        ],
    )
