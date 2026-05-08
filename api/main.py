"""
Sales Forecasting API
---------------------
Serves 8-week ahead beverage sales forecasts for US states.
Each state uses the best-performing model selected during training
(SARIMA / Prophet / XGBoost / LSTM).

Start locally:
    uvicorn api.main:app --reload --port 8000

Docs available at:
    http://localhost:8000/docs   (Swagger UI)
    http://localhost:8000/redoc  (ReDoc)
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router, get_df, get_model
from src.model_selector import get_registry


@asynccontextmanager
async def lifespan(app: FastAPI):
    # pre-load data and all models at startup so first requests are instant
    print("Loading dataset...", flush=True)
    df, states = get_df()
    print(f"Dataset loaded. {len(states)} states.", flush=True)

    registry = get_registry()
    print(f"Pre-loading {len(registry)} trained models...", flush=True)
    for state in registry:
        try:
            get_model(state)
        except Exception as e:
            print(f"  Warning: could not load model for {state}: {e}", flush=True)
    print("All models loaded. API is ready.", flush=True)

    yield  # server runs here


app = FastAPI(
    title="Sales Forecasting API",
    description=(
        "End-to-end time series forecasting system for US state-level beverage sales. "
        "Trains SARIMA, Prophet, XGBoost and LSTM models per state, "
        "automatically selects the best performer, and serves 8-week forecasts via REST."
    ),
    version="1.0.0",
    contact={"name": "Sales Forecasting Project"},
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/", tags=["System"])
def root():
    return {
        "message": "Sales Forecasting API is running.",
        "docs": "/docs",
        "endpoints": "/api/v1",
    }
