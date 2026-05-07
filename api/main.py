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
from pathlib import Path

# make sure src/ is importable regardless of where uvicorn is launched from
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router

app = FastAPI(
    title="Sales Forecasting API",
    description=(
        "End-to-end time series forecasting system for US state-level beverage sales. "
        "Trains SARIMA, Prophet, XGBoost and LSTM models per state, "
        "automatically selects the best performer, and serves 8-week forecasts via REST."
    ),
    version="1.0.0",
    contact={
        "name": "Sales Forecasting Project",
    },
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
