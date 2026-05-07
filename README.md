# Sales Forecasting API

End-to-end time series forecasting system for US state-level beverage sales.
Trains four models per state, auto-selects the best performer, and serves
8-week predictions via a REST API.

## Models
| Model | Description |
|---|---|
| SARIMA | Seasonal ARIMA via pmdarima auto_arima |
| Prophet | Facebook Prophet with multiplicative seasonality |
| XGBoost | Gradient boosting with lag + rolling features |
| LSTM | 2-layer PyTorch LSTM with MinMax scaling |

## Feature Engineering
- Lag features: t-1, t-7, t-30 (weeks)
- Rolling mean/std: 4-week and 12-week windows
- Time features: week of year, month, quarter, year
- US holiday flag (week-level)

## Setup

```bash
pip install -r requirements.txt
```

## Train models

```bash
# train all 43 states (takes ~30 min)
python train.py

# train specific states only
python train.py --states California Texas Florida
```

## Run API locally

```bash
uvicorn api.main:app --reload --port 8000
```

Swagger docs: http://localhost:8000/docs

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/v1/health` | API health + trained state count |
| GET | `/api/v1/states` | All available states |
| GET | `/api/v1/models` | Best model + RMSE for all trained states |
| GET | `/api/v1/models/{state}` | Model comparison for one state |
| POST | `/api/v1/forecast` | Get 8-week forecast for a state |

## Forecast request example

```bash
curl -X POST http://localhost:8000/api/v1/forecast \
  -H "Content-Type: application/json" \
  -d '{"state": "California", "weeks": 8}'
```

## Dataset
Weekly beverage sales data across 43 US states (2019–2023).
