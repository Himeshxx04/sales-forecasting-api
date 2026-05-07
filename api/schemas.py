from pydantic import BaseModel, field_validator
from typing import Optional


class ForecastRequest(BaseModel):
    state: str
    weeks: Optional[int] = 8

    @field_validator("weeks")
    @classmethod
    def weeks_must_be_positive(cls, v):
        if v < 1 or v > 52:
            raise ValueError("weeks must be between 1 and 52")
        return v

    @field_validator("state")
    @classmethod
    def state_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("state cannot be empty")
        return v.strip()


class WeeklyForecast(BaseModel):
    week: int
    date: str
    predicted_sales: float


class ForecastResponse(BaseModel):
    state: str
    model_used: str
    weeks_ahead: int
    forecast: list[WeeklyForecast]


class ModelInfo(BaseModel):
    state: str
    best_model: str
    rmse_scores: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    trained_states: int
    message: str
