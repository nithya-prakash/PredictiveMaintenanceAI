from typing import Dict, List, Optional

from pydantic import BaseModel, Field, create_model, field_validator

from ml.data.cmapss import SENSOR_DESCRIPTIONS, SENSORS
from ml.features.preprocessing import MIN_HISTORY


Reading = create_model(
    "Reading",
    cycle=(int, Field(ge=1, description="Operating cycle number of this reading")),
    **{s: (float, Field(description=SENSOR_DESCRIPTIONS[s])) for s in SENSORS},
)


class EngineHistory(BaseModel):
    """An engine's most recent consecutive cycles, oldest first. The models use
    rolling statistics over the last 15 cycles, so at least 15 are required;
    the prediction is for the last reading."""
    machine_id: int = Field(ge=1)
    readings: List[Reading] = Field(min_length=MIN_HISTORY, max_length=500)

    @field_validator("readings")
    @classmethod
    def consecutive_cycles(cls, readings):
        cycles = [r.cycle for r in readings]
        if any(b != a + 1 for a, b in zip(cycles, cycles[1:])):
            raise ValueError("readings must be consecutive cycles in ascending order")
        return readings


class RULResponse(BaseModel):
    machine_id: int
    cycle: int
    predicted_rul: float
    rul_lower: Optional[float] = Field(default=None, description="Lower bound of the conformal prediction interval")
    rul_upper: Optional[float] = Field(default=None, description="Upper bound of the conformal prediction interval")
    interval_confidence: Optional[float] = Field(default=None, description="Nominal coverage of [rul_lower, rul_upper], from engine-wise out-of-fold residuals")
    rul_cap: int = Field(description="Training labels are capped at this many cycles; predictions near it mean 'healthy, no visible degradation yet'")


class FailureResponse(BaseModel):
    machine_id: int
    cycle: int
    failure_probability: float
    threshold: float
    failure_imminent: bool
    window_cycles: int


class AnomalyResponse(BaseModel):
    machine_id: int
    cycle: int
    anomaly_score: float
    threshold: float
    is_anomaly: bool


class ExplanationResponse(BaseModel):
    machine_id: int
    cycle: int
    output: str
    feature_contributions: Dict[str, float]
    top_contributor: str


class MaintenanceRecommendation(BaseModel):
    machine_id: int
    cycle: int
    urgency: str
    action: str
    reason: str
    predicted_rul: float
    failure_probability: float
    is_anomaly: bool
    persisted: bool = Field(description="Whether this recommendation was written to the database")
