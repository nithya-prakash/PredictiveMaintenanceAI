from pydantic import BaseModel
from typing import List, Optional, Dict

class SensorData(BaseModel):
    machine_id: int
    temperature: float
    pressure: float
    vibration: float
    rpm: float
    voltage: float
    current: float
    oil_quality: float

class BatchSensorData(BaseModel):
    data: List[SensorData]

class RULResponse(BaseModel):
    machine_id: int
    predicted_rul: float

class FailureResponse(BaseModel):
    machine_id: int
    failure_probability: float
    failure_imminent: bool

class AnomalyResponse(BaseModel):
    machine_id: int
    is_anomaly: bool

class ExplanationResponse(BaseModel):
    machine_id: int
    feature_importance: Dict[str, float]
    top_contributor: str

class MaintenanceRecommendation(BaseModel):
    machine_id: int
    action: str
    urgency: str
    reason: str
