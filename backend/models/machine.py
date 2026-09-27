from sqlalchemy import JSON, Boolean, Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.sql import func

from backend.core.database import Base


class Machine(Base):
    __tablename__ = "machines"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    status = Column(String, default="active")


class SensorReading(Base):
    """The most recent reading of each request (the history itself is sent by the client)."""
    __tablename__ = "sensor_readings"

    id = Column(Integer, primary_key=True, index=True)
    machine_id = Column(Integer, ForeignKey("machines.id"), index=True)
    timestamp = Column(DateTime, default=func.now())
    cycle = Column(Integer)
    sensors = Column(JSON)


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    machine_id = Column(Integer, ForeignKey("machines.id"), index=True)
    timestamp = Column(DateTime, default=func.now())
    cycle = Column(Integer)
    predicted_rul = Column(Float)
    failure_probability = Column(Float)
    anomaly_score = Column(Float)
    urgency = Column(String)
    maintenance_recommended = Column(Boolean, default=False)
