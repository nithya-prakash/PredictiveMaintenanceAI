from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey
from sqlalchemy.sql import func
from backend.core.database import Base

class Machine(Base):
    __tablename__ = "machines"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    location = Column(String)
    installation_date = Column(DateTime)
    status = Column(String, default="active")

class SensorReading(Base):
    __tablename__ = "sensor_readings"

    id = Column(Integer, primary_key=True, index=True)
    machine_id = Column(Integer, ForeignKey("machines.id"), index=True)
    timestamp = Column(DateTime, default=func.now())
    
    # Sensor values
    temperature = Column(Float)
    pressure = Column(Float)
    vibration = Column(Float)
    rpm = Column(Float)
    voltage = Column(Float)
    current = Column(Float)
    oil_quality = Column(Float)

class Prediction(Base):
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    machine_id = Column(Integer, ForeignKey("machines.id"), index=True)
    timestamp = Column(DateTime, default=func.now())
    
    predicted_rul = Column(Float)
    failure_probability = Column(Float)
    anomaly_score = Column(Float)
    maintenance_recommended = Column(Boolean, default=False)
