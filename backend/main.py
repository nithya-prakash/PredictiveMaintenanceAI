from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from backend.schemas.predict import SensorData, RULResponse, FailureResponse, AnomalyResponse, ExplanationResponse, MaintenanceRecommendation
from backend.services.predictor import predictor_service
from backend.services.xai import xai_service

app = FastAPI(
    title="Predictive Maintenance AI",
    description="Industrial AI for RUL prediction, failure classification, and anomaly detection.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus Monitoring
Instrumentator().instrument(app).expose(app)

@app.get("/health")
def health_check():
    return {"status": "healthy", "models_loaded": predictor_service.model_rul is not None}

@app.post("/api/v1/predict-rul", response_model=RULResponse)
def predict_rul(data: SensorData):
    rul = predictor_service.predict_rul(data)
    return RULResponse(machine_id=data.machine_id, predicted_rul=rul)

@app.post("/api/v1/predict-failure", response_model=FailureResponse)
def predict_failure(data: SensorData):
    prob = predictor_service.predict_failure(data)
    return FailureResponse(
        machine_id=data.machine_id, 
        failure_probability=prob, 
        failure_imminent=prob > 0.5
    )

@app.post("/api/v1/anomaly", response_model=AnomalyResponse)
def detect_anomaly(data: SensorData):
    is_anomaly = predictor_service.detect_anomaly(data)
    return AnomalyResponse(machine_id=data.machine_id, is_anomaly=is_anomaly)

@app.post("/api/v1/explain", response_model=ExplanationResponse)
def explain_prediction(data: SensorData):
    explanation = xai_service.explain_failure(data)
    if "error" in explanation:
        raise HTTPException(status_code=503, detail=explanation["error"])
        
    return ExplanationResponse(
        machine_id=data.machine_id,
        feature_importance=explanation["feature_importance"],
        top_contributor=explanation["top_contributor"]
    )

@app.post("/api/v1/recommend-maintenance", response_model=MaintenanceRecommendation)
def recommend_maintenance(data: SensorData):
    prob = predictor_service.predict_failure(data)
    rul = predictor_service.predict_rul(data)
    
    explanation = xai_service.explain_failure(data)
    top_feature = explanation.get("top_contributor", "Unknown")
    
    if prob > 0.8:
        urgency = "HIGH"
        action = f"Immediate inspection required. High probability of imminent failure."
    elif prob > 0.4 or rul < 50:
        urgency = "MEDIUM"
        action = f"Schedule maintenance within {int(rul)} cycles."
    else:
        urgency = "LOW"
        action = "Machine operating normally. Continue standard monitoring."
        
    reason = f"Driven primarily by anomalous '{top_feature}' readings." if urgency != "LOW" else "All readings nominal."
    
    return MaintenanceRecommendation(
        machine_id=data.machine_id,
        action=action,
        urgency=urgency,
        reason=reason
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
