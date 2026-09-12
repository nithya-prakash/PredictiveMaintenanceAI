import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

valid_payload = {
    "machine_id": 999,
    "temperature": 80.5,
    "pressure": 110.2,
    "vibration": 0.8,
    "rpm": 1450.0,
    "voltage": 215.0,
    "current": 16.5,
    "oil_quality": 85.0
}

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["models_loaded"] is True

def test_predict_rul():
    response = client.post("/api/v1/predict-rul", json=valid_payload)
    assert response.status_code == 200
    data = response.json()
    assert "machine_id" in data
    assert "predicted_rul" in data
    assert isinstance(data["predicted_rul"], float)
    assert data["machine_id"] == 999

def test_predict_failure():
    response = client.post("/api/v1/predict-failure", json=valid_payload)
    assert response.status_code == 200
    data = response.json()
    assert "machine_id" in data
    assert "failure_probability" in data
    assert "failure_imminent" in data
    assert isinstance(data["failure_probability"], float)
    assert isinstance(data["failure_imminent"], bool)

def test_anomaly():
    response = client.post("/api/v1/anomaly", json=valid_payload)
    assert response.status_code == 200
    data = response.json()
    assert "is_anomaly" in data
    assert isinstance(data["is_anomaly"], bool)

def test_explain():
    response = client.post("/api/v1/explain", json=valid_payload)
    assert response.status_code == 200
    data = response.json()
    assert "feature_importance" in data
    assert "top_contributor" in data
    assert isinstance(data["top_contributor"], str)
    assert len(data["feature_importance"]) > 0

def test_recommend_maintenance():
    response = client.post("/api/v1/recommend-maintenance", json=valid_payload)
    assert response.status_code == 200
    data = response.json()
    assert "action" in data
    assert "urgency" in data
    assert "reason" in data
    assert data["urgency"] in ["LOW", "MEDIUM", "HIGH"]

def test_malformed_input():
    malformed_payload = {
        "machine_id": "abc",  # Should be int
        "temperature": "hot"  # Should be float
    }
    response = client.post("/api/v1/predict-rul", json=malformed_payload)
    assert response.status_code == 422  # Unprocessable Entity
