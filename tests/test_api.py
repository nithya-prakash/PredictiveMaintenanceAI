import pytest
from fastapi.testclient import TestClient

from backend import main
from backend.services.predictor import predictor_service
from ml.data.cmapss import SENSORS, load_test
from ml.features.preprocessing import FEATURE_COLS, build_features


@pytest.fixture
def client():
    with TestClient(main.app) as c:
        yield c


@pytest.fixture
def engine(fixture_dir):
    return load_test(fixture_dir).query("engine_id == 1")


def payload(engine, n=30, machine_id=7):
    return {"machine_id": machine_id, "readings": engine.tail(n)[["cycle"] + SENSORS].to_dict(orient="records")}


def test_health(client):
    body = client.get("/health").json()
    assert body["models_loaded"] is True and body["status"] == "ok"


def test_predictions_match_the_training_feature_path(client, engine):
    r = client.post("/api/v1/predict-rul", json=payload(engine)).json()
    X = build_features(engine)[FEATURE_COLS].tail(1)
    assert r["cycle"] == int(engine.cycle.max())
    assert r["predicted_rul"] == pytest.approx(max(0.0, predictor_service.bundle["rul"]["model"].predict(X)[0]))


def test_failure_uses_the_trained_threshold(client, engine):
    r = client.post("/api/v1/predict-failure", json=payload(engine)).json()
    assert r["threshold"] == predictor_service.bundle["failure"]["threshold"]
    assert r["failure_imminent"] == (r["failure_probability"] >= r["threshold"])


@pytest.mark.parametrize("n", [1, 14])
def test_too_little_history_is_rejected(client, engine, n):
    assert client.post("/api/v1/predict-failure", json=payload(engine, n=n)).status_code == 422


def test_non_consecutive_cycles_are_rejected(client, engine):
    body = payload(engine)
    body["readings"][5]["cycle"] += 3
    assert client.post("/api/v1/predict-rul", json=body).status_code == 422


def test_explain_and_anomaly(client, engine):
    x = client.post("/api/v1/explain", json=payload(engine)).json()
    assert x["top_contributor"] in FEATURE_COLS and len(x["feature_contributions"]) == 10
    a = client.post("/api/v1/anomaly", json=payload(engine)).json()
    assert a["is_anomaly"] == (a["anomaly_score"] < a["threshold"])


def test_rul_trajectory(client, engine):
    body = payload(engine, n=len(engine))  # the whole trajectory of this test engine
    r = client.post("/api/v1/rul-trajectory", json=body).json()
    # one prediction per cycle that has 15 cycles of history (the first 14 have too little)
    assert len(r["points"]) == len(body["readings"]) - 14
    assert r["points"][0]["cycle"] == body["readings"][14]["cycle"]


def test_recommendation_reports_persistence(client, engine):
    r = client.post("/api/v1/recommend-maintenance", json=payload(engine)).json()
    assert r["urgency"] in {"LOW", "MEDIUM", "HIGH"}
    assert r["persisted"] == main.state["database"]


def test_missing_models_give_503_not_500(client, engine, monkeypatch):
    monkeypatch.setattr(predictor_service, "bundle", None)
    monkeypatch.setattr(predictor_service, "error", "FileNotFoundError: model_bundle.joblib")
    assert client.post("/api/v1/predict-rul", json=payload(engine)).status_code == 503
    assert client.get("/health").status_code == 503


def test_rate_limit(client, engine, monkeypatch):
    monkeypatch.setattr(main.settings, "RATE_LIMIT_PER_MINUTE", 2)
    codes = [client.post("/api/v1/predict-rul", json=payload(engine)).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_trusted_dashboard_token_bypasses_rate_limit(client, engine, monkeypatch):
    monkeypatch.setattr(main.settings, "RATE_LIMIT_PER_MINUTE", 1)
    monkeypatch.setattr(main.settings, "API_TOKEN", "dashboard-secret")
    ok = {"X-API-Key": "dashboard-secret"}
    assert [client.post("/api/v1/predict-rul", json=payload(engine), headers=ok).status_code for _ in range(3)] == [200] * 3
    wrong = {"X-API-Key": "nope"}
    assert [client.post("/api/v1/predict-rul", json=payload(engine), headers=wrong).status_code for _ in range(2)] == [200, 429]


def test_predict_rul_returns_a_conformal_interval(client, engine):
    r = client.post("/api/v1/predict-rul", json=payload(engine)).json()
    assert r["interval_confidence"] == pytest.approx(0.9)
    assert 0.0 <= r["rul_lower"] <= r["predicted_rul"] <= r["rul_upper"] <= r["rul_cap"]
