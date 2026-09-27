import secrets
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from threading import Lock

from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from backend.core.config import settings
from backend.core.database import get_db, init_db
from backend.models.machine import Machine, Prediction, SensorReading
from backend.schemas.predict import (AnomalyResponse, EngineHistory, ExplanationResponse, FailureResponse,
                                     MaintenanceRecommendation, RULResponse)
from backend.services.predictor import predictor_service
from backend.services.xai import explain_rows

state = {"database": False}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["database"] = init_db()
    yield


app = FastAPI(
    title=settings.PROJECT_NAME,
    description="RUL estimation, imminent-failure classification, anomaly detection and SHAP explanations "
                "for turbofan engines, trained on NASA C-MAPSS FD001.",
    version=settings.VERSION,
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=settings.CORS_ORIGINS, allow_methods=["GET", "POST"],
                   allow_headers=["Content-Type", "X-API-Key"])
Instrumentator().instrument(app).expose(app)


# ---- per-client-IP rate limit (in-process; each API replica counts separately) ----
_calls = defaultdict(deque)
_calls_lock = Lock()


def rate_limit(request: Request, x_api_key: Optional[str] = Header(default=None)):
    if settings.API_TOKEN and x_api_key and secrets.compare_digest(x_api_key, settings.API_TOKEN):
        return  # trusted internal caller (the dashboard)
    key = request.client.host if request.client else "unknown"
    now = time.monotonic()
    with _calls_lock:
        calls = _calls[key]
        while calls and now - calls[0] > 60:
            calls.popleft()
        if len(calls) >= settings.RATE_LIMIT_PER_MINUTE:
            raise HTTPException(status_code=429, detail=f"Rate limit exceeded ({settings.RATE_LIMIT_PER_MINUTE}/minute).")
        calls.append(now)


def models_ready():
    if not predictor_service.loaded:
        raise HTTPException(status_code=503, detail=f"Models not loaded: {predictor_service.error}")
    return predictor_service


def features_for(history: EngineHistory, svc):
    return svc.features(history), history.readings[-1].cycle


@app.get("/health")
def health_check():
    body = {"status": "ok" if predictor_service.loaded else "degraded", "models_loaded": predictor_service.loaded,
            "database": state["database"]}
    if not predictor_service.loaded:
        body["model_error"] = predictor_service.error
        raise HTTPException(status_code=503, detail=body)
    return body


@app.get(f"{settings.API_V1_STR}/model-info")
def model_info(svc=Depends(models_ready)):
    b = svc.bundle
    return {"rul_model": b["rul"]["algorithm"], "rul_cap": b["rul"]["cap"],
            "failure_model": b["failure"]["algorithm"], "failure_threshold": b["failure"]["threshold"],
            "failure_window_cycles": b["failure"]["window"], "min_history": b["min_history"],
            "features": len(b["feature_cols"]), "metadata": b["metadata"]}


@app.post(f"{settings.API_V1_STR}/predict-rul", response_model=RULResponse, dependencies=[Depends(rate_limit)])
def predict_rul(history: EngineHistory, svc=Depends(models_ready)):
    X, cycle = features_for(history, svc)
    return RULResponse(machine_id=history.machine_id, cycle=cycle, predicted_rul=svc.predict_rul(X),
                       rul_cap=svc.bundle["rul"]["cap"])


@app.post(f"{settings.API_V1_STR}/rul-trajectory", dependencies=[Depends(rate_limit)])
def rul_trajectory(history: EngineHistory, svc=Depends(models_ready)):
    """Predicted RUL at every cycle of the history that has 15+ cycles before it
    (same features as training), for plotting a prediction over an engine's life."""
    import pandas as pd
    from ml.data.cmapss import SENSORS
    from ml.features.preprocessing import FEATURE_COLS, build_features
    readings = pd.DataFrame([r.model_dump() for r in history.readings])[["cycle"] + SENSORS]
    featured = build_features(readings.assign(engine_id=0))
    preds = svc.bundle["rul"]["model"].predict(featured[FEATURE_COLS]).clip(min=0)
    return {"machine_id": history.machine_id,
            "points": [{"cycle": int(c), "predicted_rul": float(p)} for c, p in zip(featured["cycle"], preds)]}


@app.post(f"{settings.API_V1_STR}/predict-failure", response_model=FailureResponse, dependencies=[Depends(rate_limit)])
def predict_failure(history: EngineHistory, svc=Depends(models_ready)):
    X, cycle = features_for(history, svc)
    prob, thr = svc.failure_probability(X), svc.bundle["failure"]["threshold"]
    return FailureResponse(machine_id=history.machine_id, cycle=cycle, failure_probability=prob, threshold=thr,
                           failure_imminent=prob >= thr, window_cycles=svc.bundle["failure"]["window"])


@app.post(f"{settings.API_V1_STR}/anomaly", response_model=AnomalyResponse, dependencies=[Depends(rate_limit)])
def detect_anomaly(history: EngineHistory, svc=Depends(models_ready)):
    X, cycle = features_for(history, svc)
    score, thr = svc.anomaly_score(X), svc.bundle["anomaly"]["score_threshold"]
    return AnomalyResponse(machine_id=history.machine_id, cycle=cycle, anomaly_score=score, threshold=thr,
                           is_anomaly=score < thr)


@app.post(f"{settings.API_V1_STR}/explain", response_model=ExplanationResponse, dependencies=[Depends(rate_limit)])
def explain_prediction(history: EngineHistory, svc=Depends(models_ready)):
    X, cycle = features_for(history, svc)
    contrib = explain_rows(svc.bundle, X).iloc[0]
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(10)
    return ExplanationResponse(machine_id=history.machine_id, cycle=cycle,
                               output=f"log-odds of failure within {svc.bundle['failure']['window']} cycles",
                               feature_contributions={k: float(v) for k, v in top.items()},
                               top_contributor=str(top.index[0]))


@app.post(f"{settings.API_V1_STR}/recommend-maintenance", response_model=MaintenanceRecommendation,
          dependencies=[Depends(rate_limit)])
def recommend_maintenance(history: EngineHistory, svc=Depends(models_ready), db: Session = Depends(get_db)):
    X, cycle = features_for(history, svc)
    prob, thr = svc.failure_probability(X), svc.bundle["failure"]["threshold"]
    rul = svc.predict_rul(X)
    score = svc.anomaly_score(X)
    is_anomaly = score < svc.bundle["anomaly"]["score_threshold"]

    # Rules on top of the models: the failure threshold is the one tuned in
    # training; the 60-cycle RUL margin gives two failure windows of notice.
    if prob >= thr:
        urgency, action = "HIGH", "Plan an inspection now: failure is likely within the next " \
                                  f"{svc.bundle['failure']['window']} cycles."
    elif rul <= 60 or is_anomaly:
        urgency, action = "MEDIUM", f"Schedule maintenance within about {int(rul)} cycles."
    else:
        urgency, action = "LOW", "No action needed. Continue standard monitoring."

    if urgency == "LOW":
        reason = "Failure probability and degradation indicators are low."
    else:
        contrib = explain_rows(svc.bundle, X).iloc[0]
        risk = contrib[contrib > 0].sort_values(ascending=False)
        driver = f"'{risk.index[0]}'" if len(risk) else "no single sensor"
        reason = (f"Failure probability {prob:.0%}, predicted RUL {rul:.0f} cycles"
                  f"{', anomalous readings' if is_anomaly else ''}; largest risk contribution from {driver}.")

    persisted = False
    if state["database"]:
        try:
            if db.get(Machine, history.machine_id) is None:
                db.add(Machine(id=history.machine_id, name=f"Engine-{history.machine_id}"))
                db.flush()
            last = history.readings[-1].model_dump()
            db.add(SensorReading(machine_id=history.machine_id, cycle=cycle,
                                 sensors={k: v for k, v in last.items() if k != "cycle"}))
            db.add(Prediction(machine_id=history.machine_id, cycle=cycle, predicted_rul=rul, failure_probability=prob,
                              anomaly_score=score, urgency=urgency, maintenance_recommended=urgency != "LOW"))
            db.commit()
            persisted = True
        except SQLAlchemyError:
            db.rollback()

    return MaintenanceRecommendation(machine_id=history.machine_id, cycle=cycle, urgency=urgency, action=action,
                                     reason=reason, predicted_rul=rul, failure_probability=prob,
                                     is_anomaly=bool(is_anomaly), persisted=persisted)
