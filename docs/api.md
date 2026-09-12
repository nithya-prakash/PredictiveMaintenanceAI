# API Reference

Base URL: `http://localhost:8001/api/v1` (via Docker Compose; the FastAPI container itself listens on `8000`, mapped to host port `8001`).

All endpoints accept a `SensorData` payload:

```json
{
  "machine_id": 1,
  "temperature": 85.0,
  "pressure": 75.0,
  "vibration": 1.8,
  "rpm": 1250.0,
  "voltage": 205.0,
  "current": 19.0,
  "oil_quality": 15.0
}
```

## `GET /health`

Returns service status and whether models loaded successfully at startup.

```json
{"status": "healthy", "models_loaded": true}
```

## `POST /api/v1/predict-rul`

Predicts Remaining Useful Life in cycles.

```json
{"machine_id": 1, "predicted_rul": 42.3}
```

## `POST /api/v1/predict-failure`

Predicts the probability of failure within the next 30 cycles, using whichever classifier won the PR-AUC benchmark (see [Models & XAI](models.md)).

```json
{"machine_id": 1, "failure_probability": 0.87, "failure_imminent": true}
```

## `POST /api/v1/anomaly`

Flags whether the current reading is an outlier relative to training data, via `IsolationForest`.

```json
{"machine_id": 1, "is_anomaly": false}
```

## `POST /api/v1/explain`

Returns SHAP feature attributions for the failure prediction.

```json
{
  "machine_id": 1,
  "feature_importance": {"vibration_roll_mean_15": 20.66, "oil_quality_roll_std_5": -20.36},
  "top_contributor": "vibration_roll_mean_15"
}
```

## `POST /api/v1/recommend-maintenance`

Combines the RUL, failure, and explanation calls into a single actionable recommendation. This is the only endpoint that writes to PostgreSQL — it get-or-creates the `Machine` row and logs the input `SensorReading` and the resulting `Prediction`.

```json
{
  "machine_id": 1,
  "action": "Immediate inspection required. High probability of imminent failure.",
  "urgency": "HIGH",
  "reason": "Driven primarily by anomalous 'vibration_roll_mean_15' readings."
}
```

Interactive Swagger docs are also available at `/docs` on the running API.
