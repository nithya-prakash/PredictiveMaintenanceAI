# API reference

Base URL with Docker Compose: `http://127.0.0.1:8001/api/v1`. Interactive docs: `http://127.0.0.1:8001/docs`.

## Request body

All `POST` endpoints take an engine's most recent **consecutive** cycles, oldest first, at least 15 (the models use rolling statistics over 15 cycles) and at most 500. The prediction is for the last reading. Sensor names follow C-MAPSS (`s2` = T24 LPC outlet temperature, `s11` = Ps30 HPC outlet static pressure, … — see the schema in `/docs`).

```json
{"machine_id": 31, "readings": [{"cycle": 182, "s2": 643.23, "s3": 1599.84, "...": "14 sensors"}, "... 15+ readings"]}
```

Errors: `422` for fewer than 15 readings or non-consecutive cycles, `429` when the rate limit is exceeded, `503` when the models are not loaded.

## Endpoints

| Method and path | Returns |
|---|---|
| `POST /predict-rul` | `predicted_rul` (cycles; the model was trained with RUL capped at 125, so values near 125 mean "no visible degradation yet") |
| `POST /predict-failure` | `failure_probability`, `threshold` (tuned in training), `failure_imminent`, `window_cycles` (30) |
| `POST /anomaly` | `anomaly_score`, `threshold`, `is_anomaly` |
| `POST /explain` | top 10 SHAP contributions to the failure log-odds, `top_contributor` |
| `POST /rul-trajectory` | predicted RUL for every cycle of the history that has 15 cycles before it |
| `POST /recommend-maintenance` | `urgency` (LOW/MEDIUM/HIGH), `action`, `reason`, the underlying predictions, `persisted` |
| `GET /model-info` | deployed algorithms, threshold, training metadata |
| `GET /health` (no `/api/v1` prefix) | `models_loaded`, `database`; 503 if models are missing |
| `GET /metrics` (no prefix) | Prometheus metrics |

## Recommendation rules

`HIGH` if the failure probability is at or above the trained threshold; `MEDIUM` if the predicted RUL is 60 cycles or less or the readings are anomalous; otherwise `LOW`. For MEDIUM/HIGH the reason names the feature with the largest positive SHAP contribution to failure risk.
