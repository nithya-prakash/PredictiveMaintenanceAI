# Manufacturing Predictive Maintenance AI

**End-to-end AI platform capable of predicting industrial machine failures before they happen, estimating Remaining Useful Life (RUL), detecting anomalies in sensor streams, explaining predictions via SHAP, and recommending maintenance actions.**

*Built for Industrial ML Engineering portfolios (e.g., Siemens, Bosch, BMW, NVIDIA).*

---

## Features

- **Predictive Maintenance Engine:** Estimates probability of failure within the next 30 cycles using `RandomForestClassifier`.
- **Remaining Useful Life (RUL):** Forecasts exactly how many cycles a machine has left using `RandomForestRegressor`.
- **Anomaly Detection:** Identifies novel operating conditions and sensor drift via `IsolationForest`.
- **Explainable AI (XAI):** Calculates SHAP feature importance to explain exactly *why* a machine is predicted to fail (e.g., "Driven primarily by anomalous vibration readings").
- **Recommendation Engine:** Suggests actionable maintenance tasks based on failure probabilities and root causes.
- **Microservices Architecture:** 
  - **FastAPI** backend for high-performance inference.
  - **Streamlit** dashboard for monitoring.
  - **PostgreSQL / Redis** for storage and caching.
  - **Prometheus / Grafana** for monitoring system health.
- **MLOps Integration:** `MLflow` for experiment tracking, model registry, and metrics logging. `Optuna` for automated hyperparameter optimization. `Docker Compose` for seamless deployment.

---

## Repository Structure

```
PredictiveMaintenanceAI/
├── backend/                  # FastAPI Application (API, Services, ORM)
├── ml/                       # Machine Learning Pipeline (Preprocessing, Training)
├── models/                   # Saved artifacts (Scalers, RandomForests, IsolationForest)
├── dashboard/                # Streamlit UI
├── scripts/                  # Data generators (Simulated telemetry)
├── docker/                   # Dockerfiles
├── docker-compose.yml        # Orchestration
├── mkdocs.yml                # Documentation configuration
└── docs/                     # Architecture and API documentation
```

---

## Tech Stack

- **Machine Learning:** `scikit-learn`, `pandas`, `numpy`, `SHAP`
- **MLOps:** `MLflow`, `Optuna`
- **Backend:** `FastAPI`, `SQLAlchemy`, `Pydantic`, `Uvicorn`
- **Database:** `PostgreSQL`, `Redis`
- **UI:** `Streamlit`, `Plotly`
- **DevOps:** `Docker`, `Prometheus`, `Grafana`

---

## Getting Started

### 1. Generate the Dataset
Since industrial data is proprietary, we include a high-fidelity synthetic data generator that simulates multi-sensor degradation over time.
```bash
python scripts/generate_dataset.py --machines 150
```

### 2. Train the Models
Run the ML pipelines to train the RUL, Classification, and Anomaly models. This automatically tracks experiments in MLflow and saves artifacts to the `models/` directory.
```bash
export PYTHONPATH=.
python ml/training/train_rul.py --trials 10
python ml/training/train_classifier.py --trials 10
python ml/training/train_anomaly.py
```

### 3. Deploy the Stack
Spin up the entire microservices architecture using Docker Compose.
```bash
docker-compose up --build
```
- **FastAPI Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Streamlit Dashboard:** [http://localhost:8501](http://localhost:8501)
- **Prometheus Metrics:** [http://localhost:9090](http://localhost:9090)

---

## XAI & Maintenance Recommendations

The API returns not just predictions, but explanations:
```json
{
  "machine_id": 42,
  "action": "Immediate inspection required. High probability of imminent failure.",
  "urgency": "HIGH",
  "reason": "Driven primarily by anomalous 'vibration_roll_mean_5' readings."
}
```

---

## 📜 License
MIT License.
