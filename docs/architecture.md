# Architecture

The platform is a small set of Docker Compose services (`docker-compose.yml`):

## System Components

1. **FastAPI Backend** (`backend/`): Serves REST endpoints for RUL, failure probability, anomaly detection, SHAP explanations, and maintenance recommendations. Loads model artifacts from `models/` at startup.
2. **Streamlit Dashboard** (`dashboard/`): Lets an operator adjust sensor telemetry and view diagnostics, calling the FastAPI backend over REST.
3. **PostgreSQL**: Runs as a service and ORM models exist for it (`backend/models/machine.py`), but the API does not currently persist predictions to it — the models are defined but not yet wired into the request handlers.
4. **Redis**: Runs as a service; nothing in the application code currently reads or writes to it.
5. **Prometheus / Grafana**: Prometheus scrapes `/metrics` on the FastAPI backend (via `prometheus-fastapi-instrumentator`); Grafana is available for dashboards on top of that data.
6. **MLflow**: Used by the training scripts (`ml/training/`) for experiment tracking, not by the serving path.

There is no message broker or background worker in this system — sensor data is sent directly to the FastAPI backend in each request; it isn't streamed through a queue.

## Request Flow

```mermaid
graph TD
    A[Streamlit Dashboard] -->|REST| B[FastAPI Backend]
    B -->|joblib.load| C[models/*.pkl]
    B -->|SHAP| D[Explainability]
    E[Training Scripts] -->|joblib.dump| C
    E -->|Track experiments| F[MLflow]
    B -->|/metrics| G[Prometheus]
    G --> H[Grafana]
```

## Model Selection

`ml/training/train_classifier.py` benchmarks a `LogisticRegression` baseline against an Optuna-tuned `RandomForestClassifier` by PR-AUC and saves whichever wins as `models/classifier.pkl`, tagged with the winning algorithm. `backend/services/predictor.py` and `backend/services/xai.py` read that tag and behave accordingly (e.g. picking `TreeExplainer` vs `LinearExplainer` for SHAP) rather than assuming a specific algorithm. See [Models & XAI](models.md) for details.
