# Predictive Maintenance AI

Welcome to the documentation for the **Predictive Maintenance AI Platform**.

This platform predicts industrial machine failures before they happen, estimates Remaining Useful Life (RUL), detects anomalies in sensor streams, and explains predictions using SHAP.

![Dashboard demo](assets/demo.gif)

## Features

- **Remaining Useful Life (RUL) Prediction**: A `RandomForestRegressor`, tuned with Optuna, estimates cycles-to-failure.
- **Imminent Failure Classification**: Benchmarks a `LogisticRegression` baseline against a tuned `RandomForestClassifier` by PR-AUC and deploys whichever actually wins — see [Models & XAI](models.md) for why this matters and what the numbers are.
- **Anomaly Detection**: Unsupervised detection of novel operating regimes using an `IsolationForest`.
- **Explainable AI (XAI)**: Every failure prediction is explained with SHAP, using `TreeExplainer` or `LinearExplainer` depending on which classifier is actually deployed.
- **MLOps Integration**: MLflow for experiment tracking, Optuna for hyperparameter optimization, Docker Compose for local deployment.

## Data & Validation

All training and evaluation data is **synthetically generated** (`scripts/generate_dataset.py`) — this platform has not been trained or validated on real machine telemetry. Treat it as a demonstration of an end-to-end MLOps pipeline, not a production-ready failure predictor. See the [main README](https://github.com/nithya-prakash/PredictiveMaintenanceAI#data--validation) for details.
