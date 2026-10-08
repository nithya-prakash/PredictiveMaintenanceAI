# Turbofan Predictive Maintenance AI

A predictive-maintenance system trained and evaluated on **NASA's C-MAPSS turbofan engine dataset (FD001)**. From an engine's recent sensor history it estimates remaining useful life (RUL), the probability of failure within 30 cycles, and whether the readings are anomalous, explains the failure risk with SHAP, and recommends a maintenance action.

![Dashboard demo](assets/demo.gif)

## At a glance

| | Deployed model | NASA test set (100 engines) |
|---|---|---|
| Remaining useful life | Random Forest | RMSE 17.19, NASA score 573 (published Random Forest: 17.91 / 480) |
| Failure within 30 cycles | Logistic regression, threshold 0.80 | PR-AUC 0.842, precision 0.77, recall 0.72 |
| Anomaly detection | IsolationForest on early-life cycles | flag rate 1% when healthy → 99% in the last 30 cycles |

Every modelling choice was made with engine-wise cross-validation on the training engines; the official test engines were used once, for these numbers. See [Models & evaluation](models.md).

## Pages

- [Architecture](architecture.md): services, request flow, security.
- [API reference](api.md): endpoints and payloads.
- [Models & evaluation](models.md): data, features, training protocol, full results, limitations.
- [Details and setup](details.md): full results, setup, security notes.
- [Power BI](powerbi.md): star-schema export and DAX measures.
