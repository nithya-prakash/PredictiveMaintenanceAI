# Power BI report

Data: `python -m scripts.export_bi` writes four CSVs to `bi/` (committed, ~1.6 MB), all from the
trained model and the official C-MAPSS FD001 test engines.

## Load and model
Power BI Desktop → Get data → Text/CSV → select all four files from `bi/`.

| Relationship | Cardinality |
|---|---|
| `dim_engine[engine_id]` → `fact_cycle_predictions[engine_id]` | 1 : many |
| `dim_engine[engine_id]` → `fact_sensor_readings[engine_id]` | 1 : many |
| `fact_cycle_predictions[engine_id, cycle]` ↔ `fact_sensor_readings[engine_id, cycle]` | add a `Key = engine_id & "-" & cycle` column to both, 1 : 1 |

`model_metrics` stands alone (long format: task, model, metric, value).

## Measures
```DAX
Engines At Risk = CALCULATE(DISTINCTCOUNT(dim_engine[engine_id]), dim_engine[risk_band_at_last_cycle] = "High")
RUL RMSE (last cycle) = SQRT(AVERAGEX(dim_engine, (MIN(dim_engine[rul_true_at_last_cycle], 125) - dim_engine[rul_pred_at_last_cycle]) ^ 2))
Anomaly Flag Rate = AVERAGE(fact_cycle_predictions[anomaly_flag])
Failure Alert Recall =
VAR pos = CALCULATE(COUNTROWS(fact_cycle_predictions), fact_cycle_predictions[failure_imminent] = 1)
VAR hit = CALCULATE(COUNTROWS(fact_cycle_predictions), fact_cycle_predictions[failure_imminent] = 1, fact_cycle_predictions[failure_flag] = 1)
RETURN DIVIDE(hit, pos)
```
`RUL RMSE (last cycle)` should read 17.19 and `Failure Alert Recall` 0.717 only if you filter to the
cycles the evaluation used; check against `evaluation/results/FINAL_REPORT.md` before quoting either.

## Suggested pages
1. **Fleet overview** — cards (engines, At Risk, RMSE), engine table with conditional formatting on `risk_band_at_last_cycle`, slicer on risk band.
2. **Engine drill-down** — engine slicer; line chart of `rul_pred` vs `rul_true_capped` by `cycle`; line of `failure_probability` with a constant line at the served threshold; sensor trend from `fact_sensor_readings`.
3. **Model quality** — `model_metrics` matrix (model × metric), anomaly flag rate by RUL bucket (shows the flag rate rising as failure nears).

Status: CSV export verified (last-cycle RMSE recomputed from the export equals the report's 17.19).
The `.pbix` itself is not included — build it in Power BI Desktop from the steps above.
