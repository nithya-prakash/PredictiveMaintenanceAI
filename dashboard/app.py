"""
Dashboard: replays NASA C-MAPSS FD001 *test* engines through the API.

Pick an engine and a cycle; the dashboard sends that engine's real sensor
history up to that cycle to the API, exactly as a monitoring system would. NASA
publishes the true RUL for the test engines, so the prediction can be compared
with the truth on screen.
"""
import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from ml.data.cmapss import DEFAULT_DIR, SENSOR_DESCRIPTIONS, SENSORS, load_test

API_URL = os.environ.get("API_URL", "http://localhost:8000/api/v1")
# Marks the dashboard as a trusted caller, exempt from the API's per-IP rate limit.
HEADERS = {"X-API-Key": os.environ["API_TOKEN"]} if os.environ.get("API_TOKEN") else {}
DATA_DIR = Path(os.environ.get("CMAPSS_DIR", DEFAULT_DIR))
TIMEOUT = 15
MAX_READINGS = 500  # API limit per request

st.set_page_config(page_title="Predictive Maintenance | C-MAPSS", layout="wide")
st.title("Turbofan predictive maintenance")
st.caption("NASA C-MAPSS FD001 test engines replayed through the API. True RUL is known for these engines, "
           "so every prediction can be checked.")


@st.cache_data
def test_data():
    return load_test(DATA_DIR)


if not (DATA_DIR / "test_FD001.txt").exists():
    st.error(f"C-MAPSS data not found in {DATA_DIR}. Run `python -m ml.data.cmapss` (downloads it from NASA).")
    st.stop()

df = test_data()
engine_id = st.sidebar.selectbox("Test engine", sorted(df.engine_id.unique()), index=30)
engine = df[df.engine_id == engine_id].reset_index(drop=True)
last_cycle = int(engine.cycle.max())
cycle = st.sidebar.slider("Current cycle", 15, last_cycle, last_cycle,
                          help="The API needs at least 15 consecutive cycles of history.")
history = engine[engine.cycle <= cycle].tail(MAX_READINGS)
true_rul = int(history.rul.iloc[-1])
payload = {"machine_id": int(engine_id), "readings": history[["cycle"] + SENSORS].to_dict(orient="records")}


def call(path, method="post"):
    r = requests.request(method, f"{API_URL}/{path}", json=payload if method == "post" else None,
                         headers=HEADERS, timeout=TIMEOUT)
    if not r.ok:
        raise RuntimeError(f"{path}: HTTP {r.status_code} {r.json().get('detail', '')}")
    return r.json()


@st.cache_data(ttl=300)
def model_info():
    return call("model-info", "get")


try:
    # The recommendation already contains RUL, failure probability and the anomaly
    # flag, so one refresh needs 3 calls instead of 6.
    info = model_info()
    rec, xai, traj = call("recommend-maintenance"), call("explain"), call("rul-trajectory")
except Exception as e:  # noqa: BLE001 - shown to the user
    st.error(f"API request failed: {e}")
    st.stop()
rul = {"predicted_rul": rec["predicted_rul"], "rul_cap": info["rul_cap"]}
fail = {"failure_probability": rec["failure_probability"], "threshold": info["failure_threshold"],
        "window_cycles": info["failure_window_cycles"]}
anom = {"is_anomaly": rec["is_anomaly"]}

c1, c2, c3, c4 = st.columns(4)
c1.metric("Predicted RUL (cycles)", f"{rul['predicted_rul']:.0f}",
          help=f"Model trained with RUL capped at {rul['rul_cap']}: values near it mean 'no visible degradation yet'.")
c2.metric("True RUL (NASA)", f"{true_rul}", delta=f"{rul['predicted_rul'] - min(true_rul, rul['rul_cap']):+.0f} vs capped truth",
          delta_color="off")
c3.metric(f"Failure within {fail['window_cycles']} cycles", f"{fail['failure_probability']:.0%}",
          help=f"Flagged as imminent at or above {fail['threshold']:.0%} (threshold tuned on training engines).")
c4.metric("Anomaly", "Yes" if anom["is_anomaly"] else "No")

box = {"HIGH": st.error, "MEDIUM": st.warning, "LOW": st.success}[rec["urgency"]]
box(f"**{rec['urgency']}** — {rec['action']}  \n{rec['reason']}"
    + ("" if rec["persisted"] else "  \n_(not saved: database unavailable)_"))

left, right = st.columns(2)
with left:
    points = pd.DataFrame(traj["points"])
    capped_truth = engine.set_index("cycle")["rul"].clip(upper=rul["rul_cap"])
    fig = go.Figure()
    fig.add_scatter(x=points.cycle, y=points.predicted_rul, name="Predicted RUL")
    fig.add_scatter(x=capped_truth.index[capped_truth.index <= cycle], y=capped_truth[capped_truth.index <= cycle],
                    name="True RUL (capped)", line=dict(dash="dash"))
    fig.update_layout(title="Remaining useful life over this engine's history", xaxis_title="Cycle",
                      yaxis_title="Cycles", height=380)
    st.plotly_chart(fig, use_container_width=True)
with right:
    contrib = pd.Series(xai["feature_contributions"]).iloc[::-1]
    fig = go.Figure(go.Bar(x=contrib.values, y=contrib.index, orientation="h",
                           marker_color=["#c0392b" if v > 0 else "#2e86c1" for v in contrib.values]))
    fig.update_layout(title="Why: SHAP contributions to failure risk (red = raises risk)", height=380,
                      xaxis_title=xai["output"])
    st.plotly_chart(fig, use_container_width=True)

sensor = st.selectbox("Sensor trend", SENSORS, index=SENSORS.index("s11"),
                      format_func=lambda s: f"{s}: {SENSOR_DESCRIPTIONS[s]}")
fig = go.Figure()
fig.add_scatter(x=engine.cycle, y=engine[sensor], name=sensor, line=dict(color="#95a5a6"))
fig.add_scatter(x=history.cycle, y=history[sensor], name="sent to API")
fig.add_vline(x=cycle, line_dash="dot")
fig.update_layout(height=300, xaxis_title="Cycle", showlegend=False)
st.plotly_chart(fig, use_container_width=True)
