"""Replay real FD001 test-engine data through the API to exercise the drift gauges.

    API_URL=http://localhost:8011 API_TOKEN=... python -m scripts.replay_drift_demo

Phase 1 sends windows from early life (up to cycle 60), phase 2 from each engine's
last cycles (real degradation: shifts stay under the 0.5 std alert level), phase 3
re-sends phase 2 with a SYNTHETIC calibration offset (+1.5 training std on s2 and s11)
to show what a real alert looks like. Phase 3 data is injected, not measured.
"""
import json
import os
import sys
import time
from pathlib import Path

import httpx

from ml.data.cmapss import DEFAULT_DIR, SENSORS, load_test

API = os.environ.get("API_URL", "http://localhost:8001").rstrip("/") + "/api/v1/predict-failure"
HEADERS = {"X-API-Key": os.environ["API_TOKEN"]} if os.environ.get("API_TOKEN") else {}


def window(engine, end_cycle, offsets=None):
    rows = engine[engine.cycle <= end_cycle].tail(30).copy()
    for sensor, delta in (offsets or {}).items():
        rows[sensor] = rows[sensor] + delta
    return {"machine_id": int(engine.engine_id.iloc[0]), "readings": rows[["cycle", *SENSORS]].to_dict("records")}


def phase(name, engines, pick_end, delay, offsets=None):
    sent = 0
    for _, e in engines:
        r = httpx.post(API, json=window(e, pick_end(e), offsets), headers=HEADERS, timeout=30)
        sent += r.status_code == 200
        if r.status_code == 429:
            print("rate limited: set API_TOKEN to a trusted token", file=sys.stderr)
            return
        time.sleep(delay)
    print(f"{name}: {sent} requests accepted")


if __name__ == "__main__":
    delay = float(os.environ.get("DELAY", "0.05"))
    engines = list(load_test(DEFAULT_DIR).sort_values(["engine_id", "cycle"]).groupby("engine_id"))
    for _ in range(2):  # 2 passes x 100 engines = 200 observations per phase (fills the window)
        phase("early life", engines, lambda e: 60, delay)
    time.sleep(10)
    for _ in range(2):
        phase("late life", engines, lambda e: int(e.cycle.max()), delay)
    time.sleep(10)
    ref = json.loads(Path("models/drift_reference.json").read_text())["sensors"]
    offsets = {s: 1.5 * ref[s]["std"] for s in ("s2", "s11")}
    for _ in range(2):
        phase("late life + SYNTHETIC offset", engines, lambda e: int(e.cycle.max()), delay, offsets)
