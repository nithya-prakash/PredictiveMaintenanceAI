"""Input-drift monitor.

Keeps the last WINDOW observed sensor vectors (the latest reading of each
prediction request) and exposes, per sensor, how far their mean sits from the
training mean in training standard deviations. Prometheus gauges feed the
Grafana "Input drift" dashboard.

What it can and cannot say: a shift means "incoming data no longer looks like
the training population". In a turbofan fleet a shift also appears when engines
simply age (degradation moves sensors), so a shift is a prompt to look, not
proof that the model is wrong. Needs MIN_OBSERVATIONS vectors before reporting.
"""
import json
import logging
from collections import deque
from pathlib import Path
from threading import Lock
from typing import Dict, Optional

from prometheus_client import Gauge

log = logging.getLogger(__name__)
WINDOW = 200
MIN_OBSERVATIONS = 20
SHIFT_ALERT_STD = 0.5  # |mean shift| above half a training std counts as drifted

input_shift = Gauge("pdm_input_shift_std", "Window mean minus training mean, in training std", ["sensor"])
drifted_sensors = Gauge("pdm_drifted_sensors", f"Sensors with |shift| > {SHIFT_ALERT_STD} training std")
window_size = Gauge("pdm_drift_window_size", "Observations currently in the drift window")


class DriftMonitor:
    def __init__(self, reference: Optional[Dict[str, Dict[str, float]]], window: int = WINDOW):
        self.reference = reference or {}
        self._window = deque(maxlen=window)
        self._lock = Lock()

    @classmethod
    def from_file(cls, path: str) -> "DriftMonitor":
        try:
            return cls(json.loads(Path(path).read_text())["sensors"])
        except (OSError, KeyError, ValueError):
            log.warning("No drift reference at %s; drift monitoring disabled", path)
            return cls(None)

    def observe(self, reading: Dict[str, float]) -> None:
        if not self.reference:
            return
        with self._lock:
            self._window.append({s: float(reading[s]) for s in self.reference if s in reading})
            window_size.set(len(self._window))
            if len(self._window) >= MIN_OBSERVATIONS:
                self._publish()

    def shifts(self) -> Dict[str, float]:
        n = len(self._window)
        out = {}
        for s, ref in self.reference.items():
            vals = [r[s] for r in self._window if s in r]
            if vals and ref["std"] > 0:
                out[s] = (sum(vals) / len(vals) - ref["mean"]) / ref["std"]
        return out if n else {}

    def _publish(self) -> None:
        shifts = self.shifts()
        for s, v in shifts.items():
            input_shift.labels(sensor=s).set(v)
        drifted_sensors.set(sum(abs(v) > SHIFT_ALERT_STD for v in shifts.values()))
