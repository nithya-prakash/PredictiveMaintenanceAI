"""Load test for the prediction API:

    locust -f loadtest/locustfile.py --host http://localhost:8001

Env: API_TOKEN (if the API requires one). Each simulated user replays a real
FD001 test engine's last 30 cycles (data/raw/cmapss, downloaded by ml.data.cmapss).
The API has a per-client rate limit (settings.RATE_LIMIT_PER_MINUTE); 429s are
counted as expected throttling, not failures, so run with a trusted API_TOKEN
or a raised limit to measure model latency rather than the limiter.
"""
import os
import random

from locust import HttpUser, between, task

from ml.data.cmapss import DEFAULT_DIR, SENSORS, load_test

_TEST = load_test(DEFAULT_DIR)
ENGINES = {eid: g.sort_values("cycle").tail(30) for eid, g in _TEST.groupby("engine_id")}
HEADERS = {"X-API-Key": os.environ["API_TOKEN"]} if os.environ.get("API_TOKEN") else {}


def payload(engine_id: int) -> dict:
    rows = ENGINES[engine_id][["cycle", *SENSORS]].to_dict(orient="records")
    return {"machine_id": engine_id, "readings": rows}


class Operator(HttpUser):
    wait_time = between(0.2, 1)

    def _post(self, path: str):
        with self.client.post(f"/api/v1/{path}", json=payload(random.choice(list(ENGINES))),
                              headers=HEADERS, name=path, catch_response=True) as r:
            if r.status_code == 429:
                r.success()

    @task(5)
    def predict_failure(self): self._post("predict-failure")

    @task(3)
    def predict_rul(self): self._post("predict-rul")

    @task(2)
    def anomaly(self): self._post("anomaly")

    @task(1)
    def explain(self): self._post("explain")
