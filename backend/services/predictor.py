import logging
from typing import Optional

import joblib
import pandas as pd

from backend.core.config import settings
from ml.data.cmapss import SENSORS
from ml.features.preprocessing import latest_features
from ml.training.intervals import interval_bounds

log = logging.getLogger(__name__)


class PredictorService:
    """Loads the model bundle produced by ml/training/train.py. If loading fails,
    the reason is kept and every prediction endpoint answers 503 with it, instead
    of crashing on a missing model."""

    def __init__(self, path: Optional[str] = None):
        self.path = path or settings.MODEL_PATH
        self.bundle = None
        self.error: Optional[str] = None
        self.load()

    def load(self):
        try:
            self.bundle = joblib.load(self.path)
            self.error = None
            log.info("Loaded model bundle: RUL=%s, failure=%s",
                     self.bundle["rul"]["algorithm"], self.bundle["failure"]["algorithm"])
        except Exception as e:  # noqa: BLE001 - reported via /health and 503s
            self.bundle = None
            self.error = f"{type(e).__name__}: {e}"
            log.error("Could not load model bundle from %s: %s", self.path, self.error)

    @property
    def loaded(self) -> bool:
        return self.bundle is not None

    def features(self, history) -> pd.DataFrame:
        readings = pd.DataFrame([r.model_dump() for r in history.readings])[["cycle"] + SENSORS]
        return latest_features(readings)

    def predict_rul(self, X) -> float:
        return float(max(0.0, self.bundle["rul"]["model"].predict(X)[0]))

    def rul_interval(self, rul: float) -> Optional[dict]:
        """Conformal interval around a RUL prediction, or None for older bundles without one."""
        cfg = self.bundle["rul"].get("interval")
        if not cfg:
            return None
        lo, hi = interval_bounds(rul, cfg["halfwidth"], self.bundle["rul"]["cap"])
        return {"lower": lo, "upper": hi, "confidence": 1 - cfg["alpha"]}

    def failure_probability(self, X) -> float:
        return float(self.bundle["failure"]["model"].predict_proba(X)[0, 1])

    def anomaly_score(self, X) -> float:
        # IsolationForest.score_samples: lower = more anomalous.
        return float(self.bundle["anomaly"]["model"].score_samples(X)[0])


predictor_service = PredictorService()
