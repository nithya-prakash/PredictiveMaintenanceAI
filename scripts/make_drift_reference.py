"""Writes models/drift_reference.json: per-sensor mean/std over the FD001 training rows.

    python -m scripts.make_drift_reference
"""
import json
from pathlib import Path

from ml.data.cmapss import DEFAULT_DIR, SENSORS, download, load_train

OUT = Path("models/drift_reference.json")

if __name__ == "__main__":
    train = load_train(download(DEFAULT_DIR))
    ref = {s: {"mean": float(train[s].mean()), "std": float(train[s].std())} for s in SENSORS}
    OUT.write_text(json.dumps({"source": "C-MAPSS FD001 training rows", "rows": int(len(train)), "sensors": ref}, indent=2))
    print(f"wrote {OUT} ({len(ref)} sensors, {len(train)} rows)")
