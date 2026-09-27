"""
NASA C-MAPSS turbofan engine degradation data (subset FD001).

Source: NASA Prognostics Center of Excellence, "Turbofan Engine Degradation
Simulation Data Set" (Saxena, Goebel, Simon & Eklund, 2008). FD001: 100 training
engines run until failure, 100 test engines stopped some time before failure,
one operating condition, one fault mode (HPC degradation). US-government
public data; it is downloaded, not stored in this repository.

Conventions used here are the standard ones from the RUL literature, so results
are comparable with published FD001 numbers:
- the 7 sensors that are constant in FD001 and the operating settings are dropped;
- the training RUL label is capped at 125 cycles (piecewise-linear degradation:
  early in life an engine is "healthy" and its exact remaining life is not
  predictable from the sensors).
"""
import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

DATA_URL = "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
DATA_SHA256 = "c9c5dec12a945a82e8bb4446589d7fb3cc057b5e5d81fa1a12e25ee9912ad3b2"
DEFAULT_DIR = Path("data/raw/cmapss")

COLUMNS = ["engine_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"] + [f"s{i}" for i in range(1, 22)]
# Sensors with (near-)zero variance in FD001 carry no information and are
# dropped, as in most FD001 studies.
CONSTANT_SENSORS = ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]
SENSORS = [f"s{i}" for i in range(1, 22) if f"s{i}" not in CONSTANT_SENSORS]  # 14 sensors
# C-MAPSS sensor meanings (Saxena et al. 2008). The 14 informative FD001 sensors.
SENSOR_DESCRIPTIONS = {
    "s2": "T24, total temperature at LPC outlet (°R)", "s3": "T30, total temperature at HPC outlet (°R)",
    "s4": "T50, total temperature at LPT outlet (°R)", "s7": "P30, total pressure at HPC outlet (psia)",
    "s8": "Nf, physical fan speed (rpm)", "s9": "Nc, physical core speed (rpm)",
    "s11": "Ps30, static pressure at HPC outlet (psia)", "s12": "phi, ratio of fuel flow to Ps30 (pps/psi)",
    "s13": "NRf, corrected fan speed (rpm)", "s14": "NRc, corrected core speed (rpm)",
    "s15": "BPR, bypass ratio", "s17": "htBleed, bleed enthalpy",
    "s20": "W31, HPT coolant bleed (lbm/s)", "s21": "W32, LPT coolant bleed (lbm/s)",
}
RUL_CAP = 125
FAILURE_WINDOW = 30  # "failure imminent" = true RUL <= 30 cycles


def download(target_dir: Path = DEFAULT_DIR) -> Path:
    """Downloads the official NASA archive, verifies its checksum, and extracts
    the C-MAPSS text files into target_dir. Skips work if already present."""
    target_dir = Path(target_dir)
    if (target_dir / "train_FD001.txt").exists():
        return target_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(DATA_URL, timeout=300) as resp:
        payload = resp.read()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != DATA_SHA256:
        raise RuntimeError(f"Checksum mismatch for the NASA archive: {digest} != {DATA_SHA256}")
    with zipfile.ZipFile(io.BytesIO(payload)) as outer:
        inner_name = next(n for n in outer.namelist() if n.endswith("CMAPSSData.zip"))
        with zipfile.ZipFile(io.BytesIO(outer.read(inner_name))) as inner:
            inner.extractall(target_dir)
    return target_dir


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)
    return df[["engine_id", "cycle"] + SENSORS]


def load_train(data_dir: Path = DEFAULT_DIR, subset: str = "FD001") -> pd.DataFrame:
    """Training engines, run to failure, with labels:
    rul (true), rul_capped (target for regression), failure_imminent (rul <= 30)."""
    df = _read(Path(data_dir) / f"train_{subset}.txt")
    last = df.groupby("engine_id")["cycle"].transform("max")
    return add_labels(df, last - df["cycle"])


def load_test(data_dir: Path = DEFAULT_DIR, subset: str = "FD001") -> pd.DataFrame:
    """Official test engines. Each trajectory stops before failure; NASA provides
    the true RUL at the last cycle, so the true RUL of every earlier cycle is
    that value plus the cycles remaining in the trajectory."""
    df = _read(Path(data_dir) / f"test_{subset}.txt")
    final_rul = pd.read_csv(Path(data_dir) / f"RUL_{subset}.txt", header=None).iloc[:, 0]
    final_rul.index = range(1, len(final_rul) + 1)  # engine ids are 1-based, in file order
    last = df.groupby("engine_id")["cycle"].transform("max")
    return add_labels(df, df["engine_id"].map(final_rul) + (last - df["cycle"]))


def add_labels(df: pd.DataFrame, rul: pd.Series) -> pd.DataFrame:
    df = df.copy()
    df["rul"] = rul.astype(int)
    df["rul_capped"] = df["rul"].clip(upper=RUL_CAP)
    df["failure_imminent"] = (df["rul"] <= FAILURE_WINDOW).astype(int)
    return df


if __name__ == "__main__":
    path = download()
    train, test = load_train(path), load_test(path)
    print(f"Data in {path}: {train.engine_id.nunique()} training engines ({len(train)} rows), "
          f"{test.engine_id.nunique()} test engines ({len(test)} rows).")
