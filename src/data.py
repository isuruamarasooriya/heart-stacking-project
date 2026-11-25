"""Download, load and validate the UCI Heart Disease (Cleveland) data.

Source: https://archive.ics.uci.edu/dataset/45/heart+disease
Authors: Janosi, Steinbrunn, Pfisterer, Detrano (1988). Public, de-identified data.
"""
from __future__ import annotations

import io
import urllib.request
import warnings
import zipfile
from pathlib import Path

import pandas as pd

# The raw file has no header row, so we name the 14 columns ourselves.
COLUMNS = [
    "age",       # years
    "sex",       # 1 = male, 0 = female
    "cp",        # chest pain type: 1 typical angina, 2 atypical, 3 non-anginal, 4 asymptomatic
    "trestbps",  # resting blood pressure (mm Hg)
    "chol",      # serum cholesterol (mg/dl)
    "fbs",       # fasting blood sugar > 120 mg/dl (1 = true)
    "restecg",   # resting ECG result (0, 1, 2)
    "thalach",   # maximum heart rate achieved
    "exang",     # exercise-induced angina (1 = yes)
    "oldpeak",   # ST depression induced by exercise
    "slope",     # slope of the peak exercise ST segment (1, 2, 3)
    "ca",        # number of major vessels coloured by fluoroscopy (0-3)
    "thal",      # 3 normal, 6 fixed defect, 7 reversible defect
    "num",       # diagnosis: 0 = no disease, 1-4 = disease of increasing severity
]
FEATURES = COLUMNS[:-1]
TARGET = "target"

CP_LABELS = {1: "typical angina", 2: "atypical angina", 3: "non-anginal pain", 4: "asymptomatic"}

MANUAL_HELP = (
    "Could not download the data automatically.\n"
    "Manual fix: open https://archive.ics.uci.edu/dataset/45/heart+disease, download the zip,\n"
    "extract 'processed.cleveland.data' and place it in data/raw/ (same name), then re-run."
)


def download_raw_data(raw_path: str | Path, urls: list[str], timeout: int = 30) -> Path:
    """Try each URL in turn and save the Cleveland file to raw_path."""
    raw_path = Path(raw_path)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for url in urls:
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                content = response.read()
            if url.endswith(".zip"):  # the zip contains several files; we only need one
                with zipfile.ZipFile(io.BytesIO(content)) as z:
                    content = z.read("processed.cleveland.data")
            raw_path.write_bytes(content)
            return raw_path
        except Exception as exc:  # network errors, bad zip, etc. -> try the next URL
            errors.append(f"  {url}: {exc}")
    raise RuntimeError("All download attempts failed:\n" + "\n".join(errors) + "\n" + MANUAL_HELP)


def load_raw(raw_path: str | Path) -> pd.DataFrame:
    """Read the headerless CSV. Missing values are written as '?' in this file."""
    return pd.read_csv(raw_path, header=None, names=COLUMNS, na_values="?")


def validate(df: pd.DataFrame, expected_rows: int | None = None) -> None:
    """Fail loudly if the data is not what we think it is."""
    missing_cols = set(COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns: {sorted(missing_cols)}")
    if not df["num"].dropna().isin([0, 1, 2, 3, 4]).all():
        raise ValueError("Column 'num' must only contain 0-4.")
    if not df["sex"].dropna().isin([0, 1]).all():
        raise ValueError("Column 'sex' must only contain 0 or 1.")
    if df["num"].isna().any():
        raise ValueError("The target column 'num' must not have missing values.")
    if expected_rows is not None and len(df) != expected_rows:
        warnings.warn(f"Expected {expected_rows} rows but found {len(df)}.")


def make_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split into features X and a binary target y (0 = healthy, 1 = heart disease).

    The original 'num' has 5 levels (0-4). Following common practice we ask the
    simpler question "is any heart disease present?", i.e. num > 0.
    """
    X = df[FEATURES].copy()
    y = (df["num"] > 0).astype(int).rename(TARGET)
    return X, y


def load_dataset(cfg: dict, raw_path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Download if needed, then load, validate and return (X, y)."""
    raw_path = Path(raw_path)
    if not raw_path.exists():
        print(f"Data not found at {raw_path} - downloading from UCI ...")
        download_raw_data(raw_path, cfg["data"]["urls"])
    df = load_raw(raw_path)
    validate(df, cfg["data"].get("expected_rows"))
    return make_xy(df)
