"""Load, clean and build features.

The same functions run on the training loads, on the validation loads and on
the fixed December lane, so there is one code path from raw file to model
input. Anything learned from data (the weight fill value, the city codes) is
computed on the training slice and reused, never recomputed on the data being
predicted.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"

# Location enters as coordinates, not as city names. On the chronological
# holdout the model with names did worse (MAE $132 vs $122), and coordinates
# also cover the 8 validation cities that never appear in training.
CATEGORICAL = ["equipment"]
NAMES = ["pickup", "delivery"]
NUMERIC = [
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
    "distance", "weight", "month", "dow",
]
FEATURES = CATEGORICAL + NUMERIC
TARGET = "posted_rate"


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(DATA / "train_test.csv", parse_dates=["date"])
    validation = pd.read_csv(DATA / "validation.csv", parse_dates=["date"])
    december = pd.read_csv(DATA / "december_chart_inputs.csv", parse_dates=["date"])
    return train, validation, december


def city_coordinates(*frames: pd.DataFrame) -> pd.DataFrame:
    """One fixed coordinate per city name in every file, so a lookup table
    fills lat/lon for rows that only carry the city name (the December lane)."""
    parts = []
    for frame in frames:
        parts.append(frame[["pickup", "pickup_lat", "pickup_lon"]]
                     .set_axis(["city", "lat", "lon"], axis=1))
        parts.append(frame[["delivery", "delivery_lat", "delivery_lon"]]
                     .set_axis(["city", "lat", "lon"], axis=1))
    return pd.concat(parts).dropna().drop_duplicates("city").set_index("city")


def clean(frame: pd.DataFrame, weight_fill: float) -> pd.DataFrame:
    out = frame.copy()
    # 292 training loads carry a negative weight. Their magnitude (median 31.7k lb)
    # and their rate per mile ($2.16) match the positive loads, so this is a sign
    # error at entry, not a different kind of load. Flip the sign instead of
    # dropping the rows.
    out["weight"] = out["weight"].abs()
    # 300 training loads have no weight. The fill value is the median of the
    # training slice and is passed in, so the data being predicted never sets it.
    out["weight"] = out["weight"].fillna(weight_fill)
    return out


def add_features(frame: pd.DataFrame, coords: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for side in ("pickup", "delivery"):
        if f"{side}_lat" not in out.columns:
            out[f"{side}_lat"] = out[side].map(coords["lat"])
            out[f"{side}_lon"] = out[side].map(coords["lon"])
    # With market_index out of the model, the date is what tells the model
    # "when": month carries the seasonal level, day of week the weekly pattern.
    out["month"] = out["date"].dt.month
    out["dow"] = out["date"].dt.dayofweek
    return out


class Encoder:
    """Maps category names (equipment, and city names when an experiment asks
    for them) to integer codes for the tree model. A name not seen in training
    becomes NaN, which the model treats as its own bucket."""

    def fit(self, frame: pd.DataFrame) -> "Encoder":
        self.maps = {
            col: {value: code for code, value in enumerate(sorted(frame[col].dropna().unique()))}
            for col in CATEGORICAL + NAMES
        }
        return self

    def transform(self, frame: pd.DataFrame, features: list[str] = FEATURES) -> pd.DataFrame:
        out = frame.copy()
        for col in CATEGORICAL + NAMES:
            if col in features:
                out[col] = out[col].map(self.maps[col]).astype(float)
        return out[features]


def categorical_mask(features: list[str]) -> list[bool]:
    return [name in CATEGORICAL + NAMES for name in features]
