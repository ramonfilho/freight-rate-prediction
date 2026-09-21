"""Tests for the cleaning and feature code. Run: python -m pytest -q"""
import numpy as np
import pandas as pd

from src.predict import with_range
from src.prepare import FEATURES, Encoder, add_features, city_coordinates, clean


def loads() -> pd.DataFrame:
    return pd.DataFrame({
        "pickup": ["Richmond", "Richmond", "Newtown"],
        "delivery": ["Baltimore", "Baltimore", "Baltimore"],
        "pickup_lat": [38.1, 38.1, np.nan], "pickup_lon": [-76.8, -76.8, np.nan],
        "delivery_lat": [38.2, 38.2, 38.2], "delivery_lon": [-72.7, -72.7, -72.7],
        "distance": [274.3, 274.3, 300.0],
        "equipment": ["Dry Van", "Reefer", "Flatbed"],
        "weight": [-30658.0, np.nan, 20000.0],
        "date": pd.to_datetime(["2025-01-01", "2025-06-15", "2025-12-03"]),
    })


def test_negative_weight_becomes_positive():
    out = clean(loads(), weight_fill=31000.0)
    assert out.loc[0, "weight"] == 30658.0


def test_missing_weight_gets_the_given_fill_value():
    out = clean(loads(), weight_fill=31000.0)
    assert out.loc[1, "weight"] == 31000.0
    assert out["weight"].notna().all()


def test_date_features_are_month_and_day_of_week():
    coords = city_coordinates(loads())
    out = add_features(loads(), coords)
    assert out.loc[1, "month"] == 6
    assert out.loc[2, "dow"] == 2  # 2025-12-03 is a Wednesday


def test_city_without_coordinates_gets_them_from_the_lookup():
    frame = pd.DataFrame({"pickup": ["Richmond"], "delivery": ["Baltimore"],
                          "date": pd.to_datetime(["2025-12-01"])})
    coords = city_coordinates(loads())
    out = add_features(frame, coords)
    assert out.loc[0, "pickup_lat"] == 38.1
    assert out.loc[0, "delivery_lon"] == -72.7


def test_model_input_has_the_feature_columns_in_order():
    coords = city_coordinates(loads())
    prepared = add_features(clean(loads(), 31000.0), coords)
    x = Encoder().fit(prepared).transform(prepared)
    assert list(x.columns) == FEATURES
    assert x["equipment"].notna().all()


def test_price_range_sits_around_the_point_prediction():
    frame = pd.DataFrame({"predicted_rate": [1000.0, 2500.0]})
    out = with_range(frame, {"p10_factor": 0.97, "p90_factor": 1.06})
    assert (out["p10"] < out["predicted_rate"]).all()
    assert (out["p90"] > out["predicted_rate"]).all()
    assert out.loc[0, "p10"] == 970.0 and out.loc[0, "p90"] == 1060.0
