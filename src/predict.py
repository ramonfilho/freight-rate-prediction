"""Fit the final model on every labeled load and write the two deliverables.

Run:  python -m src.predict
Writes validation_predictions.csv (12,000 loads, November and December) and
fills the predicted_rate column of data/december_chart_inputs.csv (31 days of
the fixed Lexington to Fort Wayne lane). Then run score.py to check both.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.prepare import (DATA, FEATURES, TARGET, Encoder, add_features,
                         categorical_mask, city_coordinates, clean, load_raw)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    train, validation, december = load_raw()
    coords = city_coordinates(train, validation)

    # Everything learned from data comes from the labeled loads only.
    weight_fill = float(train["weight"].abs().median())
    train = add_features(clean(train, weight_fill), coords)
    encoder = Encoder().fit(train)

    model = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.06, max_leaf_nodes=63,
        categorical_features=categorical_mask(FEATURES), random_state=0,
    ).fit(encoder.transform(train), np.log1p(train[TARGET]))

    def predict(frame: pd.DataFrame) -> np.ndarray:
        prepared = add_features(clean(frame, weight_fill), coords)
        return np.expm1(model.predict(encoder.transform(prepared)))

    # Deliverable 1: the 12,000 validation loads, in the template's order.
    template = pd.read_csv(DATA / "validation_predictions_template.csv")
    scored = validation.assign(predicted_rate=predict(validation))[["load_id", "predicted_rate"]]
    out = template[["load_id"]].merge(scored, on="load_id", how="left")
    out["predicted_rate"] = out["predicted_rate"].round(2)
    out.to_csv(ROOT / "validation_predictions.csv", index=False)

    # Deliverable 2: the fixed December lane, keeping the file's seven columns.
    december["predicted_rate"] = predict(december).round(2)
    december.assign(date=december["date"].dt.strftime("%Y-%m-%d")).to_csv(
        DATA / "december_chart_inputs.csv", index=False)

    # Price range (P10 to P90) from the factors measured in src.validate on the
    # holdout. Kept in separate files so the scorer's files keep their columns.
    factors = json.loads((ROOT / "reports" / "holdout_metrics.json").read_text())["price_range_p10_p90"]
    with_range(out, factors).to_csv(ROOT / "reports" / "validation_predictions_with_range.csv", index=False)
    band = with_range(december[["date", "predicted_rate"]], factors)
    band.assign(date=band["date"].dt.strftime("%Y-%m-%d")).to_csv(
        ROOT / "reports" / "december_lane_with_range.csv", index=False)
    december_band_figure(band, ROOT / "reports" / "figures" / "december_with_range.png")

    print(f"validation_predictions.csv: {len(out):,} rows, "
          f"median ${out['predicted_rate'].median():,.0f}, "
          f"missing {int(out['predicted_rate'].isna().sum())}")
    print(f"december lane: ${december['predicted_rate'].min():,.0f} to "
          f"${december['predicted_rate'].max():,.0f}")


def with_range(frame: pd.DataFrame, factors: dict) -> pd.DataFrame:
    """Adds p10 and p90 columns: the point prediction scaled by the two factors."""
    out = frame.copy()
    out["p10"] = (out["predicted_rate"] * factors["p10_factor"]).round(2)
    out["p90"] = (out["predicted_rate"] * factors["p90_factor"]).round(2)
    return out


def december_band_figure(band: pd.DataFrame, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10.8, 4.2), dpi=150)
    ax.fill_between(band["date"], band["p10"], band["p90"], color="#064A56", alpha=0.15,
                    label="P10 to P90 range")
    ax.plot(band["date"], band["predicted_rate"], color="#064A56", linewidth=2.2, marker="o",
            markersize=3, label="point prediction")
    ax.set_title("December 2025, Lexington to Fort Wayne: predicted rate and price range",
                 loc="left", fontsize=12, fontweight="bold")
    ax.set_ylabel("rate ($)"); ax.legend(loc="upper right", frameon=False)
    ax.grid(axis="y", color="#D9E2E4", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False); ax.tick_params(axis="x", rotation=35)
    fig.tight_layout(); fig.savefig(output, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
