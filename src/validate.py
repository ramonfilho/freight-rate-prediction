"""Chronological validation and model comparison.

Training data covers January to October 2025 and the loads to predict are
November and December, so the holdout has to sit after the training slice in
time: fit on January to August, measure on September and October. A random
split would let the model see the future it is later asked to predict.

Run:  python -m src.validate
Writes reports/holdout_metrics.json and reports/figures/*.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from src.prepare import (FEATURES, NAMES, TARGET, Encoder, add_features,
                         categorical_mask, city_coordinates, clean, load_raw)

REPORTS = Path(__file__).resolve().parents[1] / "reports"
SPLIT_DATE = "2025-09-01"

# Columns present in train_test.csv and validation.csv but absent from the
# December lane inputs. The ablation below checks whether they earn their place.
EXTRA = ["market_index", "quote_signal"]


def metrics(y: pd.Series, pred: np.ndarray) -> dict[str, float]:
    err = y.to_numpy() - pred
    return {
        "MAE": round(float(np.mean(np.abs(err))), 1),
        "MAPE": round(float(np.mean(np.abs(err) / y.to_numpy()) * 100), 2),
        "RMSE": round(float(np.sqrt(np.mean(err ** 2))), 1),
    }


def gbm(features: list[str]) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.06, max_leaf_nodes=63,
        categorical_features=categorical_mask(features), random_state=0,
    )


def fit_predict_gbm(train: pd.DataFrame, holdout: pd.DataFrame, features: list[str],
                    encoder: Encoder, log_target: bool = True) -> np.ndarray:
    x_train = encoder.transform(train, features)
    x_hold = encoder.transform(holdout, features)
    y = np.log1p(train[TARGET]) if log_target else train[TARGET]
    model = gbm(features).fit(x_train, y)
    pred = model.predict(x_hold)
    return np.expm1(pred) if log_target else pred


def linear_design(frame: pd.DataFrame) -> pd.DataFrame:
    """Distance, log distance and equipment dummies. The log term lets a straight
    line bend: rate per mile falls from $2.80 under 200 miles to $1.95 over 1,500."""
    x = pd.DataFrame({
        "distance": frame["distance"],
        "log_distance": np.log(frame["distance"]),
    })
    return pd.concat([x, pd.get_dummies(frame["equipment"], dtype=float)], axis=1)


def main() -> None:
    train_all, validation, _ = load_raw()
    coords = city_coordinates(train_all, validation)

    train = train_all[train_all["date"] < SPLIT_DATE]
    holdout = train_all[train_all["date"] >= SPLIT_DATE]
    weight_fill = float(train["weight"].abs().median())
    train = add_features(clean(train, weight_fill), coords)
    holdout = add_features(clean(holdout, weight_fill), coords)
    encoder = Encoder().fit(train)
    y_hold = holdout[TARGET]

    results: dict[str, dict] = {}

    # 1. Baseline: median rate per mile of the training slice times distance.
    rate_per_mile = float((train[TARGET] / train["distance"]).median())
    results["baseline_median_rate_per_mile"] = metrics(y_hold, rate_per_mile * holdout["distance"])

    # 2. Linear model on log target with distance, log distance and equipment.
    ridge = Ridge(alpha=1.0).fit(linear_design(train), np.log1p(train[TARGET]))
    results["linear_log_distance"] = metrics(y_hold, np.expm1(ridge.predict(linear_design(holdout))))

    # 3. Gradient boosting, raw target vs log target, same features.
    results["gbm_raw_target"] = metrics(y_hold, fit_predict_gbm(train, holdout, FEATURES, encoder, log_target=False))
    results["gbm_log_target"] = metrics(y_hold, fit_predict_gbm(train, holdout, FEATURES, encoder))

    # 4. Ablations. (a) Do the columns absent from the December lane help?
    results["gbm_plus_market_index_and_quote_signal"] = metrics(
        y_hold, fit_predict_gbm(train, holdout, FEATURES + EXTRA, encoder))
    # (b) City names on top of coordinates: memorizes lanes, cannot cover new cities.
    results["gbm_plus_city_names"] = metrics(y_hold, fit_predict_gbm(train, holdout, FEATURES + NAMES, encoder))
    # (c) No location at all: what the coordinates are worth.
    no_location = [f for f in FEATURES if not f.endswith(("_lat", "_lon"))]
    results["gbm_without_location"] = metrics(y_hold, fit_predict_gbm(train, holdout, no_location, encoder))

    # 5. Price range around the point prediction. The ratio actual/predicted on
    # months the model did not train on gives the two factors. Calibrated on
    # September and checked on October: an 80% range should cover about 80%.
    chosen = fit_predict_gbm(train, holdout, FEATURES, encoder)
    ratio = y_hold.to_numpy() / chosen
    september = (holdout["date"] < "2025-10-01").to_numpy()
    low_sep, high_sep = np.quantile(ratio[september], [0.10, 0.90])
    actual_oct, pred_oct = y_hold.to_numpy()[~september], chosen[~september]
    covered = (actual_oct >= pred_oct * low_sep) & (actual_oct <= pred_oct * high_sep)
    low, high = np.quantile(ratio, [0.10, 0.90])   # final factors: whole holdout
    results["price_range_p10_p90"] = {
        "how": "point prediction times the 10th and 90th percentile of actual/predicted on the holdout",
        "check_calibrated_on_september_coverage_on_october_pct": round(float(covered.mean() * 100), 1),
        "p10_factor": round(float(low), 4), "p90_factor": round(float(high), 4),
        "width_pct_of_rate": round(float((high - low) * 100), 1),
    }

    # 6. Error of the chosen model by distance band, to show where it misses.
    bands = pd.cut(holdout["distance"], [0, 200, 400, 800, 1500, 3500])
    by_band = (pd.DataFrame({"band": bands.astype(str), "y": y_hold, "p": chosen})
               .groupby("band", sort=False)
               .apply(lambda g: pd.Series(metrics(g["y"], g["p"].to_numpy())))
               .round(2))
    results["chosen_model_by_distance_band"] = by_band.to_dict(orient="index")
    results["split"] = {"train": "2025-01-01 to 2025-08-31", "holdout": "2025-09-01 to 2025-10-31",
                        "train_rows": int(len(train)), "holdout_rows": int(len(holdout))}

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "holdout_metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    figures(train_all, holdout, chosen)


def figures(train_all: pd.DataFrame, holdout: pd.DataFrame, pred: np.ndarray) -> None:
    fig_dir = REPORTS / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    rpm = train_all[TARGET] / train_all["distance"]

    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    sample = train_all.sample(5000, random_state=0)
    ax.scatter(sample["distance"], sample[TARGET], s=3, alpha=0.3)
    ax.set_xlabel("distance (miles)"); ax.set_ylabel("posted rate ($)")
    ax.set_title("Rate grows with distance, and the spread grows with it")
    fig.tight_layout(); fig.savefig(fig_dir / "rate_vs_distance.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    bands = pd.cut(train_all["distance"], [0, 200, 400, 800, 1500, 3500])
    rpm.groupby(bands, observed=True).median().plot.bar(ax=ax, color="#064A56")
    ax.set_ylabel("median rate per mile ($)"); ax.set_xlabel("distance band (miles)")
    ax.set_title("Rate per mile falls with distance"); ax.tick_params(axis="x", rotation=0)
    fig.tight_layout(); fig.savefig(fig_dir / "rate_per_mile_by_distance.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    monthly = rpm.groupby(train_all["date"].dt.month).median()
    ax.plot(monthly.index, monthly.values, marker="o", color="#064A56")
    ax.set_xlabel("month of 2025"); ax.set_ylabel("median rate per mile ($)")
    ax.set_title("Seasonal level, January to October")
    fig.tight_layout(); fig.savefig(fig_dir / "rate_per_mile_by_month.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
    ax.scatter(holdout[TARGET], pred, s=3, alpha=0.3)
    lim = [0, float(holdout[TARGET].max())]
    ax.plot(lim, lim, color="gray", linewidth=1)
    ax.set_xlabel("actual rate ($), Sep-Oct holdout"); ax.set_ylabel("predicted rate ($)")
    ax.set_title("Chosen model on the holdout")
    fig.tight_layout(); fig.savefig(fig_dir / "holdout_actual_vs_predicted.png"); plt.close(fig)


if __name__ == "__main__":
    main()
