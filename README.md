# Freight rate prediction

Predicts the posted rate (US dollars) of a truckload from its lane, distance,
equipment, weight and pickup date. Built for the Spotter Machine Learning
Engineer assessment.

## Run

```bash
python -m pip install -r requirements.txt
python -m pytest -q        # tests for the cleaning and feature code
python -m src.validate     # chronological holdout, model comparison, price range, figures
python -m src.predict      # writes validation_predictions.csv and fills data/december_chart_inputs.csv
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

`score.py` is the scorer supplied with the assessment. It checks both output
files and draws `scorer_results/candidate_december.png`.

## What is in `src/`

| File | What it does |
|---|---|
| `prepare.py` | Loads the three CSVs, cleans weight, builds features. One code path for training, validation and the December lane. |
| `validate.py` | Splits January to August (fit) from September and October (measure), compares a baseline, a linear model and gradient boosting, and runs the feature ablations. Writes `reports/holdout_metrics.json`. |
| `predict.py` | Refits the chosen model on all labeled loads, writes the two deliverables, and adds a P10 to P90 price range next to each prediction (`reports/*_with_range.csv`). |
| `tests/test_prepare.py` | Six tests: negative weight flipped, missing weight filled, date features, coordinate lookup, model input columns, price range around the point. |

## How the model was validated

The labeled loads run from January to October 2025 and the loads to predict
are November and December. The holdout therefore sits after the training
slice in time: the model is fit on January to August and measured on
September and October, and only then refit on everything to predict. A random
split would let the model see market conditions it is later asked to forecast.

Everything learned from data (the weight fill value, the equipment codes) is
computed on the training slice and reused on the data being predicted.

## Results on the September to October holdout (9,523 loads)

| Model | MAE | MAPE |
|---|---|---|
| Baseline: median rate per mile times distance | $257 | 11.6% |
| Linear model on log rate, with distance, log distance and equipment | $134 | 6.0% |
| Gradient boosting, raw target | $138 | 6.4% |
| **Gradient boosting, log target (chosen)** | **$122** | **5.4%** |
| Chosen model plus `market_index` and `quote_signal` | $126 | 5.4% |
| Chosen model plus city names | $132 | 5.8% |
| Chosen model without coordinates | $130 | 5.9% |

The chosen model uses equipment, pickup and delivery coordinates, distance,
weight, month and day of week. The two columns absent from the December lane
inputs (`market_index`, `quote_signal`) did not improve the holdout, so the
final model does not use them and the same model serves both deliverables.
City names memorize lanes and cannot cover the 8 validation cities that never
appear in training; coordinates do both jobs and scored better.

## Price range, not only a point

A dispatcher works with a range. The range around each prediction is the
point prediction times two factors: the 10th and 90th percentile of
actual/predicted measured on the holdout, months the model had not trained
on. Calibrated on September alone and checked on October, the range covered
84.7% of the loads, close to the 80% it aims for. The factors are 0.974 and
1.061, so the range is about 9% of the rate wide. On the December lane the
point sits between $839 and $848 and the range between $819 and $899.

## Data quality decisions

- 292 training loads (and 145 validation loads) carry a negative weight. Their
  magnitude and their rate per mile match the positive loads, so the sign is
  flipped instead of dropping the rows.
- 300 training loads have no weight. They receive the training median. The
  median is close to identical across equipment types, so a single value is
  enough.
- `market_index` has 374 gaps. It left the model, so no fill was needed.
- `load_id` is an identifier and never enters the model.

## If this went to production

The same three files are the shape of a production job. Training would run
on a schedule as `src.validate` plus `src.predict` do here, the model would
be promoted only if the holdout MAE stayed inside the band seen at launch,
and the monitor would compare predicted against realized rate per mile every
week, by distance band and equipment, and stop publishing when the drift
crossed a limit. A new model would take a share of the loads first and the
rest only after its realized error matched the old one. This is how the
lead-scoring system I run at Bring Data is operated, with a quality gate
before the registry, daily checks that stop the send, and a canary rollout.

## Known limits

- December does not exist in the training period, so the model carries the
  level of the last months it saw. The fixed-lane chart moves only with the
  day of week, which is what the remaining inputs allow.
- The largest errors sit on long hauls above 1,500 miles (MAE $230 on that
  band against $22 under 200 miles); errors scale with the rate.
