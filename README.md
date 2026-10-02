# Spotter Machine Learning Engineer Assessment — Freight Rate Prediction

## Approach

The development data contains 48,000 labeled loads from January through October 2025. The final validation set contains 12,000 loads from November–December 2025.

I used chronological validation rather than a random split so that the validation period better represents prediction on future loads.

### Feature engineering

The model uses only features available both in the development/validation data and in the supplied December chart input:

- pickup
- delivery
- distance
- equipment
- weight
- date-derived features: year, month, day, day of week, day of year, week of year, days since 2025-01-01
- route = pickup + delivery
- distance_per_weight
- weight_missing

The model deliberately does not depend on fields that are absent from the December chart input, such as market_index and quote_signal.

Missing weight values are imputed with the development-set median for the engineered distance_per_weight feature. LightGBM handles the remaining numeric missing values natively.

### Model

I used LightGBM with:

- objective: regression on `log1p(posted_rate)`
- 300 estimators
- learning rate: 0.03
- 63 leaves
- subsample: 0.90
- column subsampling: 0.90
- L2 regularization: 1.0
- random seed: 42

The log-target transformation was selected after chronological validation because the target is right-skewed and the log-target model gave lower MAE and percentage error than the tested alternatives.

## Chronological validation

Two forward-looking monthly validation checks were used:

| Train period | Validation period | MAE | RMSE | MAPE |
|---|---|---:|---:|---:|
| Jan–Aug 2025 | Sep 2025 | 17.17 | 186.22 | 0.47% |
| Jan–Sep 2025 | Oct 2025 | 17.98 | 159.42 | 0.59% |

These are internal validation measurements only. Spotter's final validation metric is calculated after submission.

## Data quality findings

- 48,000 labeled rows were available.
- `posted_rate` was complete.
- `market_index` had 374 missing values.
- `weight` had 300 missing values.
- There were 64 pickup cities and 64 delivery cities.
- The target was strongly right-skewed, making a log-target transformation useful.
- The December chart input contains 31 dates from 2025-12-01 through 2025-12-31.

## Output

The final submission file is:

`validation_predictions.csv`

with exactly:

```text
load_id,predicted_rate
```

The December predictions are stored in:

`data/december_chart_inputs.csv`

The supplied scorer was run successfully and produced:

`scorer_results/candidate_december.png`

## Reproduce

```bash
pip install -r requirements.txt
python train_predict.py
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

## Files

- `train_predict.py` — feature engineering, training, and prediction
- `validation_predictions.csv` — 12,000 final validation predictions
- `data/december_chart_inputs.csv` — 31 December predictions
- `score.py` — supplied validation/chart scorer
- `scorer_results/candidate_december.png` — fixed December prediction chart
- `report.pdf` — assessment report
- `loom_script.md` — 2–3 minute presentation script
