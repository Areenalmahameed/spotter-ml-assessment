from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor


RANDOM_STATE = 42
N_ESTIMATORS = 300
LEARNING_RATE = 0.03
NUM_LEAVES = 63


def make_features(frame: pd.DataFrame, weight_median: float) -> pd.DataFrame:
    x = frame.copy()
    d = pd.to_datetime(x["date"])

    x["year"] = d.dt.year
    x["month"] = d.dt.month
    x["day"] = d.dt.day
    x["dayofweek"] = d.dt.dayofweek
    x["dayofyear"] = d.dt.dayofyear
    x["weekofyear"] = d.dt.isocalendar().week.astype(int)
    x["days_since_2025_01_01"] = (
        d - pd.Timestamp("2025-01-01")
    ).dt.days.astype(int)

    x["route"] = x["pickup"].astype(str) + "__" + x["delivery"].astype(str)
    x["distance_per_weight"] = x["distance"] / (
        x["weight"].fillna(weight_median) + 1.0
    )
    x["weight_missing"] = x["weight"].isna().astype(int)

    # These fields are not present in the December chart input.
    # Restricting the model to the common feature set makes the final
    # December prediction fully reproducible from the supplied inputs.
    drop = [
        "load_id", "date",
        "pickup_lat", "pickup_lon",
        "delivery_lat", "delivery_lon",
        "market_index", "quote_signal",
        "posted_rate", "predicted_rate",
    ]
    return x.drop(columns=[c for c in drop if c in x.columns])


def align_categories(train_x: pd.DataFrame, other_x: pd.DataFrame):
    categorical = [c for c in train_x.columns if train_x[c].dtype == "object"]
    for c in categorical:
        categories = train_x[c].astype("category").cat.categories
        train_x[c] = train_x[c].astype("category")
        other_x[c] = other_x[c].astype("category").cat.set_categories(categories)
    return categorical


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/train_test.csv")
    parser.add_argument("--validation", default="data/validation.csv")
    parser.add_argument("--december", default="data/december_chart_inputs.csv")
    parser.add_argument("--output", default="validation_predictions.csv")
    args = parser.parse_args()

    train = pd.read_csv(args.train)
    validation = pd.read_csv(args.validation)
    december = pd.read_csv(args.december)

    weight_median = float(train["weight"].median())

    X_train = make_features(train, weight_median)
    X_validation = make_features(validation, weight_median)
    X_december = make_features(december, weight_median)

    categorical = align_categories(X_train, X_validation)
    # Re-align December using exactly the training categories.
    for c in categorical:
        categories = X_train[c].cat.categories
        X_december[c] = X_december[c].astype("category").cat.set_categories(categories)

    model = LGBMRegressor(
        objective="regression",
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        num_leaves=NUM_LEAVES,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        verbosity=-1,
        n_jobs=-1,
    )

    # Log-target training reduces the impact of the strongly right-skewed rate
    # distribution and performed best in chronological validation.
    model.fit(
        X_train,
        np.log1p(train["posted_rate"]),
        categorical_feature=categorical,
    )

    validation_pred = np.maximum(
        np.expm1(model.predict(X_validation)), 0.01
    )
    validation_predictions = pd.DataFrame(
        {
            "load_id": validation["load_id"].astype(str),
            "predicted_rate": validation_pred,
        }
    )
    validation_predictions.to_csv(args.output, index=False)

    december_pred = np.maximum(
        np.expm1(model.predict(X_december)), 0.01
    )
    december_out = december.copy()
    december_out["predicted_rate"] = december_pred
    december_out.to_csv(args.december, index=False)

    print(f"Wrote {len(validation_predictions):,} validation predictions to {args.output}")
    print(f"Wrote {len(december_out):,} December predictions to {args.december}")


if __name__ == "__main__":
    main()
