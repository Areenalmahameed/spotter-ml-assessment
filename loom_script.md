# Loom Script — 2–3 Minutes

## 1. Introduction
Hi, I’m Areen. In this assessment I built a freight-rate prediction model for the Spotter Machine Learning Engineer role.

## 2. Data exploration
The development dataset contains 48,000 labeled loads covering January through October 2025. The target is `posted_rate`.

I found a few data-quality issues. There are missing values in `market_index` and `weight`, while the target itself is complete. The rate distribution is strongly right-skewed, and distance is an important driver of the target.

## 3. Feature engineering
I created date features including month, day of week, day of year, week of year, and a day index. I also created a route feature from pickup and delivery, plus distance per weight and a missing-weight indicator.

I intentionally restricted the final feature set to fields that are also available in the December chart input. This makes the final December predictions reproducible without inventing unavailable market or quote features.

## 4. Model choice
I tested tree-based approaches and compared standard regression with a log-transformed target. LightGBM with a log1p target performed best and was able to model nonlinear relationships between route, distance, equipment, weight, and date.

## 5. Validation
I used chronological validation instead of a random split. I validated on September after training on January through August, and then on October after training through September.

The model achieved approximately 17.17 MAE on September and 17.98 MAE on October. These are internal validation results; the final Spotter metric is calculated after submission.

## 6. Final prediction and outputs
After validation, I retrained the model on all 48,000 labeled development rows and generated predictions for all 12,000 validation loads.

I also generated predictions for all 31 December dates required by the assessment and ran the provided scorer successfully.

The final repository contains the training code, dependencies, predictions, report, and the December chart.

Thank you.
