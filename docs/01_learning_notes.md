# Step 01: Why inventory and join keys matter

- `WAFER_ID`: wafer identifier, **not automatically a production lot**.
- `STAGE`: process stage (A or B). Avoid combining A and B into a single SPC chart before reviewing process differences.
- `TIMESTAMP`: process measurement time; do not interpret as a calendar date without source documentation.
- `AVG_REMOVAL_RATE`: label or output of a wafer-stage process segment. MRR means material removal rate.
- `(WAFER_ID, STAGE)` is the dataset-level matching key for sensor samples and one corresponding quality output.

## Why do we inspect the raw data first?

If a wafer has 300 time samples but only one quality target, a machine learning model must not treat its 300 samples as 300 separate independent wafers.

The first script **only counts and validates** raw data, labels, and matching keys. It **does not** clean duplicate timestamps, delete suspicious MRR values, construct SPC limits, or train a prediction model.
