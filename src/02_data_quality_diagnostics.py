"""Step 02A: Read-only diagnostics for PHM 2016 CMP sensor events.

Run from CMP-Process-QE project root:
    python src/02_data_quality_diagnostics.py --source "C:\\...\\data\\2016 PHM Data Challenge"

Outputs under outputs/audit/:
    02_diagnostics_summary.json
    02_wafer_stage_inventory.csv
    02_exact_duplicate_examples.csv
    02_same_timestamp_different_values_examples.csv

Important: no original data files are modified, filtered, or overwritten.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

SENSOR_COLS = [
    "MACHINE_ID", "MACHINE_DATA", "TIMESTAMP", "WAFER_ID", "STAGE", "CHAMBER",
    "USAGE_OF_BACKING_FILM", "USAGE_OF_DRESSER", "USAGE_OF_POLISHING_TABLE",
    "USAGE_OF_DRESSER_TABLE", "PRESSURIZED_CHAMBER_PRESSURE",
    "MAIN_OUTER_AIR_BAG_PRESSURE", "CENTER_AIR_BAG_PRESSURE",
    "RETAINER_RING_PRESSURE", "RIPPLE_AIR_BAG_PRESSURE", "USAGE_OF_MEMBRANE",
    "USAGE_OF_PRESSURIZED_SHEET", "SLURRY_FLOW_LINE_A", "SLURRY_FLOW_LINE_B",
    "SLURRY_FLOW_LINE_C", "WAFER_ROTATION", "STAGE_ROTATION", "HEAD_ROTATION",
    "DRESSING_WATER_STATUS", "EDGE_AIR_BAG_PRESSURE",
]
LABEL_COLS = ["WAFER_ID", "STAGE", "AVG_REMOVAL_RATE"]
WAFER_KEY = ["WAFER_ID", "STAGE"]
TIME_KEY = ["WAFER_ID", "STAGE", "MACHINE_ID", "MACHINE_DATA", "CHAMBER", "TIMESTAMP"]
CHANNEL_KEY = ["WAFER_ID", "STAGE", "MACHINE_ID", "MACHINE_DATA", "CHAMBER"]


def find_paths(root: Path):
    cmp_root = root / "2016 PHM DATA CHALLENGE CMP DATA SET"
    val_root = root / "2016 PHM DATA CHALLENGE CMP VALIDATION DATA SET"
    return {
        "training": (cmp_root / "CMP-data" / "training", cmp_root / "CMP-training-removalrate.csv"),
        "test": (cmp_root / "CMP-data" / "test", cmp_root / "CMP-test-removalrate.csv"),
        "validation": (val_root / "validation", val_root / "CMP-validation-removalrate.csv"),
    }


def main():
    parser = argparse.ArgumentParser(description="Audit PHM sensor duplicates, timestamps, and wafer-stage granularity")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("outputs/audit"))
    args = parser.parse_args()
    root = args.source.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Missing source directory: {root}")
    args.out.mkdir(parents=True, exist_ok=True)

    all_summary = {}
    all_wafer_rows = []
    all_duplicate_examples = []
    all_time_conflict_examples = []

    for split, (sensor_dir, label_path) in find_paths(root).items():
        files = sorted(sensor_dir.glob("*.csv"))
        if not files or not label_path.exists():
            raise FileNotFoundError(f"Missing {split} source files: {sensor_dir} / {label_path}")
        labels = pd.read_csv(label_path, dtype={"WAFER_ID": "string", "STAGE": "string"})
        if list(labels.columns) != LABEL_COLS:
            raise ValueError(f"Unexpected columns in {label_path}")
        if labels[WAFER_KEY].duplicated().any():
            raise ValueError(f"Duplicated wafer-stage target in {label_path}")
        if labels[LABEL_COLS].isna().any().any():
            raise ValueError(f"Missing label values in {label_path}")

        chunks = []
        empty_files = 0
        reversals_within_file = 0
        for path in files:
            frame = pd.read_csv(path, dtype={"WAFER_ID": "string", "STAGE": "string"})
            if list(frame.columns) != SENSOR_COLS:
                raise ValueError(f"Unexpected columns in {path}: {list(frame.columns)}")
            if frame.empty:
                empty_files += 1
                continue
            if frame[SENSOR_COLS].isna().any().any():
                raise ValueError(f"Missing sensor data in {path.name}; investigate before continuing")
            # Only count downward timestamps within the same file and channel.
            # A global reversal count based on file-name order would be misleading.
            reversals_within_file += int(
                frame.groupby(CHANNEL_KEY, sort=False, dropna=False)["TIMESTAMP"]
                .diff().lt(0).sum()
            )
            frame["source_file"] = path.name
            frame["source_row"] = pd.RangeIndex(2, len(frame) + 2)  # CSV header is line 1.
            chunks.append(frame)

        if not chunks:
            raise ValueError(f"No process rows found in {split}")
        df = pd.concat(chunks, ignore_index=True)
        df["split"] = split
        rows_before = len(df)

        # A duplicate is identical across ALL 25 original sensor columns.
        # Do not deduplicate by timestamp alone.
        exact_duplicate_mask = df.duplicated(subset=SENSOR_COLS, keep="first")
        exact_duplicates = int(exact_duplicate_mask.sum())
        exact_dup_examples = df.loc[exact_duplicate_mask, ["split", "source_file", "source_row"] + TIME_KEY].head(20)
        all_duplicate_examples.append(exact_dup_examples)

        # Same physical key and time, but differing sensor values; keep every row.
        distinct_values = df.drop_duplicates(subset=SENSOR_COLS, keep="first")
        time_key_counts = distinct_values.groupby(TIME_KEY, dropna=False).size()
        conflicting_time_keys = time_key_counts[time_key_counts > 1]
        count_conflicting_keys = int(len(conflicting_time_keys))
        conflict_rows = int(conflicting_time_keys.sum())
        if count_conflicting_keys:
            example_keys = conflicting_time_keys.head(20).reset_index()[TIME_KEY]
            example_keys.insert(0, "split", split)
            example_keys["different_rows_at_same_time"] = conflicting_time_keys.head(20).to_numpy()
            all_time_conflict_examples.append(example_keys)

        # Summarize at one wafer-stage observation, do not multiply the MRR label.
        agg = df.groupby(WAFER_KEY, sort=False, dropna=False).agg(
            n_sensor_rows=("TIMESTAMP", "size"),
            n_distinct_timestamps=("TIMESTAMP", "nunique"),
            first_timestamp=("TIMESTAMP", "min"),
            last_timestamp=("TIMESTAMP", "max"),
            n_source_files=("source_file", "nunique"),
            n_machine_ids=("MACHINE_ID", "nunique"),
            n_machine_data_values=("MACHINE_DATA", "nunique"),
            n_chambers=("CHAMBER", "nunique"),
        ).reset_index()
        # Annotate exact duplicate counts without removing observations.
        duplicate_counts = df.loc[exact_duplicate_mask].groupby(WAFER_KEY).size().rename("n_exact_duplicate_rows")
        agg = agg.merge(duplicate_counts.reset_index(), on=WAFER_KEY, how="left", validate="one_to_one")
        agg["n_exact_duplicate_rows"] = agg["n_exact_duplicate_rows"].fillna(0).astype(int)
        agg = agg.merge(labels, on=WAFER_KEY, how="outer", indicator=True, validate="one_to_one")
        unmatched = agg["_merge"].value_counts().to_dict()
        if unmatched.get("left_only", 0) or unmatched.get("right_only", 0):
            raise ValueError(f"Unmatched wafer-stage keys in {split}: {unmatched}")
        agg = agg.drop(columns="_merge")
        agg.insert(0, "split", split)
        agg["mrr_review_gt_1000"] = agg["AVG_REMOVAL_RATE"] > 1000
        all_wafer_rows.append(agg)

        all_summary[split] = {
            "process_files": len(files),
            "header_only_files": empty_files,
            "raw_sensor_rows": rows_before,
            "quality_label_rows": len(labels),
            "wafer_stage_rows": len(agg),
            "exact_duplicate_rows_all_25_columns": exact_duplicates,
            "same_time_key_distinct_values_groups": count_conflicting_keys,
            "rows_in_same_time_key_distinct_values_groups": conflict_rows,
            "timestamp_reversals_within_files_and_channels": reversals_within_file,
            "missing_wafer_stage_keys": 0,
            "mrr_review_gt_1000": int(agg["mrr_review_gt_1000"].sum()),
            "rows_removed": 0,
        }
        print(f"{split:10}: {rows_before:,} sensor rows | {len(agg):,} wafer-stage rows | "
              f"{exact_duplicates:,} exact repeats | {count_conflicting_keys:,} same-time conflicts | "
              f"{reversals_within_file:,} within-file time reversals")

    inventories = pd.concat(all_wafer_rows, ignore_index=True)
    inventories.to_csv(args.out / "02_wafer_stage_inventory.csv", index=False, encoding="utf-8-sig")
    cols_dup = ["split", "source_file", "source_row"] + TIME_KEY
    pd.concat(all_duplicate_examples, ignore_index=True).reindex(columns=cols_dup).to_csv(
        args.out / "02_exact_duplicate_examples.csv", index=False, encoding="utf-8-sig"
    )
    cols_time = ["split"] + TIME_KEY + ["different_rows_at_same_time"]
    if all_time_conflict_examples:
        time_examples = pd.concat(all_time_conflict_examples, ignore_index=True)
    else:
        time_examples = pd.DataFrame(columns=cols_time)
    time_examples.to_csv(args.out / "02_same_timestamp_different_values_examples.csv", index=False, encoding="utf-8-sig")
    (args.out / "02_diagnostics_summary.json").write_text(
        json.dumps(all_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nSaved 4 diagnostics reports to: {args.out.resolve()}")
    print("No raw rows removed or modified. Never drop records based on timestamp alone.")


if __name__ == "__main__":
    main()
