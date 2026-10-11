"""Step 01: Read-only inventory of PHM 2016 CMP data.

This script does not remove, deduplicate, impute, or modify source rows.
Run: python src/01_data_inventory.py --source "C:\\...\\data\\2016 PHM Data Challenge"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

EXPECTED_SENSOR_COLUMNS = [
    "MACHINE_ID", "MACHINE_DATA", "TIMESTAMP", "WAFER_ID", "STAGE", "CHAMBER",
    "USAGE_OF_BACKING_FILM", "USAGE_OF_DRESSER", "USAGE_OF_POLISHING_TABLE",
    "USAGE_OF_DRESSER_TABLE", "PRESSURIZED_CHAMBER_PRESSURE",
    "MAIN_OUTER_AIR_BAG_PRESSURE", "CENTER_AIR_BAG_PRESSURE",
    "RETAINER_RING_PRESSURE", "RIPPLE_AIR_BAG_PRESSURE", "USAGE_OF_MEMBRANE",
    "USAGE_OF_PRESSURIZED_SHEET", "SLURRY_FLOW_LINE_A", "SLURRY_FLOW_LINE_B",
    "SLURRY_FLOW_LINE_C", "WAFER_ROTATION", "STAGE_ROTATION", "HEAD_ROTATION",
    "DRESSING_WATER_STATUS", "EDGE_AIR_BAG_PRESSURE",
]
EXPECTED_LABEL_COLUMNS = ["WAFER_ID", "STAGE", "AVG_REMOVAL_RATE"]


def split_paths(root: Path):
    main = root / "2016 PHM DATA CHALLENGE CMP DATA SET"
    validation = root / "2016 PHM DATA CHALLENGE CMP VALIDATION DATA SET"
    return {
        "training": (main / "CMP-data" / "training", main / "CMP-training-removalrate.csv"),
        "test": (main / "CMP-data" / "test", main / "CMP-test-removalrate.csv"),
        "validation": (validation / "validation", validation / "CMP-validation-removalrate.csv"),
    }


def keys(df: pd.DataFrame) -> set[tuple[str, str]]:
    return set(zip(df["WAFER_ID"].astype(str), df["STAGE"].astype(str)))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True, help="Path to data/2016 PHM Data Challenge")
    p.add_argument("--out", type=Path, default=Path("outputs/audit"))
    args = p.parse_args()

    root = args.source.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Cannot locate source data folder: {root}")
    args.out.mkdir(parents=True, exist_ok=True)
    records, combined_labels, summary = [], [], {}
    wafer_ids: dict[str, set[str]] = {}
    stage_keys: dict[str, set[tuple[str, str]]] = {}

    for split, (sensor_dir, label_path) in split_paths(root).items():
        if not sensor_dir.is_dir() or not label_path.is_file():
            raise FileNotFoundError(f"Missing {split} source: {sensor_dir} OR {label_path}")
        files = sorted(sensor_dir.glob("*.csv"))
        if not files:
            raise RuntimeError(f"No CSV files in {sensor_dir}")

        labels = pd.read_csv(label_path, dtype={"WAFER_ID": "string", "STAGE": "string"})
        if list(labels.columns) != EXPECTED_LABEL_COLUMNS:
            raise ValueError(f"Unexpected columns in {label_path}: {list(labels.columns)}")
        if labels.duplicated(["WAFER_ID", "STAGE"]).any():
            raise ValueError(f"Duplicate label keys within {split}")
        labels["split"] = split
        labels["mrr_review_gt_1000"] = labels["AVG_REMOVAL_RATE"] > 1000
        combined_labels.append(labels)
        label_keys = keys(labels)
        sensor_keys: set[tuple[str, str]] = set()
        n_sensor_rows = 0
        n_empty_files = 0
        n_missing_cells = 0
        for f in files:
            data = pd.read_csv(f, dtype={"WAFER_ID": "string", "STAGE": "string"})
            if list(data.columns) != EXPECTED_SENSOR_COLUMNS:
                raise ValueError(f"Unexpected sensor columns in {f.name}: {list(data.columns)}")
            rows = len(data)
            n_sensor_rows += rows
            n_empty_files += rows == 0
            missing = int(data.isna().sum().sum())
            n_missing_cells += missing
            sensor_keys.update(keys(data))
            records.append({"split": split, "file": f.name, "rows": rows, "columns": len(data.columns), "missing_cells": missing})

        wafer_ids[split] = set(labels["WAFER_ID"].astype(str))
        stage_keys[split] = label_keys
        unmatched_labels = label_keys - sensor_keys
        unmatched_sensors = sensor_keys - label_keys
        summary[split] = {
            "sensor_csv_files": len(files),
            "header_only_files": int(n_empty_files),
            "sensor_rows": int(n_sensor_rows),
            "sensor_missing_cells": int(n_missing_cells),
            "label_rows": int(len(labels)),
            "unique_wafer_ids": int(labels["WAFER_ID"].nunique()),
            "unique_wafer_stage_keys": int(len(label_keys)),
            "sensor_keys_without_labels": int(len(unmatched_sensors)),
            "label_keys_without_sensors": int(len(unmatched_labels)),
            "mrr_median": float(labels["AVG_REMOVAL_RATE"].median()),
            "mrr_max": float(labels["AVG_REMOVAL_RATE"].max()),
            "mrr_review_gt_1000_count": int(labels["mrr_review_gt_1000"].sum()),
        }
        print(f"{split:10} {len(files)} CSV | {n_sensor_rows:,} sensor rows | "
              f"{len(labels):,} labels | {n_empty_files} header-only files | "
              f"unmatched labels/sensors: {len(unmatched_labels)}/{len(unmatched_sensors)}")

    pairs = [("training", "test"), ("training", "validation"), ("test", "validation")]
    summary["cross_split_overlap"] = {
        f"{a}_vs_{b}": {
            "shared_wafer_ids": len(wafer_ids[a] & wafer_ids[b]),
            "shared_wafer_stage_keys": len(stage_keys[a] & stage_keys[b]),
        } for a, b in pairs
    }

    pd.DataFrame(records).to_csv(args.out / "01_sensor_files.csv", index=False, encoding="utf-8-sig")
    pd.concat(combined_labels, ignore_index=True).to_csv(
        args.out / "01_quality_labels.csv", index=False, encoding="utf-8-sig"
    )
    (args.out / "01_dataset_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nCross-split wafer ID overlap:")
    for p, d in summary["cross_split_overlap"].items():
        print(f"  {p}: {d['shared_wafer_ids']} shared IDs; {d['shared_wafer_stage_keys']} shared wafer-stage keys")
    print(f"\nSaved reports to: {args.out.resolve()}")
    print("No source data was changed; >1000 is an inspection flag, not a specification limit.")


if __name__ == "__main__":
    main()
