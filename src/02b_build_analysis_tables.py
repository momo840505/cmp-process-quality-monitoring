"""Step 02B — PHM CMP: conservative derived-event view and wafer-stage quality table.

Reads but never changes the original third-party source CSVs. Keeps separate
training, test, validation partitions. Exact duplicate events (all 25 original
columns identical) are removed ONLY from a derived, reproducible event view.
All same-time, different-value events are kept and flagged for review.

Run (project root):
    python src/02b_build_analysis_tables.py --source "C:/.../data/2016 PHM Data Challenge"
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
EVENT_SORT_KEY = ["WAFER_ID", "STAGE", "MACHINE_ID", "MACHINE_DATA", "CHAMBER", "TIMESTAMP", "source_file", "source_row"]


def source_paths(root: Path):
    primary = root / "2016 PHM DATA CHALLENGE CMP DATA SET"
    valid = root / "2016 PHM DATA CHALLENGE CMP VALIDATION DATA SET"
    return {
        "training": (primary / "CMP-data" / "training", primary / "CMP-training-removalrate.csv"),
        "test": (primary / "CMP-data" / "test", primary / "CMP-test-removalrate.csv"),
        "validation": (valid / "validation", valid / "CMP-validation-removalrate.csv"),
    }


def load_split(split: str, sensor_dir: Path, label_file: Path):
    file_paths = sorted(sensor_dir.glob("*.csv"))
    if len(file_paths) != 185 or not label_file.is_file():
        raise FileNotFoundError(f"Expected 185 {split} sensor CSV files and one label CSV: {sensor_dir} / {label_file}")
    labels = pd.read_csv(label_file, dtype={"WAFER_ID": "string", "STAGE": "string"})
    if list(labels.columns) != LABEL_COLS or labels[LABEL_COLS].isna().any().any():
        raise ValueError(f"Bad quality-label schema or missing cells: {label_file}")
    if labels[WAFER_KEY].duplicated().any():
        raise ValueError(f"A wafer-stage has multiple MRR values in {label_file}")

    frames = []
    empty_count = 0
    for file in file_paths:
        frame = pd.read_csv(file, dtype={"WAFER_ID": "string", "STAGE": "string"})
        if list(frame.columns) != SENSOR_COLS:
            raise ValueError(f"Sensor schema differs in {file}")
        if frame.empty:
            empty_count += 1
            continue
        if frame[SENSOR_COLS].isna().any().any():
            raise ValueError(f"Missing sensor cells in {file}; stop instead of silently imputing")
        frame["source_file"] = file.name
        frame["source_row"] = pd.RangeIndex(2, len(frame) + 2)
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    return raw, labels, len(file_paths), empty_count


def audit_and_build(split: str, raw: pd.DataFrame, labels: pd.DataFrame, output: Path, audit: Path):
    raw_rows = len(raw)
    dup_mask = raw.duplicated(subset=SENSOR_COLS, keep="first")
    dropped = raw.loc[dup_mask, ["source_file", "source_row"] + TIME_KEY].copy()
    dropped.insert(0, "split", split)
    # Removing exact duplicates is an analysis VIEW choice, NOT a change to source data.
    clean = raw.loc[~dup_mask].copy()

    # A collision only exists when all of TIME_KEY is the same but one or more of the
    # other 19 sensor values differ. Preserve every such row, flagging the whole group.
    per_time = clean.groupby(TIME_KEY, dropna=False, sort=False).size().rename("different_rows_at_same_time")
    conflicts = per_time.loc[per_time > 1].reset_index()
    conflicts.insert(0, "split", split)
    marked = conflicts[TIME_KEY].copy()
    marked["same_time_multiple_sensor_values"] = True
    clean = clean.merge(marked, on=TIME_KEY, how="left", validate="many_to_one")
    clean["same_time_multiple_sensor_values"] = clean["same_time_multiple_sensor_values"].eq(True)

    # Source file/row are retained so any downstream anomaly can be traced back.
    clean.sort_values(EVENT_SORT_KEY, kind="stable", inplace=True)
    clean.insert(0, "split", split)
    clean.to_csv(output / f"02b_sensor_events_{split}.csv.gz", index=False, compression={"method": "gzip", "compresslevel": 1})

    quality = clean.groupby(WAFER_KEY, dropna=False, sort=False).agg(
        clean_sensor_rows=("TIMESTAMP", "size"),
        first_timestamp=("TIMESTAMP", "min"),
        last_timestamp=("TIMESTAMP", "max"),
        distinct_timestamps=("TIMESTAMP", "nunique"),
        n_machine_data_values=("MACHINE_DATA", "nunique"),
        n_chambers=("CHAMBER", "nunique"),
        n_source_files=("source_file", "nunique"),
    ).reset_index()
    raw_count = raw.groupby(WAFER_KEY).size().rename("raw_sensor_rows").reset_index()
    conflict_count = conflicts.groupby(WAFER_KEY).size().rename("same_time_conflict_groups").reset_index()
    quality = quality.merge(raw_count, on=WAFER_KEY, how="left", validate="one_to_one")
    quality = quality.merge(conflict_count, on=WAFER_KEY, how="left", validate="one_to_one")
    quality["same_time_conflict_groups"] = quality["same_time_conflict_groups"].fillna(0).astype(int)
    quality["removed_exact_duplicates"] = quality["raw_sensor_rows"] - quality["clean_sensor_rows"]
    quality["recorded_time_span_seconds"] = quality["last_timestamp"] - quality["first_timestamp"]
    quality["has_same_time_conflict"] = quality["same_time_conflict_groups"] > 0
    quality = quality.merge(labels, on=WAFER_KEY, how="outer", indicator=True, validate="one_to_one")
    unmatched = quality["_merge"].value_counts().to_dict()
    if unmatched.get("left_only", 0) or unmatched.get("right_only", 0):
        raise ValueError(f"Unmatched wafer-stage labels in {split}: {unmatched}")
    quality.drop(columns="_merge", inplace=True)
    quality["mrr_review_gt_1000"] = quality["AVG_REMOVAL_RATE"] > 1000
    quality.insert(0, "split", split)
    # Reproducible stable ordering for inspection, not a claim of factory chronology.
    quality.sort_values(WAFER_KEY, kind="stable", inplace=True)

    return quality, dropped, conflicts, {
        "raw_sensor_rows": int(raw_rows),
        "exact_duplicate_rows_removed_from_derived_view": int(dup_mask.sum()),
        "derived_sensor_rows": int(len(clean)),
        "same_time_distinct_values_groups_preserved": int(len(conflicts)),
        "same_time_distinct_values_rows_preserved": int(clean["same_time_multiple_sensor_values"].sum()),
        "quality_rows": int(len(quality)),
        "quality_rows_with_same_time_conflicts": int(quality["has_same_time_conflict"].sum()),
        "quality_rows_mrr_review_gt_1000": int(quality["mrr_review_gt_1000"].sum()),
        "quality_matching": "all wafer-stage keys match",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("outputs/processed"))
    ap.add_argument("--audit", type=Path, default=Path("outputs/audit"))
    args = ap.parse_args()
    root = args.source.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Source does not exist: {root}")
    args.out.mkdir(parents=True, exist_ok=True)
    args.audit.mkdir(parents=True, exist_ok=True)
    all_quality = []
    all_dups = []
    all_conflicts = []
    summary = {}
    for split, (sensor_dir, label_file) in source_paths(root).items():
        raw, labels, count_files, empty_files = load_split(split, sensor_dir, label_file)
        quality, dups, conflicts, s = audit_and_build(split, raw, labels, args.out, args.audit)
        s["sensor_files"] = count_files
        s["header_only_files"] = empty_files
        summary[split] = s
        all_quality.append(quality)
        all_dups.append(dups)
        all_conflicts.append(conflicts)
        print(f"{split:10} | raw {s['raw_sensor_rows']:,} | exact duplicates {s['exact_duplicate_rows_removed_from_derived_view']:,} | derived {s['derived_sensor_rows']:,} | time-value conflicts {s['same_time_distinct_values_groups_preserved']:,} | quality {s['quality_rows']:,}")

    combined_quality = pd.concat(all_quality, ignore_index=True)
    combined_quality.to_csv(args.out / "02b_wafer_stage_quality.csv", index=False, encoding="utf-8-sig")
    pd.concat(all_dups, ignore_index=True).to_csv(args.audit / "02b_exact_duplicate_removal_log.csv", index=False, encoding="utf-8-sig")
    pd.concat(all_conflicts, ignore_index=True).to_csv(args.audit / "02b_same_time_conflicts_all.csv", index=False, encoding="utf-8-sig")
    (args.audit / "02b_cleaning_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    EXPECTED = {
        "training": (672744, 1819, 401, 1981),
        "test": (156262, 98, 89, 424),
        "validation": (144148, 434, 64, 424),
    }
    for split, expected in EXPECTED.items():
        s = summary[split]
        actual = (s["raw_sensor_rows"], s["exact_duplicate_rows_removed_from_derived_view"], s["same_time_distinct_values_groups_preserved"], s["quality_rows"])
        if actual != expected:
            raise AssertionError(f"Output differs from earlier Step 02A audit for {split}. Expected {expected}, got {actual}.")
    if len(combined_quality) != 2829 or combined_quality[["split"] + WAFER_KEY].duplicated().any():
        raise AssertionError("Wrong number or duplicate keys in output quality table")
    print("PASS: original inventory reconciles with Step 02A; 2,829 quality rows with unique split/wafer/stage.")
    print(f"Derived files saved to: {args.out.resolve()}")
    print(f"Cleaning audit saved to: {args.audit.resolve()}")
    print("Original source files unchanged. No numerical/time cutoffs or MRR thresholds were used to drop events.")


if __name__ == "__main__":
    main()
