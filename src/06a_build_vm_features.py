"""Build supervised Virtual Metrology feature table from TRAINING Stage A only.

English code comments; Chinese study notes are in docs/06a_learning_notes.md.
Usage: python src/06a_build_vm_features.py
This file never modifies original PHM data and never fits a model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SIGNALS = [
    "PRESSURIZED_CHAMBER_PRESSURE",
    "RETAINER_RING_PRESSURE",
    "MAIN_OUTER_AIR_BAG_PRESSURE",
    "SLURRY_FLOW_LINE_A",
    "SLURRY_FLOW_LINE_C",
    "WAFER_ROTATION",
    "STAGE_ROTATION",
    "HEAD_ROTATION",
]
CONSUMABLES = [
    "USAGE_OF_BACKING_FILM",
    "USAGE_OF_DRESSER",
    "USAGE_OF_POLISHING_TABLE",
    "USAGE_OF_MEMBRANE",
]
KEYS = ["WAFER_ID", "STAGE"]
SEED = 20261011


def group_name(codes: tuple[int, ...]) -> str:
    if codes == (1, 2, 3):
        return "Codes 1-3"
    if codes == (4, 5, 6):
        return "Codes 4-6"
    if set(codes).issubset({1, 2, 3}):
        return "Codes 1-3 (incomplete)"
    if set(codes).issubset({4, 5, 6}):
        return "Codes 4-6 (incomplete)"
    return "Other/unexpected"


def split_wafers(frame: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Wafer-exclusive 70/15/15 assignment, stratified on SENSOR code family only.

    Never stratify on MRR or MRR review flags. Use a reproducible sorted key order.
    """
    pairs = frame[["WAFER_ID", "code_family"]].drop_duplicates()
    assert pairs.WAFER_ID.is_unique, "A wafer is associated with multiple code families"
    rng = np.random.default_rng(seed)
    results = []
    for family, part in sorted(pairs.groupby("code_family"), key=lambda p: p[0]):
        ids = np.sort(part.WAFER_ID.to_numpy(dtype=np.int64)).copy()
        rng.shuffle(ids)
        n = len(ids)
        n_train = round(n * 0.70)
        n_valid = round(n * 0.15)
        names = (["train"] * n_train + ["valid"] * n_valid
                 + ["test"] * (n - n_train - n_valid))
        results.extend(zip(ids.tolist(), names))
    out = pd.DataFrame(results, columns=["WAFER_ID", "vm_partition"])
    assert out.WAFER_ID.is_unique
    return out


def make_features(sensor: pd.DataFrame, quality: pd.DataFrame,
                  seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Transform sensors without reading outcome values for feature creation.

    Quality is joined *after* all sensor-only transformations have completed.
    """
    if sensor.empty:
        raise ValueError("Sensor file is empty")
    if not (sensor["split"].eq("training") & sensor.STAGE.eq("A")).all():
        raise ValueError("Expected TRAINING Stage A sensor events only")
    if not (quality["split"].eq("training") & quality.STAGE.eq("A")).all():
        raise ValueError("Expected TRAINING Stage A quality rows only")
    if not quality.WAFER_ID.is_unique:
        raise ValueError("Duplicate Stage A wafer quality records")

    needed = [*KEYS, "MACHINE_DATA", "same_time_multiple_sensor_values", *SIGNALS, *CONSUMABLES]
    missing = set(needed) - set(sensor.columns)
    if missing:
        raise ValueError(f"Missing sensor columns: {sorted(missing)}")
    sensor = sensor.loc[:, needed].copy()
    sensor.MACHINE_DATA = pd.to_numeric(sensor.MACHINE_DATA, errors="raise").astype(int)
    if not sensor.MACHINE_DATA.isin([1, 2, 3, 4, 5, 6]).all():
        raise ValueError("Unexpected MACHINE_DATA outside Codes 1..6")

    # Wafer group assignment uses sensor codes exclusively, not observed MRR.
    code_sig = (sensor.groupby("WAFER_ID").MACHINE_DATA
                .agg(lambda x: tuple(sorted(int(v) for v in x.unique()))))
    meta = pd.DataFrame({"WAFER_ID": code_sig.index,
                         "code_signature": code_sig.map(lambda t: ",".join(map(str, t))).values,
                         "code_family": code_sig.map(group_name).values})
    meta["has_complete_code_signature"] = meta.code_signature.isin(["1,2,3", "4,5,6"])
    meta["has_recorded_time_conflict"] = sensor.groupby("WAFER_ID")[
        "same_time_multiple_sensor_values"].any().reindex(meta.WAFER_ID).to_numpy()

    # Sensor-derived summaries: do NOT use MRR, wafer ID, timestamps or QA flags as predictors.
    chunks = []
    descriptions: list[tuple[str, str, str]] = []
    for code in range(1, 7):
        part = sensor.loc[sensor.MACHINE_DATA == code]
        if part.empty:
            raise AssertionError(f"Code {code} not found in sensor rows")
        g = part.groupby("WAFER_ID", sort=True)
        feat = pd.DataFrame(index=g.size().index)
        col = f"code{code}__event_count"
        feat[col] = g.size().astype(int)
        descriptions.append((col, f"Code {code} 的感測事件數", "Sensor observation count; not elapsed process time"))
        for name in SIGNALS:
            col_frac = f"code{code}__{name}__nonzero_fraction"
            col_med = f"code{code}__{name}__median_nonzero"
            # Zero/NA are not interchangeable. A missing non-zero median stays NaN.
            feat[col_frac] = g[name].agg(lambda x: float((x != 0).mean()))
            feat[col_med] = g[name].agg(lambda x: float(x.loc[x != 0].median()) if (x != 0).any() else np.nan)
            descriptions.append((col_frac, f"Code {code}，{name} 非零比例", "Fraction of recorded nonzero values; zero meaning is unknown"))
            descriptions.append((col_med, f"Code {code}，{name} 非零中位數", "Median of observed nonzero values; NA if no nonzero readings"))
        for name in CONSUMABLES:
            col_use = f"code{code}__{name}__median"
            feat[col_use] = g[name].median()
            descriptions.append((col_use, f"Code {code}，{name} 中位數", "Consumable usage summary from process sensors"))
        feat.index.name = "WAFER_ID"
        chunks.append(feat)

    combined = pd.concat(chunks, axis=1).reset_index()
    combined = meta.merge(combined, on="WAFER_ID", how="left", validate="one_to_one")
    for c in range(1, 7):
        col = f"code{c}__present"
        combined[col] = combined[f"code{c}__event_count"].notna().astype(int)
        descriptions.append((col, f"是否有 Code {c} 的感測紀錄", "Presence indicator derived only from available sensor events"))
    # A dedicated, stable categorical predictor. Incomplete signature is audit only.
    combined["mode_from_sensor"] = combined.code_family.map({
        "Codes 1-3": "group_123", "Codes 4-6": "group_456",
        "Codes 1-3 (incomplete)": "group_123",
        "Codes 4-6 (incomplete)": "group_456"}).fillna("other")
    descriptions.append(("mode_from_sensor", "從感測代碼推得的運作群組", "Anonymous code family; not proven physical recipe or tool"))
    combined = combined.merge(split_wafers(combined, seed=seed), on="WAFER_ID", validate="one_to_one")

    # Join the target LAST to make the no-target-feature principle auditable.
    q_cols = ["WAFER_ID", "AVG_REMOVAL_RATE", "mrr_review_gt_1000"]
    missing_q = set(q_cols) - set(quality.columns)
    if missing_q:
        raise ValueError(f"Missing quality columns: {sorted(missing_q)}")
    result = combined.merge(quality.loc[:, q_cols], on="WAFER_ID", how="left",
                            validate="one_to_one", indicator=True)
    if not result["_merge"].eq("both").all() or len(result) != len(quality):
        raise AssertionError("Sensor–MRR matching failed for Stage A wafers")
    result.drop(columns="_merge", inplace=True)
    if result.AVG_REMOVAL_RATE.isna().any():
        raise AssertionError("Missing MRR after exact wafer-key join")
    result.sort_values(["vm_partition", "code_family", "WAFER_ID"], inplace=True)
    result.reset_index(drop=True, inplace=True)

    # An allowlist is safer than accidentally feeding quality flags into a model.
    feature_names = [x[0] for x in descriptions]
    assert len(feature_names) == len(set(feature_names))
    assert not any("MRR" in x.upper() or "TARGET" in x.upper() or "TIMESTAMP" in x.upper()
                   for x in feature_names)
    assert result["WAFER_ID"].is_unique
    catalog = pd.DataFrame(descriptions, columns=["feature_name", "chinese_description", "english_definition"])
    assert set(catalog.feature_name).issubset(result.columns)
    partition_info = []
    for (family, partition), d in result.groupby(["code_family", "vm_partition"]):
        partition_info.append({"code_family": family, "vm_partition": partition,
                               "n_wafers": len(d), "n_mrr_review_gt_1000": int(d.mrr_review_gt_1000.sum()),
                               "n_incomplete_signatures": int((~d.has_complete_code_signature).sum()),
                               "n_with_timestamp_conflict": int(d.has_recorded_time_conflict.sum())})
    stats = {"scope": "Original TRAINING Stage A only; no original TEST/VALIDATION MRR accessed",
             "input_events": len(sensor), "wafer_stage_rows": len(result),
             "n_feature_columns_allowlisted": len(feature_names),
             "feature_name_file": "06a_feature_catalog.csv",
             "partition_counts": {k:int(v) for k,v in result.vm_partition.value_counts().items()},
             "code_group_counts": {k:int(v) for k,v in result.code_family.value_counts().items()},
             "n_mrr_review_flags_retained": int(result.mrr_review_gt_1000.sum()),
             "n_incomplete_code_signatures_retained": int((~result.has_complete_code_signature).sum()),
             "why_no_model_yet": "Prepare no-leakage feature table first. Fit preprocessing and baselines on internal TRAIN only in Step 06B.",
             "constraints": [
                 "Output includes target MRR and quality-review flags, which MUST NOT be treated as X features",
                 "WAFER_ID, recorded timestamps and code signature are metadata; never feed IDs as predictive numeric features",
                 "Sensor code groups are anonymous and not validated recipes or stable SPC modes",
                 "Per-code absence remains NaN; only fit missing-value handling using training partition",
                 "Original competition Test/Validation remain untouched for future external evaluation",
                 "MRR>1000 review records were RETAINED; no outcome-based filtering or partition stratification",
                 "Split is reproducible but original sensor timestamps do not prove independent temporal production order",
                 "Wafer-stage feature summaries represent complete process observations; not live early-stage sensor FDC",
             ]}
    return result, catalog, {"audit":stats, "partitions":partition_info}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensor", type=Path, default=ROOT / "outputs/processed/02b_sensor_events_training.csv.gz")
    parser.add_argument("--quality", type=Path, default=ROOT / "outputs/processed/02b_wafer_stage_quality.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/analysis")
    args = parser.parse_args()

    sensor = pd.read_csv(args.sensor)
    quality = pd.read_csv(args.quality)
    train_stage = sensor.loc[sensor.split.eq("training") & sensor.STAGE.eq("A")].copy()
    quality_stage = quality.loc[quality.split.eq("training") & quality.STAGE.eq("A")].copy()
    result, catalog, report = make_features(train_stage, quality_stage)
    args.output.mkdir(exist_ok=True, parents=True)
    result.to_csv(args.output / "06a_stageA_vm_features.csv", index=False)
    catalog.to_csv(args.output / "06a_feature_catalog.csv", index=False)
    pd.DataFrame(report["partitions"]).to_csv(args.output / "06a_partition_summary.csv", index=False)
    (args.output / "06a_feature_audit.json").write_text(
        json.dumps(report["audit"], ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"PASS: {len(train_stage):,} Training Stage A sensor events -> {len(result):,} unique wafer-stage rows")
    print(f"Sensor-only model features: {len(catalog)} columns; labels joined only after feature construction")
    print("Mode counts:", report["audit"]["code_group_counts"])
    print("Internal wafer-exclusive partitions:", report["audit"]["partition_counts"])
    print(f"Review flags retained: {report['audit']['n_mrr_review_flags_retained']}; "
          f"incomplete code signatures: {report['audit']['n_incomplete_code_signatures_retained']}")
    print("No model fitted, no target-based feature selection, original Test/Validation untouched.")
    print("Saved 4 files to:", args.output.resolve())


if __name__ == "__main__":
    main()
