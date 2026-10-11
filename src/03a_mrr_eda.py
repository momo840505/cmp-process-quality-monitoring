"""Step 03A: training-only CMP MRR exploratory analysis.

Read the quality table built in Step 02B, validate key assumptions, and
produce auditable tables and charts. No rows are changed or excluded from
statistics. The >1000 threshold is an existing inspection flag, NOT an SPC limit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STAGES = ("A", "B")
SPLITS = {"training": 1981, "test": 424, "validation": 424}
REQUIRED = {
    "split", "WAFER_ID", "STAGE", "AVG_REMOVAL_RATE", "mrr_review_gt_1000",
    "first_timestamp", "clean_sensor_rows", "n_source_files",
    "has_same_time_conflict", "recorded_time_span_seconds",
}


def summary_stats(frame: pd.DataFrame) -> dict:
    y = frame["AVG_REMOVAL_RATE"]
    return {
        "n": int(len(y)),
        "mean": float(y.mean()),
        "median": float(y.median()),
        "std_sample": float(y.std(ddof=1)),
        "min": float(y.min()),
        "p05": float(y.quantile(0.05)),
        "p25": float(y.quantile(0.25)),
        "p75": float(y.quantile(0.75)),
        "p95": float(y.quantile(0.95)),
        "max": float(y.max()),
        "review_gt_1000_count": int((y > 1000).sum()),
    }


def render_figures(training: pd.DataFrame, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    bins = np.linspace(40, 180, 48)
    fig, ax = plt.subplots(figsize=(10, 5.7))
    for stage in STAGES:
        s = training[training["STAGE"] == stage]["AVG_REMOVAL_RATE"]
        ax.hist(s, bins=bins, alpha=0.60, label=f"Stage {stage} (n={len(s):,})")
    ax.set(title="Training MRR distribution (display window: 40–180)",
           xlabel="Average removal rate (dataset scale)", ylabel="Wafer-stage count")
    ax.legend()
    ax.grid(alpha=0.20)
    fig.text(0.5, 0.01,
             "All values are retained in calculations. Four >1000 Stage A records are outside the displayed window.\n"
             "This display range is for visualization only; it is not a process specification or SPC limit.",
             ha="center", va="bottom", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(fig_dir / "03a_training_mrr_distribution_zoom.png", dpi=155)
    plt.close(fig)

    # A log scale displays both ordinary MRR and the exceptional positive values.
    a = training[training["STAGE"] == "A"].sort_values(
        ["first_timestamp", "WAFER_ID"], kind="mergesort"
    ).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 5.6))
    x = np.arange(1, len(a) + 1)
    marked = a["mrr_review_gt_1000"].astype(bool).to_numpy()
    ax.scatter(x[~marked], a.loc[~marked, "AVG_REMOVAL_RATE"],
               s=10, alpha=0.55, label="Other Stage A wafer-stages")
    ax.scatter(x[marked], a.loc[marked, "AVG_REMOVAL_RATE"],
               s=45, marker="x", color="tab:red", label="Review flag: MRR > 1000")
    ax.set_yscale("log")
    ax.set(title="Training Stage A MRR in recorded order (log Y axis)",
           xlabel="Wafer-stage order by first sensor timestamp (not calendar date)",
           ylabel="Average removal rate (log scale)")
    ax.legend()
    ax.grid(alpha=0.20)
    fig.tight_layout()
    fig.savefig(fig_dir / "03a_training_stageA_mrr_log.png", dpi=155)
    plt.close(fig)


def run(quality_path: Path, audit_path: Path, output: Path) -> None:
    if not quality_path.is_file():
        raise FileNotFoundError(f"Quality table not found: {quality_path}")
    if not audit_path.is_file():
        raise FileNotFoundError(f"Step 02B audit not found: {audit_path}")
    d = pd.read_csv(quality_path)
    missing = REQUIRED - set(d.columns)
    if missing:
        raise ValueError(f"Missing required quality columns: {sorted(missing)}")
    if d.isna().any().any():
        raise ValueError("Quality table has missing values; inspect Step 02B before proceeding")
    if d.duplicated(["split", "WAFER_ID", "STAGE"]).any():
        raise ValueError("Repeated split + wafer + stage keys detected")
    observed = d.groupby("split").size().to_dict()
    if observed != SPLITS:
        raise ValueError(f"Split counts differ from accepted inventory: {observed}")
    if set(d["STAGE"]) != set(STAGES):
        raise ValueError(f"Unexpected stages: {set(d['STAGE'])}")
    if not pd.api.types.is_numeric_dtype(d["AVG_REMOVAL_RATE"]):
        raise ValueError("MRR must be numeric")
    if not np.isfinite(d["AVG_REMOVAL_RATE"]).all() or (d["AVG_REMOVAL_RATE"] <= 0).any():
        raise ValueError("MRR contains non-finite or non-positive values")
    if (d["mrr_review_gt_1000"].astype(bool) != (d["AVG_REMOVAL_RATE"] > 1000)).any():
        raise ValueError("Review flag differs from data")
    with audit_path.open(encoding="utf-8") as f:
        audit = json.load(f)
    for split, n in SPLITS.items():
        if audit[split]["quality_rows"] != n:
            raise ValueError(f"Quality count mismatch vs 02B audit for {split}")

    training = d.loc[d["split"] == "training"].copy()
    output.mkdir(parents=True, exist_ok=True)
    fig_dir = output / "figures"
    rows = []
    stats = {}
    for stage in STAGES:
        group = training.loc[training["STAGE"] == stage]
        stats[stage] = summary_stats(group)
        rows.append({"stage": stage, "analysis_scope": "all training rows", **stats[stage]})
        # Sensitivity check only. Not a new data-cleaning rule.
        subset = group.loc[~group["mrr_review_gt_1000"]]
        rows.append({"stage": stage, "analysis_scope": "sensitivity: exclude flagged >1000", **summary_stats(subset)})
    pd.DataFrame(rows).to_csv(output / "03a_training_mrr_by_stage.csv", index=False)

    flagged = training.loc[training["mrr_review_gt_1000"], [
        "WAFER_ID", "STAGE", "AVG_REMOVAL_RATE", "clean_sensor_rows",
        "n_source_files", "recorded_time_span_seconds", "has_same_time_conflict",
    ]].sort_values("AVG_REMOVAL_RATE", ascending=False)
    flagged.to_csv(output / "03a_extreme_mrr_review.csv", index=False)

    render_figures(training, fig_dir)
    report = {
        "analysis_scope": "TRAINING ONLY (test/validation counts validated; outcomes not used in EDA)",
        "training_rows": int(len(training)),
        "train_stage_stats": stats,
        "flagged_gt_1000_wafer_stage_count": int(len(flagged)),
        "method_notes": [
            "MRR > 1000 is a review flag from Step 02B, not a process limit or an anomaly ground truth.",
            "No engineering USL/LSL or stable-phase reference is available; SPC limits and Cpk cannot be claimed here.",
            "Sensitivity exclusion is a comparison only. No source or processed data is modified.",
            "Test and validation outcomes must not determine model thresholds or feature selections.",
            "First sensor timestamp represents available order, not an independently verified production calendar.",
        ],
    }
    (output / "03a_eda_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("PASS: Step 02B quality table reconciled with 02B audit; 2,829 unique wafer-stage quality rows")
    for stage in STAGES:
        s = stats[stage]
        print(f"Training Stage {stage}: n={s['n']:,} | median={s['median']:.3f} | "
              f"mean={s['mean']:.3f} | max={s['max']:.3f} | flagged >1000={s['review_gt_1000_count']}")
    print(f"Saved 3 reports and 2 figures under: {output.resolve()}")
    print("No input files were modified. No SPC control/specification limits were assumed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 03A: training-only CMP quality exploratory analysis")
    parser.add_argument("--quality", type=Path, default=Path("outputs/processed/02b_wafer_stage_quality.csv"))
    parser.add_argument("--audit", type=Path, default=Path("outputs/audit/02b_cleaning_summary.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/analysis"))
    args = parser.parse_args()
    run(args.quality, args.audit, args.output)


if __name__ == "__main__":
    main()
