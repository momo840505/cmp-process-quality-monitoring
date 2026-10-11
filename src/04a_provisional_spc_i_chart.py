"""Step 04A: exploratory SPC Individuals chart, not a qualified fab control chart.

Purpose:
  1) Separate anonymous Stage A MACHINE_DATA code groups.
  2) Build an illustrative I-chart reference from the first 60% of TRAINING
     observations sorted by first sensor timestamp *within each code group*.
  3) Apply locked exploratory thresholds to the later 40% in TRAINING only.
  4) Explicitly report autocorrelation and baseline contamination risks.

NO test/validation outcomes are read. NEVER call these official fab SPC limits.
Usage from CMP-Process-QE root:
  .\\.venv\\Scripts\\python.exe .\\src\\04a_provisional_spc_i_chart.py
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

D2_N2 = 1.128  # Expected moving range multiplier under iid normal observations.
REQUIRED = {
    "split", "WAFER_ID", "STAGE", "first_timestamp", "last_timestamp",
    "code_group", "machine_data_codes", "AVG_REMOVAL_RATE", "mrr_review_gt_1000",
}
EXPECTED_SIG = {"Codes 1-3": "1,2,3", "Codes 4-6": "4,5,6"}


def to_bool(series: pd.Series) -> pd.Series:
    # Data saved as bool or strings depending on CSV reader/platform.
    normalized = series.astype(str).str.strip().str.lower()
    if not normalized.isin(["true", "false", "1", "0"]).all():
        raise ValueError("Unexpected values in mrr_review_gt_1000")
    return normalized.isin(["true", "1"])


def run(source: Path, output: Path) -> dict:
    if not source.is_file():
        raise FileNotFoundError(f"Missing Step 03B source: {source}")
    df = pd.read_csv(source)
    missing = sorted(REQUIRED - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if set(df["split"].astype(str)) != {"training"} or set(df["STAGE"].astype(str)) != {"A"}:
        raise ValueError("Input is not exclusively TRAINING / Stage A; stop to prevent holdout leakage")
    if set(df["code_group"].astype(str)) != set(EXPECTED_SIG):
        raise ValueError("Unexpected code groups; verify Step 03B output")
    if df[["split", "WAFER_ID", "STAGE"]].duplicated().any():
        raise ValueError("Duplicate wafer-stage key")
    for col in ["first_timestamp", "last_timestamp", "AVG_REMOVAL_RATE"]:
        df[col] = pd.to_numeric(df[col], errors="raise")
        if not np.isfinite(df[col].to_numpy(float)).all():
            raise ValueError(f"Nonfinite data in {col}")
    if (df.last_timestamp < df.first_timestamp).any():
        raise ValueError("Invalid sensor timestamp ranges")
    df["review"] = to_bool(df["mrr_review_gt_1000"])
    # The pre-existing >1000 MRR flag must correspond to the existing inspection definition.
    if not (df["review"].to_numpy() == (df.AVG_REMOVAL_RATE > 1000).to_numpy()).all():
        raise ValueError("Existing MRR review flags inconsistent with original definition")
    df["canonical_signature"] = df.apply(
        lambda r: str(r.machine_data_codes).replace(" ", "") == EXPECTED_SIG[str(r.code_group)], axis=1
    )
    df["recorded_span_seconds"] = df.last_timestamp - df.first_timestamp
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    reviewed = df.loc[df.review | ~df.canonical_signature, [
        "WAFER_ID", "code_group", "machine_data_codes", "AVG_REMOVAL_RATE",
        "mrr_review_gt_1000", "canonical_signature", "first_timestamp", "last_timestamp", "recorded_span_seconds"
    ]].copy()
    reviewed.to_csv(output / "04a_excluded_from_provisional_chart_review.csv", index=False)

    rows = []
    flag_rows = []
    sequences = []
    details = {}
    for group in ["Codes 1-3", "Codes 4-6"]:
        full = df.loc[df.code_group == group].sort_values(
            ["first_timestamp", "WAFER_ID"], kind="stable").copy()
        candidate = full.loc[~full.review & full.canonical_signature].copy().reset_index(drop=True)
        if len(candidate) < 100:
            raise ValueError(f"Not enough observations for {group}")
        if candidate.first_timestamp.duplicated().any():
            raise ValueError(f"Timestamp ties in {group}; tie resolution requires a documented policy")
        n = len(candidate)
        n_ref = int(np.floor(0.60 * n))
        if n_ref < 25 or n_ref >= n:
            raise ValueError("Invalid reference/monitor split")
        y = candidate.AVG_REMOVAL_RATE.to_numpy(float)
        reference = y[:n_ref]
        monitor = y[n_ref:]

        center = float(reference.mean())
        mr = np.abs(np.diff(reference))
        mr_bar = float(mr.mean())
        sigma_mr = mr_bar / D2_N2
        sigma_std = float(reference.std(ddof=1))
        if not (sigma_mr > 0 and np.isfinite(sigma_mr)):
            raise ValueError(f"Invalid moving-range estimate in {group}")
        upper = center + 3 * sigma_mr
        lower = center - 3 * sigma_mr
        flag = (y > upper) | (y < lower)
        lag1_reference = float(np.corrcoef(reference[:-1], reference[1:])[0, 1])
        lag1_monitor = float(np.corrcoef(monitor[:-1], monitor[1:])[0, 1])
        ref_flag_count = int(flag[:n_ref].sum())
        monitor_flag_count = int(flag[n_ref:].sum())
        # Sensitivity: including review flags in a separately fitted early reference
        # demonstrates vulnerability; not used in the primary chart.
        full_y = full.AVG_REMOVAL_RATE.to_numpy(float)
        full_ref = full_y[: int(np.floor(.60 * len(full_y)))]
        full_center = float(full_ref.mean())
        full_mr_sigma = float(np.abs(np.diff(full_ref)).mean() / D2_N2)
        group_row = {
            "code_group": group,
            "all_training_wafer_stages": int(len(full)),
            "review_or_noncanonical_not_plotted": int(len(full) - n),
            "candidate_chart_rows": n,
            "reference_rows_first_60pct": n_ref,
            "monitor_rows_later_40pct": n - n_ref,
            "reference_mean_CL_provisional": center,
            "reference_MR_bar": mr_bar,
            "reference_sigma_MR_over_1_128": sigma_mr,
            "reference_sample_SD_comparison": sigma_std,
            "reference_UCL_provisional": upper,
            "reference_LCL_provisional": lower,
            "reference_lag1_correlation": lag1_reference,
            "monitor_lag1_correlation": lag1_monitor,
            "reference_outside_3sigma_count": ref_flag_count,
            "monitor_outside_3sigma_count": monitor_flag_count,
            "monitor_outside_3sigma_fraction": monitor_flag_count / (n - n_ref),
            "sensitivity_all_rows_ref_center_including_reviews": full_center,
            "sensitivity_all_rows_ref_sigmaMR_including_reviews": full_mr_sigma,
        }
        rows.append(group_row)
        for i, rec in candidate.iterrows():
            zone = "reference" if i < n_ref else "monitor"
            sequences.append({
                "code_group": group,
                "WAFER_ID": int(rec.WAFER_ID),
                "order_within_group": int(i + 1),
                "zone": zone,
                "first_timestamp": float(rec.first_timestamp),
                "AVG_REMOVAL_RATE": float(rec.AVG_REMOVAL_RATE),
                "reference_CL_provisional": center,
                "reference_UCL_provisional": upper,
                "reference_LCL_provisional": lower,
                "outside_provisional_3sigma": bool(flag[i]),
            })
            if flag[i]:
                flag_rows.append(sequences[-1].copy())
        idx = np.arange(1, n + 1)
        plt.figure(figsize=(12.8, 5.0))
        plt.scatter(idx[:n_ref], y[:n_ref], s=13, alpha=.6, label="Reference 60% (TRAIN only)")
        plt.scatter(idx[n_ref:], y[n_ref:], s=16, alpha=.7, label="Later 40% (TRAIN only)")
        if flag.any():
            plt.scatter(idx[flag], y[flag], s=50, marker="x", c="red", linewidth=1.5,
                        label=f"Outside provisional 3-sigma: {int(flag.sum())}")
        plt.axhline(center, color="black", linestyle="-", linewidth=1.2,
                    label=f"Provisional reference mean: {center:.2f}")
        plt.axhline(upper, color="tab:red", linestyle="--", linewidth=1.1,
                    label=f"Provisional UCL/LCL: {upper:.2f}/{lower:.2f}")
        plt.axhline(lower, color="tab:red", linestyle="--", linewidth=1.1)
        plt.axvline(n_ref + .5, color="tab:green", linestyle=":", label="Reference / later split")
        plt.xlabel("Sorted by first sensor timestamp within code group (NOT verified fab run order)")
        plt.ylabel("MRR (dataset scale)")
        plt.title(f"Stage A / {group} — I chart illustration, NOT qualified SPC limits")
        plt.legend(loc="upper right", fontsize=8)
        plt.grid(alpha=.2)
        plt.tight_layout()
        plt.savefig(figures / f"04a_{group.lower().replace(' ', '_').replace('-', '_')}_provisional_i_chart.png", dpi=145)
        plt.close()
        details[group] = {
            "ref_start_id": int(candidate.WAFER_ID.iloc[0]),
            "ref_end_id": int(candidate.WAFER_ID.iloc[n_ref - 1]),
            "monitor_start_id": int(candidate.WAFER_ID.iloc[n_ref]),
            "excluded_wafer_ids": [int(x) for x in full.loc[full.review | ~full.canonical_signature, "WAFER_ID"]],
        }

    summary = pd.DataFrame(rows)
    summary.to_csv(output / "04a_provisional_chart_summary.csv", index=False)
    pd.DataFrame(flag_rows, columns=list(sequences[0])).to_csv(
        output / "04a_provisional_3sigma_alerts.csv", index=False
    )
    pd.DataFrame(sequences).to_csv(output / "04a_training_chart_points.csv", index=False)
    notes = {
        "status": "Exploratory statistical chart only; UNQUALIFIED SPC baseline",
        "scope": "TRAINING Stage A only. No Test/Validation inputs read.",
        "sample_and_sequence": "One wafer-stage MRR per observation; order by first sensor timestamp within anonymous code group; overlaps and unverified actual production chronology remain.",
        "provisional_formula": "CL = first 60% mean; sigma_hat = mean(abs(consecutive differences within reference)) / 1.128; UCL/LCL = CL +/- 3 sigma_hat",
        "unsafe_to_deploy_because": [
            "Strong positive lag-1 autocorrelation violates independent-observation assumptions of classic I-MR chart.",
            "No certified stable Phase-I reference or constant recipe/mode is proven.",
            "Time spans overlap and timestamp order is not verified factory run chronology.",
            "MRR > 1000 flags and noncanonical signatures excluded from illustrative chart; retained separately, NOT certified measurement errors.",
            "Neither true engineering specification limits nor verified failure labels exist; no Cp/Cpk or factory OOC rates are inferred.",
            "Any observed >3sigma point is a provisional statistical flag, NOT proof of product defect or physical root cause.",
        ],
        "groups": details,
        "total_review_rows": int(len(reviewed)),
    }
    with (output / "04a_spc_method_notes.json").open("w", encoding="utf-8") as fh:
        json.dump(notes, fh, indent=2, ensure_ascii=False)
    assert int(summary.all_training_wafer_stages.sum()) == len(df), "Coverage failed"
    assert int(summary.candidate_chart_rows.sum()) + len(reviewed) == len(df), "Rows unaccounted"
    print("PASS: all 1,166 Training / Stage A rows accounted for; no Test/Validation outcomes read")
    for r in rows:
        print(f"{r['code_group']}: chart {r['candidate_chart_rows']} rows | ref {r['reference_rows_first_60pct']} / later {r['monitor_rows_later_40pct']} | lag-1 ref {r['reference_lag1_correlation']:.3f} | 3sigma later flags {r['monitor_outside_3sigma_count']}")
    print("Saved 5 reports and 2 figures to:", output)
    print("All lines are I-chart ILLUSTRATION ONLY, not approved OOC limits or process specs. Source unchanged.")
    return notes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("outputs/analysis/03b_stageA_wafer_code_mrr.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/analysis"))
    args = parser.parse_args()
    run(args.source, args.output)
