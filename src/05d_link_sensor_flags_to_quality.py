"""Step 05D | Frozen sensor-screen candidates vs Stage A quality outcomes.

This module is descriptive only. It does NOT train a fault classifier,
create quality specification limits, or modify Steps 05B/05C thresholds.

Data provenance:
- outputs/analysis/05c_wafer_code_magnitude_screening.csv (sensor-only flags)
- outputs/analysis/05c_later_magnitude_candidates.csv (integrity cross-check)
- outputs/analysis/05c_magnitude_reference_profiles.csv (frozen sensor ranges)
- outputs/processed/02b_wafer_stage_quality.csv (quality, joined LAST)

Output:
- outputs/analysis/05d_later_wafer_quality_link.csv
- outputs/analysis/05d_group_comparison.csv
- outputs/analysis/05d_candidate_case_review.csv
- outputs/analysis/05d_method_notes.json
- outputs/analysis/figures/05d_codes_1_3_later_mrr.png
- outputs/analysis/figures/05d_codes_4_6_later_mrr.png
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

GROUPS = ("Codes 1-3", "Codes 4-6")
SIGNALS = {
    "chamber_pressure": ("PRESSURIZED_CHAMBER_PRESSURE", "pressure"),
    "retainer_pressure": ("RETAINER_RING_PRESSURE", "pressure"),
    "slurry_flow_A": ("SLURRY_FLOW_LINE_A", "slurry"),
    "slurry_flow_C": ("SLURRY_FLOW_LINE_C", "slurry"),
    "wafer_rotation": ("WAFER_ROTATION", "motion"),
    "stage_rotation": ("STAGE_ROTATION", "motion"),
}


def load_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing input: {path} — complete previous steps first")
    return pd.read_csv(path)


def normalize_bool(series: pd.Series, column: str) -> pd.Series:
    """Require exactly true/false-style values; fail rather than coerce unknown flags."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    values = series.astype("string").str.lower().str.strip()
    allowed = {"true": True, "false": False, "1": True, "0": False}
    if not values.isin(allowed).all():
        bad = values[~values.isin(allowed)].head().tolist()
        raise ValueError(f"Unexpected {column} flag values: {bad}")
    return values.map(allowed).astype(bool)


def _require_unique(df: pd.DataFrame, keys: list[str], label: str) -> None:
    if df.duplicated(keys).any():
        raise ValueError(f"Duplicate keys found in {label}: {keys}")


def construct_wafer_tables(sensor: pd.DataFrame, quality: pd.DataFrame,
                           candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Freeze sensor decisions and establish one row per wafer before quality join."""
    key = ["WAFER_ID", "STAGE", "MACHINE_DATA"]
    for col in key + ["code_group", "reference_or_later", "05c__later_two_family_candidate"]:
        if col not in sensor:
            raise ValueError(f"Missing Step05C input column: {col}")
    _require_unique(sensor, key, "05C screening")
    s = sensor.copy()
    s["05c__later_two_family_candidate"] = normalize_bool(
        s["05c__later_two_family_candidate"], "05c__later_two_family_candidate")
    s["05c__later_any_magnitude_deviation"] = normalize_bool(
        s["05c__later_any_magnitude_deviation"], "05c__later_any_magnitude_deviation")
    if (s["STAGE"] != "A").any() or set(s["code_group"]) != set(GROUPS):
        raise ValueError("Step05D expects only Stage A and both known code groups")
    if not s["reference_or_later"].isin(["reference", "later"]).all():
        raise ValueError("Unknown reference/later partition")
    if s.loc[s["reference_or_later"] == "reference", "05c__later_two_family_candidate"].any():
        raise ValueError("Candidate flag appears in reference rows")

    # Verify that every wafer is consistently assigned to exactly one mode and partition.
    wafer_key = ["WAFER_ID", "STAGE"]
    uniq = s.groupby(wafer_key, sort=False).agg(
        n_groups=("code_group", "nunique"),
        n_partitions=("reference_or_later", "nunique"),
    )
    if ((uniq.n_groups != 1) | (uniq.n_partitions != 1)).any():
        raise ValueError("A wafer appears in multiple code groups or partitions")
    if s.WAFER_ID.nunique() != 1166 or len(s) != 3496:
        raise ValueError(f"Unexpected input rows: {len(s)} / wafer count {s.WAFER_ID.nunique()}")

    # Cross-check 05C's independent candidate CSV without using quality labels.
    c = candidates.copy()
    _require_unique(c, key, "05C candidate file")
    expected = s.loc[s["05c__later_two_family_candidate"], key]
    actual_keys = set(map(tuple, c[key].itertuples(index=False, name=None)))
    expected_keys = set(map(tuple, expected.itertuples(index=False, name=None)))
    if actual_keys != expected_keys:
        raise ValueError("05C candidate file disagrees with full 05C screening")

    # Aggregate multiple code rows into exactly one wafer-stage observation.
    wafer = s.groupby(wafer_key, as_index=False).agg(
        code_group=("code_group", "first"),
        reference_or_later=("reference_or_later", "first"),
        sensor_candidate=("05c__later_two_family_candidate", "any"),
        magnitude_any_code_count=("05c__later_any_magnitude_deviation", "sum"),
        candidate_code_count=("05c__later_two_family_candidate", "sum"),
        first_timestamp=("first_timestamp", "min"),
    )
    _require_unique(wafer, wafer_key, "wafer-level sensor flags")

    # Read Stage A training labels only. No Test/Validation outcomes in analysis.
    q = quality.loc[(quality["split"] == "training") & (quality["STAGE"] == "A"),
                    wafer_key + ["AVG_REMOVAL_RATE", "mrr_review_gt_1000"]].copy()
    _require_unique(q, wafer_key, "Stage A TRAIN labels")
    if len(q) != 1166 or q["AVG_REMOVAL_RATE"].isna().any():
        raise ValueError("Expected 1,166 nonmissing Stage A training quality measurements")
    q["mrr_review_gt_1000"] = normalize_bool(q["mrr_review_gt_1000"], "MRR review")
    linked = wafer.merge(q, on=wafer_key, how="left", validate="one_to_one", indicator=True)
    if not (linked["_merge"] == "both").all() or len(linked) != len(q):
        raise ValueError("Mismatch between Stage A wafer keys and quality labels")
    linked = linked.drop(columns="_merge")
    return linked, s


def build_comparison(linked: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use reference-only MRR to describe groups; NEVER to retune sensor flags."""
    ref_rows, all_later, stats = [], [], []
    for group in GROUPS:
        r = linked.loc[(linked.code_group == group) &
                       (linked.reference_or_later == "reference")].copy()
        later = linked.loc[(linked.code_group == group) &
                           (linked.reference_or_later == "later")].copy()
        if r.empty or later.empty:
            raise ValueError(f"Empty reference or later group: {group}")
        ref_median = float(r.AVG_REMOVAL_RATE.median())
        ref_q05 = float(r.AVG_REMOVAL_RATE.quantile(.05))
        ref_q95 = float(r.AVG_REMOVAL_RATE.quantile(.95))
        for name, values in [("reference_median", ref_median), ("reference_q05", ref_q05),
                             ("reference_q95", ref_q95)]:
            later[name] = values
        later["mrr_abs_difference_from_ref_median"] = (
            later.AVG_REMOVAL_RATE - ref_median).abs()
        later["outside_reference_q05_q95_descriptive"] = (
            (later.AVG_REMOVAL_RATE < ref_q05) | (later.AVG_REMOVAL_RATE > ref_q95))
        # Quantile range here is descriptive only, not a spec, fault label or SPC bound.
        all_later.append(later)
        ref_rows.append({
            "code_group": group, "reference_wafer_count": len(r),
            "reference_median": ref_median, "reference_q05": ref_q05,
            "reference_q95": ref_q95, "reference_review_gt1000": int(r.mrr_review_gt_1000.sum())
        })
        for is_cand in (False, True):
            x = later.loc[later.sensor_candidate == is_cand]
            stats.append({
                "code_group": group, "sensor_candidate": bool(is_cand),
                "later_wafers": len(x),
                "reference_n": len(r), "reference_mrr_median": ref_median,
                "reference_mrr_q05": ref_q05, "reference_mrr_q95": ref_q95,
                "later_mrr_median": float(x.AVG_REMOVAL_RATE.median()) if len(x) else np.nan,
                "later_abs_deviation_median": float(x.mrr_abs_difference_from_ref_median.median()) if len(x) else np.nan,
                "later_outside_reference_q05_q95_count": int(x.outside_reference_q05_q95_descriptive.sum()),
                "later_mrr_gt1000_review_count": int(x.mrr_review_gt_1000.sum()),
            })
    later_out = pd.concat(all_later, ignore_index=True)
    assert len(later_out) == 468, f"Expected 468 later wafers; got {len(later_out)}"
    return later_out, pd.DataFrame(stats)


def build_case_review(sensor: pd.DataFrame, later: pd.DataFrame,
                      profile: pd.DataFrame) -> pd.DataFrame:
    """Explain which exact sensor reference deviations triggered each case."""
    flagged = sensor.loc[sensor["05c__later_two_family_candidate"]].copy()
    _require_unique(profile, ["MACHINE_DATA", "signal"], "05C sensor profiles")
    cases = flagged.merge(
        later[["WAFER_ID", "STAGE", "code_group", "AVG_REMOVAL_RATE",
               "reference_median", "reference_q05", "reference_q95",
               "mrr_abs_difference_from_ref_median", "mrr_review_gt_1000",
               "outside_reference_q05_q95_descriptive"]],
        on=["WAFER_ID", "STAGE", "code_group"], validate="many_to_one")
    # Capture exact observed vs frozen bounds for each triggered signal.
    evidence = []
    profile_lookup = profile.set_index(["MACHINE_DATA", "signal"])
    for _, row in cases.iterrows():
        details = []
        for short, (source, family) in SIGNALS.items():
            key = f"05c__{short}__outside_ref"
            if key not in row:
                raise ValueError(f"Expected per-signal flag {key}")
            if not normalize_bool(pd.Series([row[key]]), key).iat[0]:
                continue
            pkey = (row.MACHINE_DATA, short)
            if pkey not in profile_lookup.index:
                raise ValueError(f"Missing frozen reference profile for {pkey}")
            limits = profile_lookup.loc[pkey]
            value = row[f"{source}__median_nonzero"]
            if pd.isna(value) or pd.isna(limits.q01) or pd.isna(limits.q99):
                raise ValueError(f"Flagged signal lacks value or thresholds for {pkey}")
            details.append({"signal": short, "family": family,
                            "observed_median_nonzero": round(float(value), 6),
                            "reference_q01": round(float(limits.q01), 6),
                            "reference_q99": round(float(limits.q99), 6),
                            "direction": "below" if value < limits.q01 else "above"})
        evidence.append(json.dumps(details, ensure_ascii=False))
    cases["frozen_sensor_evidence_json"] = evidence
    return cases[["WAFER_ID", "STAGE", "MACHINE_DATA", "code_group",
                  "05c__pressure_family_flag", "05c__slurry_family_flag", "05c__motion_family_flag",
                  "05c__family_count", "has_timestamp_conflict", "05c__requires_data_quality_review",
                  "AVG_REMOVAL_RATE", "reference_median", "reference_q05", "reference_q95",
                  "mrr_abs_difference_from_ref_median", "outside_reference_q05_q95_descriptive",
                  "mrr_review_gt_1000", "frozen_sensor_evidence_json"]].sort_values(
                      ["code_group", "MACHINE_DATA", "WAFER_ID"])


def plot_group(later: pd.DataFrame, group: str, output: Path) -> None:
    d = later.loc[later.code_group == group].sort_values("WAFER_ID").copy()
    # Use the previously frozen sensor order instead of assuming WAFER_ID is production time.
    d = d.sort_values("first_timestamp") if "first_timestamp" in d else d.sort_values("WAFER_ID")
    ref_median = float(d.reference_median.iloc[0]); lo = float(d.reference_q05.iloc[0]); hi = float(d.reference_q95.iloc[0])
    beyond = d.mrr_review_gt_1000
    show = d.loc[~beyond].copy()
    fig, ax = plt.subplots(figsize=(11, 4.5))
    order_map = {id_:i for i,id_ in enumerate(d.WAFER_ID.tolist())}
    xs = show.WAFER_ID.map(order_map)
    non = ~show.sensor_candidate
    ax.scatter(xs.loc[non], show.loc[non, "AVG_REMOVAL_RATE"], s=15, alpha=.55,
               color="#5a9bd5", label=f"Other later wafers (n={int(non.sum())})")
    ax.scatter(xs.loc[~non], show.loc[~non, "AVG_REMOVAL_RATE"], marker="x", s=82,
               linewidths=2.1, color="#c03c30", label=f"Frozen 05C sensor candidates (n={int((~non).sum())})")
    ax.axhline(ref_median, color="#333333", linewidth=1.5, label=f"Reference median {ref_median:.2f}")
    ax.axhline(lo, linestyle="--", color="gray", label=f"Reference q05/q95 ({lo:.1f}, {hi:.1f})")
    ax.axhline(hi, linestyle="--", color="gray")
    ax.set_title(f"Training Stage A / {group}: frozen sensor candidates vs measured MRR")
    ax.set_ylabel("MRR (published dataset scale)")
    ax.set_xlabel("Index within later 40% sorted by first available timestamp (not verified fab run order)")
    ax.grid(alpha=.23)
    ax.legend(loc="best", fontsize=8)
    fig.text(.5,.01,
       f"Reference q05/q95 are descriptive, NOT limits. {int(beyond.sum())} MRR>1000 review records omitted from display only; no records deleted.",
       ha="center", fontsize=8)
    fig.tight_layout(rect=(0,.045,1,1))
    fig.savefig(output, dpi=160)
    plt.close(fig)


def run(project: Path) -> dict:
    data = project / "outputs" / "analysis"
    quality_path = project / "outputs" / "processed" / "02b_wafer_stage_quality.csv"
    sensor = load_csv(data / "05c_wafer_code_magnitude_screening.csv")
    candidates = load_csv(data / "05c_later_magnitude_candidates.csv")
    profiles = load_csv(data / "05c_magnitude_reference_profiles.csv")
    quality = load_csv(quality_path)
    linked, checked_sensor = construct_wafer_tables(sensor, quality, candidates)
    later, comparison = build_comparison(linked)
    case_review = build_case_review(checked_sensor, later, profiles)
    if len(case_review) != 11 or later.sensor_candidate.sum() != 11:
        raise ValueError("Frozen 05C candidate count mismatch (expected 11 wafer-code and 11 unique wafers)")
    if not later.WAFER_ID.is_unique:
        raise ValueError("Non-unique later wafers: don't count multiple codes as separate quality outcomes")
    if later.loc[later.sensor_candidate, "mrr_review_gt_1000"].any():
        print("NOTICE: One or more sensor candidates have >1000 MRR review flags; no rows dropped")
    figdir = data / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    later.to_csv(data / "05d_later_wafer_quality_link.csv", index=False)
    comparison.to_csv(data / "05d_group_comparison.csv", index=False)
    case_review.to_csv(data / "05d_candidate_case_review.csv", index=False)
    plot_group(later, GROUPS[0], figdir / "05d_codes_1_3_later_mrr.png")
    plot_group(later, GROUPS[1], figdir / "05d_codes_4_6_later_mrr.png")
    notes = {
        "status": "EXPLORATORY LABEL JOIN ONLY; not validated FDC or root-cause identification",
        "scope": "TRAINING Stage A only; exactly 468 later wafer-stage outcomes; frozen 05C sensor screen",
        "no_leakage_design": "05C flags and candidate membership never use MRR; 05D joins quality last. Reference-only MRR median/q05/q95 describe baseline, do not alter thresholds, selection or sensor features.",
        "unit_of_analysis": "One wafer-stage, not one sensor event and not one wafer-code measurement; 11 candidate wafer-codes correspond to 11 distinct later wafers.",
        "comparison": "Later candidates vs later non-candidates within anonymous code group. Difference from group's reference median; medians and distribution only, no significance or detection accuracy claims.",
        "sample_limit": "Only 2 candidates in Codes 1-3 and 9 in Codes 4-6; comparisons unstable. Groups and reference are not verified same recipe/stable process.",
        "quality_limits": "No engineering MRR target, USL/LSL, actual defect labels, or validated FDC outcomes. q05/q95 is descriptive and does not define good/bad wafers.",
        "known_review": "All four MRR>1000 Stage A training points retained, explicitly flagged. Plot may omit them from display only, never from result counts.",
        "time_limit": "Earliest available timestamp sort is not confirmed fabrication run order. Interval overlap can occur.",
        "candidate_by_code": {str(int(k)):int(v) for k,v in case_review.MACHINE_DATA.value_counts().sort_index().items()},
        "n_later_wafer": int(len(later)),
        "n_candidates": int(len(case_review)),
        "files_written": ["05d_later_wafer_quality_link.csv", "05d_group_comparison.csv", "05d_candidate_case_review.csv", "05d_method_notes.json", "figures/05d_codes_1_3_later_mrr.png", "figures/05d_codes_4_6_later_mrr.png"],
    }
    with (data / "05d_method_notes.json").open("w",encoding="utf-8") as f:
        json.dump(notes,f,indent=2,ensure_ascii=False)
    print(f"PASS: {len(linked):,} Stage A TRAIN wafers; {len(later):,} later wafers; {len(case_review)} frozen 05C candidates")
    for g in GROUPS:
        subset=later[later.code_group==g]
        flags=subset[subset.sensor_candidate]
        print(f"{g}: later {len(subset)}; sensor candidates {len(flags)}; "
              f"candidate MRR median {flags.AVG_REMOVAL_RATE.median():.3f}; "
              f"reference median {subset.reference_median.iloc[0]:.3f}")
    print("Saved 4 tables/notes and 2 charts to outputs/analysis")
    print("No sensor threshold recalibration. No Test/Validation outcomes. No source changes.")
    return notes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 05D | Frozen FDC candidates vs quality")
    parser.add_argument("--project",type=Path,default=Path("."),help="CMP-Process-QE root")
    args=parser.parse_args()
    run(args.project.resolve())
