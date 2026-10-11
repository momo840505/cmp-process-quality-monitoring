"""Step 05B: exploratory code-specific sensor ACTIVITY reference screening.

Only uses 05A TRAINING Stage A sensor-derived features. Never reads MRR, test,
validation, or defect labels. The output is NOT a validated FDC alarm or fault.

Run from CMP-Process-QE project root:
    .\\.venv\\Scripts\\python.exe .\\src\\05b_reference_sensor_activity.py
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

# Four predefined sensor activity metrics; one pressure, one slurry,
# and two kinds of rotation (rotation is one family for candidate counting).
SENSORS = {
    "pressure": "PRESSURIZED_CHAMBER_PRESSURE__nonzero_fraction",
    "slurry": "SLURRY_FLOW_LINE_A__nonzero_fraction",
    "wafer_rotation": "WAFER_ROTATION__nonzero_fraction",
    "stage_rotation": "STAGE_ROTATION__nonzero_fraction",
}
FAMILIES = ("pressure", "slurry", "motion")
CODES = (1, 2, 3, 4, 5, 6)
LOW_Q, HIGH_Q = 0.01, 0.99
MIN_REFERENCE = 100
MIN_UNIQUE = 10
MIN_SPREAD = 0.025  # activity fraction points, not machine/quality specs


def assert_unique(df: pd.DataFrame, keys: list[str], name: str) -> None:
    n = int(df.duplicated(keys).sum())
    if n:
        raise AssertionError(f"{name}: {n} duplicate keys among {keys}")


def run(input_path: Path, out: Path) -> dict:
    if not input_path.is_file():
        raise FileNotFoundError(f"Missing Step 05A input: {input_path}")
    df = pd.read_csv(input_path)
    required = {
        "WAFER_ID", "STAGE", "MACHINE_DATA", "code_group",
        "events_in_code", "first_timestamp", "same_time_conflict_events",
        *SENSORS.values(),
    }
    miss = required - set(df.columns)
    if miss:
        raise ValueError(f"Missing columns in Step05A features: {sorted(miss)}")
    if len(df) != 3496 or df.WAFER_ID.nunique() != 1166:
        raise AssertionError("Step05A expected exactly 3,496 wafer-code rows and 1,166 wafers")
    if set(df.STAGE.astype(str)) != {"A"}:
        raise AssertionError("Step05B only supports Stage A TRAINING features")
    if set(df.MACHINE_DATA.astype(int)) != set(CODES):
        raise AssertionError("Unexpected MACHINE_DATA codes")
    assert_unique(df, ["WAFER_ID", "STAGE", "MACHINE_DATA"], "05A features")
    for col in SENSORS.values():
        if df[col].isna().any() or not df[col].between(0, 1).all():
            raise AssertionError(f"Invalid activity fraction: {col}")
    # Group assignment/wafer split must not be different between its channels.
    counts = df.groupby("WAFER_ID").agg(
        code_group_count=("code_group", "nunique"),
        code_group=("code_group", "first"),
        first_timestamp=("first_timestamp", "min"),
    ).reset_index()
    if int(counts.code_group_count.max()) != 1:
        raise AssertionError("A wafer appears in multiple code groups")
    if set(counts.code_group) != {"Codes 1-3", "Codes 4-6"}:
        raise AssertionError("Unexpected code groups")
    blocks=[]
    for group, group_df in counts.groupby("code_group", sort=True):
        group_df=group_df.sort_values(["first_timestamp","WAFER_ID"],kind="mergesort").copy()
        n_ref=int(np.floor(0.6*len(group_df)))
        group_df["reference_or_later"] = np.where(np.arange(len(group_df))<n_ref,"reference","later")
        group_df["wafer_group_order"] = np.arange(len(group_df))+1
        blocks.append(group_df[["WAFER_ID","reference_or_later","wafer_group_order"]])
    assignment=pd.concat(blocks,ignore_index=True)
    assert_unique(assignment,["WAFER_ID"],"assignment")
    df=df.merge(assignment,on="WAFER_ID",validate="many_to_one")

    sigs=df.groupby("WAFER_ID").MACHINE_DATA.agg(lambda v:tuple(sorted(v.astype(int)))).to_dict()
    expected={"Codes 1-3":(1,2,3),"Codes 4-6":(4,5,6)}
    df["complete_signature"] = [sigs[w] == expected[g] for w,g in zip(df.WAFER_ID,df.code_group)]
    df["has_timestamp_conflict"] = df.same_time_conflict_events > 0
    df["eligible_for_reference"] = (
        (df.reference_or_later=="reference") & df.complete_signature & ~df.has_timestamp_conflict
    )
    # No outcome data used. Build code-specific reference bounds from clean
    # *reference* wafer-channels only. Do not adapt bounds using later values.
    profiles=[]
    for code in CODES:
        g=df[df.MACHINE_DATA==code]
        ref=g[g.eligible_for_reference]
        for feature_name, col in SENSORS.items():
            a=ref[col].to_numpy(dtype=float)
            lo=float(np.quantile(a,LOW_Q)) if len(a) else float('nan')
            hi=float(np.quantile(a,HIGH_Q)) if len(a) else float('nan')
            nuniq=int(pd.Series(a).nunique())
            usable=bool(len(a)>=MIN_REFERENCE and nuniq>=MIN_UNIQUE and hi-lo>=MIN_SPREAD)
            profiles.append({
                "MACHINE_DATA":code, "code_group":g.code_group.iloc[0],
                "activity_feature":feature_name,"source_column":col,
                "reference_eligible_rows":len(a),"reference_q01":lo,
                "reference_q50":float(np.median(a)) if len(a) else float('nan'),
                "reference_q99":hi,"reference_unique_values":nuniq,
                "usable_exploratory_reference":usable,
                "reason_if_unused": "" if usable else "too few / nearly constant reference activity values",
            })
    ref_table=pd.DataFrame(profiles)
    if len(ref_table)!=24:
        raise AssertionError("Expected six codes times four selected signal fractions")

    for feature, col in SENSORS.items():
        this_ref=ref_table[ref_table.activity_feature==feature].set_index("MACHINE_DATA")
        low_map=this_ref.reference_q01.to_dict()
        high_map=this_ref.reference_q99.to_dict()
        use_map=this_ref.usable_exploratory_reference.to_dict()
        df[f"{feature}__reference_usable"]=df.MACHINE_DATA.map(use_map).astype(bool)
        df[f"{feature}__outside_reference"]= df[f"{feature}__reference_usable"] & (
            (df[col] < df.MACHINE_DATA.map(low_map)) | (df[col] > df.MACHINE_DATA.map(high_map))
        )

    # Correlated rotation sensors count as ONE domain, so the candidate rule
    # doesn't count them as two independent abnormalities.
    df["pressure_family_flag"]=df["pressure__outside_reference"]
    df["slurry_family_flag"]=df["slurry__outside_reference"]
    df["motion_family_flag"]=df[["wafer_rotation__outside_reference","stage_rotation__outside_reference"]].any(axis=1)
    df["flagged_family_count"]=df[[f"{x}_family_flag" for x in FAMILIES]].sum(axis=1)
    df["later_two_family_candidate"] = (df.reference_or_later=="later") & (df.flagged_family_count>=2)
    df["candidate_requires_quality_review"]= df.later_two_family_candidate & (
        df.has_timestamp_conflict | ~df.complete_signature
    )
    # Group-level summary only; do NOT pretend these are defect/false-positive rates.
    summary=[]
    for code in CODES:
        g=df[df.MACHINE_DATA==code]
        later=g[g.reference_or_later=="later"]
        summary.append({
            "MACHINE_DATA":code,"code_group":g.code_group.iloc[0],
            "reference_rows":int((g.reference_or_later=="reference").sum()),
            "reference_eligible_rows":int(g.eligible_for_reference.sum()),
            "later_rows":len(later),
            "later_rows_with_any_family_deviation":int((later.flagged_family_count>0).sum()),
            "later_two_family_candidate_rows":int(later.later_two_family_candidate.sum()),
            "later_candidates_with_data_quality_flag":int(later.candidate_requires_quality_review.sum()),
        })
    summary_table=pd.DataFrame(summary)
    candidates=df.loc[df.later_two_family_candidate].sort_values(
        ["code_group","MACHINE_DATA","wafer_group_order","WAFER_ID"],kind="mergesort"
    ).copy()
    # Keep all wafer/channel rows, including non-eligible QA records, fully auditable.
    out.mkdir(parents=True,exist_ok=True)
    figdir=out/"figures"
    figdir.mkdir(exist_ok=True)
    ref_table.to_csv(out/"05b_code_specific_reference_profiles.csv",index=False)
    df.to_csv(out/"05b_wafer_code_activity_screening.csv",index=False)
    candidates.to_csv(out/"05b_later_two_family_candidates.csv",index=False)
    summary_table.to_csv(out/"05b_screening_summary_by_code.csv",index=False)

    # Show observed calibration range by code (strictly a statistical reference).
    fig,axs=plt.subplots(2,2,figsize=(12,7),sharex=True)
    for ax,(name,col) in zip(axs.flat,SENSORS.items()):
        subset=ref_table[ref_table.activity_feature==name].sort_values('MACHINE_DATA')
        for _,r in subset.iterrows():
            code=int(r.MACHINE_DATA)
            if bool(r.usable_exploratory_reference):
                ax.plot([r.reference_q01,r.reference_q99],[code,code],linewidth=4,solid_capstyle='round',alpha=.75)
                ax.scatter(r.reference_q50,code,s=22)
            else:
                ax.scatter(r.reference_q50,code,marker='x',s=35)
        ax.set_title(name.replace('_',' ').title())
        ax.set_yticks(CODES)
        ax.set_ylabel('Anonymous code')
        ax.grid(True,axis='x',alpha=.25)
        ax.set_xlim(-.02,1.02)
    for ax in axs[-1]:ax.set_xlabel('Fraction of nonzero sensor records (0–1)')
    fig.suptitle('Step 05B: reference-only sensor activity intervals by code (q01–q99)')
    fig.text(.5,.01,'X = unsuitable / near-constant reference; NOT engineering or FDC alarm limits',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.94))
    fig.savefig(figdir/"05b_reference_activity_ranges_by_code.png",dpi=145)
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(10,5))
    xs=np.arange(len(summary_table))
    ax.bar(xs-.2,summary_table.later_rows_with_any_family_deviation,width=.4,label='Any one signal family outside q01/q99')
    ax.bar(xs+.2,summary_table.later_two_family_candidate_rows,width=.4,label='At least 2 distinct signal families')
    ax.set_xticks(xs,[f"Code {i}" for i in summary_table.MACHINE_DATA]);ax.set_ylabel('Later wafer–code observations (count)')
    ax.set_title('Step 05B: exploratory out-of-reference signal activity (TRAIN only)')
    ax.legend();ax.grid(True,axis='y',alpha=.25)
    fig.text(.5,.02,'Not defects, not validated fault alarms; each code has its own frozen reference',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,1))
    fig.savefig(figdir/"05b_later_activity_candidate_counts.png",dpi=145)
    plt.close(fig)

    diagnostics={
        "scope":"Training Stage A only; 05A sensor-derived wafer/channel features; no MRR labels, Test, Validation or defects",
        "input_rows":len(df),"unique_wafers":int(df.WAFER_ID.nunique()),
        "reference_split":"first 60% unique wafers in each code group ordered by earliest available timestamp; other 40% later",
        "note_on_time":"timestamp order is not verified production run order; same-wafer channels assigned together",
        "calibration":"Per code from reference wafer-channels with full code signature and no same-time conflict; 1st to 99th percentiles",
        "predefined_metrics":list(SENSORS.values()),
        "feature_eligibility":{"minimum_eligible_reference_rows":MIN_REFERENCE,"minimum_distinct_values":MIN_UNIQUE,"minimum_q99_minus_q01":MIN_SPREAD},
        "later_candidate_rule":"For the same later wafer-code, at least two families outside their code-specific reference ranges: pressure, slurry, motion (wafer/stage rotation count as ONE)",
        "limitation":"Thresholds are descriptive, 1%/99% chosen in advance; reference is not proven stable/normal; no true faults or known false alarm rate; multiple comparisons and correlated features remain.",
        "zero_warning":"Zero sensor values may represent normal inactivity, no reading or encoded missingness. Only activity rates monitored here, NOT sensor magnitude.",
        "data_quality":"Same-time conflict records and incomplete code signatures retained and flagged; candidates requiring QA review explicitly marked.",
        "counts_by_code":summary,
        "total_later_candidates":int(df.later_two_family_candidate.sum()),
        "total_usable_feature_code_profiles":int(ref_table.usable_exploratory_reference.sum()),
        "no_source_mutation":True,
    }
    with (out/"05b_screening_method_notes.json").open('w',encoding='utf-8') as f:
        json.dump(diagnostics,f,indent=2,ensure_ascii=False)
    # Final sanity check: don't drop rows or leak entire wafer across temporal subsets.
    assert len(df)==3496
    assert df.groupby('WAFER_ID').reference_or_later.nunique().eq(1).all()
    assert (df.reference_or_later=='later').sum()==int(summary_table.later_rows.sum())
    assert int(candidates.candidate_requires_quality_review.sum()) <= len(candidates)
    print('PASS: 3,496 Stage A TRAIN wafer-code rows; 1,166 distinct wafers; no MRR/Test/Validation read')
    for _,r in summary_table.iterrows():
        print(f"Code {int(r.MACHINE_DATA)}: clean reference {int(r.reference_eligible_rows)}/{int(r.reference_rows)} | "
              f"later {int(r.later_rows)} | >=2-family review candidates {int(r.later_two_family_candidate_rows)} "
              f"(QA-flagged {int(r.later_candidates_with_data_quality_flag)})")
    print(f"Output: {out.resolve()}; descriptive sensor-activity deviations ONLY, not verified FDC alarms")
    return diagnostics


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=Path('outputs/analysis/05a_wafer_channel_signal_features.csv'))
    p.add_argument('--output',type=Path,default=Path('outputs/analysis'))
    a=p.parse_args()
    run(a.input,a.output)

if __name__=='__main__':
    main()
