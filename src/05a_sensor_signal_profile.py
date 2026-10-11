"""Step 05A: inspect PHM CMP Stage A TRAINING sensor signals, without failure labels.

Only existing derived training sensor records are read; no MRR outcomes and no
Test/Validation are consulted. Outputs are descriptive and do not establish FDC limits.
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

SIGNALS = [
    "PRESSURIZED_CHAMBER_PRESSURE",
    "MAIN_OUTER_AIR_BAG_PRESSURE",
    "CENTER_AIR_BAG_PRESSURE",
    "RETAINER_RING_PRESSURE",
    "SLURRY_FLOW_LINE_A",
    "SLURRY_FLOW_LINE_C",
    "WAFER_ROTATION",
    "STAGE_ROTATION",
    "HEAD_ROTATION",
]
SHORT = ["Chamber pressure", "Outer air bag", "Center air bag", "Retainer ring", "Slurry flow A", "Slurry flow C", "Wafer rotation", "Stage rotation", "Head rotation"]
ID = ["WAFER_ID", "STAGE", "MACHINE_DATA"]


def code_group(x: int) -> str:
    if x in (1, 2, 3):
        return "Codes 1-3"
    if x in (4, 5, 6):
        return "Codes 4-6"
    raise ValueError(f"Unexpected MACHINE_DATA={x}")


def generate(src: Path, outdir: Path) -> dict:
    assert src.is_file(), f"Cannot find {src}. Complete Step02B first."
    cols = ["split", "WAFER_ID", "STAGE", "MACHINE_DATA", "TIMESTAMP", "source_file", "source_row", "same_time_multiple_sensor_values"] + SIGNALS
    raw = pd.read_csv(src, usecols=cols)
    if not (raw["split"] == "training").all():
        raise ValueError("Only the derived TRAINING sensor file is allowed")
    a = raw.loc[raw["STAGE"] == "A"].copy()
    if len(a) != 375209 or a["WAFER_ID"].nunique() != 1166:
        raise ValueError(f"Training Stage A inventory differs from Step03B: {len(a)} rows / {a.WAFER_ID.nunique()} wafers")
    if a[SIGNALS+ID+["TIMESTAMP"]].isna().any().any():
        raise ValueError("Unexpected missing values; stop instead of filling")
    a["code_group"] = a["MACHINE_DATA"].map(code_group)
    grp = a.groupby("WAFER_ID")["code_group"].nunique()
    if not grp.eq(1).all():
        raise ValueError("Some Stage A wafers span both code groups")
    if not set(a.MACHINE_DATA.unique()).issubset({1,2,3,4,5,6}):
        raise ValueError("Unexpected MACHINE_DATA")

    outdir.mkdir(parents=True, exist_ok=True)
    figs = outdir / "figures"; figs.mkdir(exist_ok=True)
    # One row per MACHINE_DATA and sensor; event-weighted summary is deliberately named.
    event_rows=[]
    for code, piece in a.groupby("MACHINE_DATA",sort=True):
        for sensor in SIGNALS:
            v=piece[sensor]
            nonzero = v.ne(0)
            event_rows.append(dict(machine_data_code=int(code),code_group=code_group(int(code)),sensor=sensor,
                                   event_count=int(len(v)),nonzero_event_count=int(nonzero.sum()),
                                   nonzero_event_fraction=float(nonzero.mean()),
                                   all_event_median=float(v.median()),
                                   nonzero_event_median=float(v[nonzero].median()) if nonzero.any() else None))
    events = pd.DataFrame(event_rows)
    events.to_csv(outdir/"05a_event_level_signal_activity.csv",index=False)

    # One per wafer-stage-channel avoids counting sensor rows as independent wafer outcomes.
    wrows=[]
    for (wafer,stage,code), part in a.groupby(ID,sort=False):
        record={"WAFER_ID":int(wafer),"STAGE":stage,"MACHINE_DATA":int(code),
                "code_group":code_group(int(code)),"events_in_code":int(len(part)),
                "first_timestamp":float(part.TIMESTAMP.min()),"last_timestamp":float(part.TIMESTAMP.max()),
                "same_time_conflict_events":int(part.same_time_multiple_sensor_values.sum())}
        for sensor in SIGNALS:
            v=part[sensor]; nonzero = v.ne(0)
            record[sensor+"__nonzero_fraction"]=float(nonzero.mean())
            record[sensor+"__median_all"]=float(v.median())
            record[sensor+"__median_nonzero"]=float(v[nonzero].median()) if nonzero.any() else np.nan
        wrows.append(record)
    wafers=pd.DataFrame(wrows)
    wafers.to_csv(outdir/"05a_wafer_channel_signal_features.csv",index=False)
    if wafers.duplicated(ID).any():
        raise AssertionError("Duplicate wafer-stage-channel output key")
    # Equal wafer-channel weight as descriptive counterpart to row-weighted event summary.
    grow=[]
    for (code,sensor), row in events.set_index(["machine_data_code","sensor"]).iterrows():
        q=wafers[wafers.MACHINE_DATA==code]
        grow.append({"machine_data_code":int(code),"code_group":code_group(int(code)),"sensor":sensor,
                     "wafer_channel_count":int(len(q)),"median_within_wafer_nonzero_fraction":float(q[sensor+"__nonzero_fraction"].median()),
                     "event_weighted_nonzero_fraction":float(row.nonzero_event_fraction)})
    gtab=pd.DataFrame(grow)
    gtab.to_csv(outdir/"05a_code_level_signal_summary.csv",index=False)

    fig, ax=plt.subplots(figsize=(12.3,6.1))
    heat=gtab.pivot(index="sensor",columns="machine_data_code",values="median_within_wafer_nonzero_fraction").reindex(index=SIGNALS,columns=[1,2,3,4,5,6])
    im=ax.imshow(heat.values,cmap="viridis",vmin=0,vmax=1,aspect="auto")
    ax.set_xticks(range(6),[f"Code {i}" for i in range(1,7)])
    ax.set_yticks(range(len(SIGNALS)),SHORT)
    ax.set_title("TRAIN Stage A: median within-wafer nonzero fraction by anonymous code")
    for i in range(len(SIGNALS)):
        for j in range(6):
            z=heat.values[i,j]
            ax.text(j,i,f"{z:.2f}",ha="center",va="center",fontsize=8,color="white" if z<0.45 else "black")
    fig.colorbar(im,ax=ax,label="Fraction of nonzero sensor observations",shrink=0.8)
    fig.text(0.5,0.01,"A zero does not prove missing data, machine-off state, or failure. No labels / specifications used.",ha="center",fontsize=9)
    fig.tight_layout(rect=[0,.035,1,1]);fig.savefig(figs/"05a_signal_nonzero_fraction_by_code.png",dpi=145);plt.close(fig)

    # Deterministic examples chosen on sensor completeness and typical span, never MRR.
    example_records=[]
    for group, codes in [("Codes 1-3",{1,2,3}),("Codes 4-6",{4,5,6})]:
        subset=a[a.code_group==group]
        df=subset.groupby("WAFER_ID").agg(n=("TIMESTAMP","size"),first=("TIMESTAMP","min"),last=("TIMESTAMP","max"),
                                           ncodes=("MACHINE_DATA","nunique"),conflicts=("same_time_multiple_sensor_values","sum"))
        df["span"]=df["last"]-df["first"]
        candidates=df[(df.ncodes==3)&(df.conflicts==0)&(df.span>0)].copy()
        # Choose a sensor-active example for readable waveforms. This is an
        # illustrative *selected* wafer, not a representative / typical wafer.
        sensor_fraction=subset.groupby("WAFER_ID")[
            ["PRESSURIZED_CHAMBER_PRESSURE","SLURRY_FLOW_LINE_A","WAFER_ROTATION"]
        ].agg(lambda series: (series != 0).mean())
        active=(sensor_fraction>0.10).all(axis=1)
        candidates=candidates.loc[candidates.index.intersection(active[active].index)]
        if candidates.empty:
            raise ValueError(f"No eligible sensor-active examples in {group}")
        medspan=candidates.span.median(); medn=candidates.n.median()
        candidates["score"]=(candidates.span-medspan).abs()/(medspan+1)+(candidates.n-medn).abs()/(medn+1)
        chosen=int(candidates.sort_values(["score","span"],kind="mergesort").index[0])
        piece=subset[subset.WAFER_ID==chosen].sort_values(["TIMESTAMP","MACHINE_DATA","source_file","source_row"],kind="mergesort")
        fig, axes=plt.subplots(3,1,figsize=(12.5,8.7),sharex=True)
        for axis,signal in zip(axes,["PRESSURIZED_CHAMBER_PRESSURE","SLURRY_FLOW_LINE_A","WAFER_ROTATION"]):
            for code,p in piece.groupby("MACHINE_DATA"):
                t=(p.TIMESTAMP-p.TIMESTAMP.min())/(p.TIMESTAMP.max()-p.TIMESTAMP.min()) if (p.TIMESTAMP.max()>p.TIMESTAMP.min()) else np.zeros(len(p))
                axis.plot(t,p[signal],marker=".",markersize=2,linewidth=0.75,label=f"Code {int(code)}",alpha=0.7)
            axis.set_ylabel(signal.replace("_"," ").title(),fontsize=9)
            axis.legend(loc="upper right",fontsize=8)
        axes[-1].set_xlabel("Within-code normalized recorded time (0 to 1; not an absolute fab clock)")
        fig.suptitle(f"TRAIN Stage A / {group} — sensor-active selected wafer {chosen} (illustrative, not typical/fault)")
        fig.text(.5,.01,"Different code timelines are independently normalized. Zero values and repeated timestamps are retained.",ha="center",fontsize=9)
        fig.tight_layout(rect=[0,.03,1,.96]); fig.savefig(figs/f"05a_example_{group.replace(' ','').replace('-','_')}_signals.png",dpi=135);plt.close(fig)
        example_records.append({"group":group,"WAFER_ID":chosen,"sample_events":int(len(piece)),"code_signature":sorted([int(x) for x in piece.MACHINE_DATA.unique()]),"recorded_span_seconds":float(df.loc[chosen,"span"])})

    summary={"scope":"TRAINING Stage A sensors only; NO MRR, Test, Validation or failure labels read",
             "stage_A_sensor_events":int(len(a)),"stage_A_wafer_ids":int(a.WAFER_ID.nunique()),
             "machine_data_event_counts":{str(k):int(v) for k,v in a.MACHINE_DATA.value_counts().sort_index().items()},
             "wafer_channel_rows":int(len(wafers)),"output_channel_codes":sorted([int(x) for x in a.MACHINE_DATA.unique()]),
             "noncanonical_wafer_code_signatures":int((a.groupby("WAFER_ID").MACHINE_DATA.nunique()!=3).sum()),
             "same_time_conflict_flagged_event_rows":int(a.same_time_multiple_sensor_values.sum()),
             "examples_selected_using_sensors_only":example_records,
             "example_selection":"All three pictured sensors >10% nonzero among complete/no-conflict wafer signatures; closest to median span/event count of eligible active subset. Selected for visibility, not representative.",
             "limits":["Not a verified FDC fault classifier or alert system.",
                       "Zero may indicate no activity, no reading, or encoded missingness; cannot determine from these data.",
                       "Codes are anonymous and do not prove distinct real machines or recipes.",
                       "Event-weighted and wafer-channel-weighted fractions are different; do not conflate.",
                       "Timestamps may overlap and code-specific clocks need not represent exact production order.",
                       "No outcomes used for feature selection, fitting, thresholds, or visual choices."]}
    (outdir/"05a_signal_profile_notes.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(f"PASS: TRAIN Stage A {len(a):,} sensor events across {a.WAFER_ID.nunique():,} wafers; {len(wafers):,} wafer-code summaries")
    print("Machine-data codes:",summary["machine_data_event_counts"])
    print("Saved 4 tables/notes and 3 figures to",outdir)
    print("NO quality outcomes or test/validation used. No FDC alarm thresholds were calculated.")
    return summary


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",default="outputs/processed/02b_sensor_events_training.csv.gz")
    parser.add_argument("--output",default="outputs/analysis")
    args=parser.parse_args()
    generate(Path(args.input),Path(args.output))
