"""Step 03D: validate Stage A sequence and code signatures BEFORE SPC baseline selection.

Run from the CMP-Process-QE project root:
  .venv\\Scripts\\python.exe src\\03d_timestamp_and_group_integrity.py

Inputs are derived views from Steps 03B and 03C. No raw data or earlier outputs
are edited. Do not interpret anonymous MACHINE_DATA codes as proven recipes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REQUIRED = [
    'split','WAFER_ID','STAGE','first_timestamp','last_timestamp',
    'recorded_time_span_seconds','machine_data_codes','code_group',
    'AVG_REMOVAL_RATE','mrr_review_gt_1000','clean_sensor_rows',
]
EXPECTED_GROUPS = {'Codes 1-3','Codes 4-6'}


def inspect(source: Path, out: Path) -> dict:
    if not source.is_file():
        raise FileNotFoundError(f'Cannot find Step 03B source: {source}')
    df = pd.read_csv(source)
    missing = sorted(set(REQUIRED) - set(df.columns))
    if missing:
        raise ValueError(f'Missing fields from 03B output: {missing}')
    assert not df.empty, 'Stage A analysis file is empty'
    if set(df['split'].astype(str)) != {'training'} or set(df['STAGE'].astype(str)) != {'A'}:
        raise ValueError('Source must contain Training / Stage A only. Holdout outcomes are off-limits.')
    if df[['WAFER_ID','STAGE','split']].duplicated().any():
        raise ValueError('Duplicate wafer-stage keys: audit before continuing.')
    if set(df.code_group) != EXPECTED_GROUPS:
        raise ValueError('Unexpected code_group values. Verify Step 03B grouping first.')
    if df[['first_timestamp','last_timestamp','recorded_time_span_seconds','AVG_REMOVAL_RATE']].isna().any().any():
        raise ValueError('Missing required numeric values: do not silently discard.')
    if (df.last_timestamp < df.first_timestamp).any():
        raise ValueError('Some records have end before start; audit timestamp fields first.')

    out.mkdir(parents=True, exist_ok=True)
    (out/'figures').mkdir(parents=True, exist_ok=True)
    signatures = (df.groupby(['code_group','machine_data_codes'], dropna=False)
                  .agg(wafer_stages=('WAFER_ID','size'),
                       min_mrr=('AVG_REMOVAL_RATE','min'),
                       median_mrr=('AVG_REMOVAL_RATE','median'),
                       max_mrr=('AVG_REMOVAL_RATE','max'))
                  .reset_index().sort_values(['code_group','wafer_stages'], ascending=[True,False]))
    signatures.to_csv(out/'03d_code_signatures.csv',index=False)
    expected = {'Codes 1-3':'1,2,3', 'Codes 4-6':'4,5,6'}
    suspicious = df[df.apply(lambda r: str(r.machine_data_codes).replace(' ','') != expected[r.code_group],axis=1)].copy()
    suspicious[['WAFER_ID','code_group','machine_data_codes','clean_sensor_rows','AVG_REMOVAL_RATE',
                'first_timestamp','last_timestamp']].to_csv(out/'03d_incomplete_code_signatures.csv',index=False)

    group_records = []
    pair_rows = []
    for group, subset in df.groupby('code_group'):
        z = subset.sort_values(['first_timestamp','WAFER_ID'], kind='stable').reset_index(drop=True)
        starts = z.first_timestamp.to_numpy(dtype=float)
        ends = z.last_timestamp.to_numpy(dtype=float)
        mrr = z.AVG_REMOVAL_RATE.to_numpy(dtype=float)
        span = ends-starts
        gaps = starts[1:]-starts[:-1]
        overlapping = starts[1:] < ends[:-1]
        for i in range(1,len(z)):
            pair_rows.append({'code_group':group,
                              'previous_wafer_id':int(z.WAFER_ID.iloc[i-1]),
                              'current_wafer_id':int(z.WAFER_ID.iloc[i]),
                              'previous_end':float(ends[i-1]),
                              'current_start':float(starts[i]),
                              'start_gap_seconds':float(gaps[i-1]),
                              'overlap_seconds':float(max(0,ends[i-1]-starts[i])),
                              'is_overlapping':bool(overlapping[i-1]),
                              'previous_mrr':float(mrr[i-1]),
                              'current_mrr':float(mrr[i])})
        positive = gaps[gaps>0]
        med_span = float(np.median(span))
        group_records.append({'code_group':group,
                'wafer_stages_all':len(z),
                'review_flagged_gt_1000':int(z.mrr_review_gt_1000.astype(str).str.lower().eq('true').sum()),
                'distinct_machine_code_signatures':int(z.machine_data_codes.nunique()),
                'start_timestamp_ties':int(len(starts)-len(np.unique(starts))),
                'negative_duration_count':int((span<0).sum()),
                'duration_min_seconds':float(span.min()),
                'duration_median_seconds':med_span,
                'duration_p95_seconds':float(np.quantile(span,.95)),
                'duration_max_seconds':float(span.max()),
                'duration_above_10x_median_review_count':int((span>10*med_span).sum()),
                'adjacent_pairs_by_first_timestamp':int(len(gaps)),
                'adjacent_pairs_with_overlapping_sensor_intervals':int(overlapping.sum()),
                'overlap_fraction':float(overlapping.mean()),
                'adjacent_start_gap_median_seconds':float(np.median(positive)),
                'adjacent_start_gap_p95_seconds':float(np.quantile(positive,.95))})
    summary = pd.DataFrame(group_records).sort_values('code_group')
    summary.to_csv(out/'03d_time_structure_by_group.csv',index=False)
    adjacent = pd.DataFrame(pair_rows)
    adjacent.to_csv(out/'03d_adjacent_wafer_intervals.csv',index=False)
    time_flag = df.groupby('code_group').apply(
        lambda g: g[g.recorded_time_span_seconds>10*g.recorded_time_span_seconds.median()],
        include_groups=False)
    if isinstance(time_flag.index,pd.MultiIndex):
        time_flag = time_flag.reset_index(level=0)
    time_flag[['WAFER_ID','code_group','machine_data_codes','recorded_time_span_seconds',
               'clean_sensor_rows','AVG_REMOVAL_RATE','first_timestamp','last_timestamp']].to_csv(
               out/'03d_long_duration_review.csv',index=False)
    # Plot source count for signatures (do not include labels from Test/Validation).
    fig, ax = plt.subplots(figsize=(9,4.5))
    left = signatures.copy()
    names = [f'{r.code_group}\n{r.machine_data_codes}' for r in left.itertuples()]
    bars = ax.bar(names,left.wafer_stages)
    ax.bar_label(bars,padding=3)
    ax.set_ylabel('Training wafer-stage count')
    ax.set_title('Stage A: anonymous machine-code signature completeness')
    fig.tight_layout()
    fig.savefig(out/'figures'/'03d_machine_code_signatures.png',dpi=150)
    plt.close(fig)

    # Duration/gap chart: two figures, no invented engineer limits.
    fig, ax = plt.subplots(figsize=(9,5))
    for group in sorted(df.code_group.unique()):
        subset = df[df.code_group==group]
        ax.hist(subset.recorded_time_span_seconds.clip(lower=1), bins=np.geomspace(10,50000,46),
                alpha=.60,label=group)
    ax.set_xscale('log'); ax.set_xlabel('Recorded first-to-last sensor time span (seconds, log scale)')
    ax.set_ylabel('Wafer-stage count');ax.set_title('Stage A: elapsed timestamp spans, not proven tool run time')
    ax.legend(); fig.tight_layout()
    fig.savefig(out/'figures'/'03d_recorded_time_span.png',dpi=150)
    plt.close(fig)

    notes = {
        'scope': 'Training Stage A, no Test or Validation outcomes examined',
        'pair_definition': 'Consecutive starts within an anonymous code group; current first_timestamp < previous last_timestamp = overlapping intervals',
        'important_limit': 'Overlapping intervals are not necessarily bad data and do not prove wafer co-processing; sequence is not validated production run order.',
        'code_limit': 'MACHINE_DATA is anonymous; matching grouped codes is not proof of a common recipe or identical control conditions.',
        'duration_review': 'Long duration >10 times within-group median is a review flag only (chosen for visibility), not engineering process time limit.',
        'spc_recommendation': 'Do NOT treat these exploratory metrics or per-group median as validated Phase-I stable SPC baseline; no control limits or Cp/Cpk are established.',
        'MRR': 'MRR>1000 four previously flagged Training rows retained; outcome not used to assign code group.'
    }
    (out/'03d_notes.json').write_text(json.dumps(notes,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    results={'status':'PASS','input_rows':len(df),'reviewed_groups':group_records,
             'nonstandard_signatures':len(suspicious),'duration_review_rows':len(time_flag),
             'outputs':[p.name for p in out.glob('03d_*')],
             'figures':[p.name for p in (out/'figures').glob('03d_*')]}
    return results


def main():
    parser=argparse.ArgumentParser(description='Step 03D: stage-A sequence and code integrity')
    parser.add_argument('--input',default='outputs/analysis/03b_stageA_wafer_code_mrr.csv')
    parser.add_argument('--out',default='outputs/analysis')
    args=parser.parse_args()
    r=inspect(Path(args.input),Path(args.out))
    print(f"PASS: inspected {r['input_rows']:,} Training Stage A wafer-stage records; no source changes")
    for g in r['reviewed_groups']:
        print(f"{g['code_group']}: {g['wafer_stages_all']} rows | "
              f"{g['adjacent_pairs_with_overlapping_sensor_intervals']}/{g['adjacent_pairs_by_first_timestamp']} "
              f"adjacent intervals overlap | "
              f"median span {g['duration_median_seconds']:.1f}s | "
              f"max span {g['duration_max_seconds']:.1f}s")
    print(f"Nonstandard code signatures: {r['nonstandard_signatures']}; long-duration review rows: {r['duration_review_rows']}")
    print('Saved 6 tables/notes and 2 figures to outputs/analysis')
    print('No official SPC control/specification limits calculated; Test/Validation outcomes not used.')

if __name__ == '__main__':
    main()
