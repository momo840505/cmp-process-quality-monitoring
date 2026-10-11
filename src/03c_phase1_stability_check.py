"""Step 03C: TRAINING-only Stage A preliminary stability diagnostics.

This does NOT calculate SPC control limits or assess specification compliance.
The four >1000 observations are retained in the source and in the review report;
a separately named sensitivity view omits them to inspect ordinary-scale behavior.
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

GROUPS = ['Codes 1-3', 'Codes 4-6']
COLS = ['split', 'WAFER_ID', 'STAGE', 'code_group', 'first_timestamp',
        'AVG_REMOVAL_RATE', 'mrr_review_gt_1000']


def parse_args():
    p = argparse.ArgumentParser(description='Preliminary, training-only CMP Stage A stability diagnostics')
    p.add_argument('--input', type=Path, default=Path('outputs/analysis/03b_stageA_wafer_code_mrr.csv'))
    p.add_argument('--output', type=Path, default=Path('outputs/analysis'))
    return p.parse_args()


def assert_dataset(df: pd.DataFrame):
    missing = set(COLS).difference(df.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    if len(df) != 1166 or not df['split'].eq('training').all() or not df['STAGE'].eq('A').all():
        raise ValueError('Expected exactly 1,166 Training Stage A rows; do not use Test/Validation')
    if df.duplicated(['split', 'WAFER_ID', 'STAGE']).any():
        raise ValueError('Duplicate wafer-stage keys')
    if df['code_group'].isna().any() or set(df['code_group']) != set(GROUPS):
        raise ValueError('Unexpected or missing code_group; check Step 03B output')
    for col in ['first_timestamp', 'AVG_REMOVAL_RATE']:
        if not np.isfinite(pd.to_numeric(df[col], errors='coerce')).all():
            raise ValueError(f'{col} has missing or nonfinite values')
    flags = df['mrr_review_gt_1000'].astype(str).str.lower().map({'true': True, 'false': False})
    if flags.isna().any():
        raise ValueError('Could not parse mrr_review_gt_1000')
    if not flags.equals(df['AVG_REMOVAL_RATE'].gt(1000)):
        raise ValueError('Step 03B review flags do not match original inspection rule')
    df['review_flag'] = flags
    counts = df.groupby('code_group', sort=False).size().to_dict()
    if counts != {'Codes 4-6': 798, 'Codes 1-3': 368} and counts != {'Codes 1-3': 368, 'Codes 4-6': 798}:
        raise ValueError(f'Code-group counts changed: {counts}')
    if int(df['review_flag'].sum()) != 4:
        raise ValueError('Expected four pre-identified review flags')
    return df


def plot_group(group: str, part: pd.DataFrame, folder: Path):
    part = part.sort_values(['first_timestamp', 'WAFER_ID'], kind='stable').reset_index(drop=True)
    values = part['AVG_REMOVAL_RATE'].to_numpy(dtype=float)
    smooth = part['AVG_REMOVAL_RATE'].rolling(window=25, min_periods=10).median()
    med = float(np.median(values))
    plt.figure(figsize=(12, 5.7))
    plt.scatter(np.arange(1, len(part) + 1), values, s=13, alpha=.65, label='Wafer-stage MRR')
    plt.plot(np.arange(1, len(part) + 1), smooth, linewidth=2, label='25-point rolling median (descriptive)')
    plt.axhline(med, color='grey', linestyle='--', linewidth=1.4, label=f'Overall median = {med:.2f} (not CL)')
    plt.xlabel('Training wafer-stage order by first sensor timestamp (not validated fab chronology)')
    plt.ylabel('MRR (dataset scale)')
    plt.title(f'Stage A / {group} — preliminary sequence stability (excluding review flags)')
    plt.legend(loc='best');plt.grid(alpha=.2);plt.tight_layout()
    name = '03c_stageA_' + ('codes_1_3' if group == 'Codes 1-3' else 'codes_4_6') + '_sequence.png'
    plt.savefig(folder/name, dpi=160)
    plt.close()


def main():
    args = parse_args()
    df = assert_dataset(pd.read_csv(args.input))
    out = args.output
    figs = out / 'figures'
    figs.mkdir(parents=True, exist_ok=True)
    rows, qrows = [], []
    review = df.loc[df.review_flag, ['split','WAFER_ID','STAGE','code_group','first_timestamp','AVG_REMOVAL_RATE']].copy()
    review.sort_values(['first_timestamp','WAFER_ID']).to_csv(out/'03c_review_flagged_rows.csv',index=False)
    for group in GROUPS:
        allg = df.loc[df.code_group.eq(group)]
        part = allg.loc[~allg.review_flag].sort_values(['first_timestamp','WAFER_ID'],kind='stable').reset_index(drop=True)
        mrr = part.AVG_REMOVAL_RATE.to_numpy(float)
        # Correlation is descriptive only. Independence cannot be established by this statistic alone.
        lag_corr = float(np.corrcoef(mrr[:-1],mrr[1:])[0,1]) if len(mrr)>2 else None
        for qi, indexes in enumerate(np.array_split(np.arange(len(part)),4),start=1):
            chunk = mrr[indexes]
            qrows.append({'code_group':group,'time_order_quartile':qi,'n':len(chunk),
                          'mrr_median':float(np.median(chunk)),
                          'mrr_mean':float(np.mean(chunk)),
                          'mrr_p25':float(np.percentile(chunk,25)),
                          'mrr_p75':float(np.percentile(chunk,75))})
        group_quartiles = [x['mrr_median'] for x in qrows if x['code_group']==group]
        rows.append({'code_group':group,'all_rows':len(allg),'review_flagged_rows_retained_elsewhere':int(allg.review_flag.sum()),
                     'ordinary_scale_sensitivity_rows':len(part),
                     'median_mrr':float(np.median(mrr)),
                     'mean_mrr':float(np.mean(mrr)),
                     'sample_sd_mrr':float(np.std(mrr,ddof=1)),
                     'mrr_min':float(np.min(mrr)), 'mrr_max':float(np.max(mrr)),
                     'lag1_correlation_in_recorded_order':lag_corr,
                     'quartile_medians_range':float(max(group_quartiles)-min(group_quartiles)),
                     'sensor_timestamp_ties':int(part.first_timestamp.duplicated().sum())})
        plot_group(group,part,figs)
    summary = pd.DataFrame(rows)
    summary.to_csv(out/'03c_phase1_summary.csv',index=False)
    pd.DataFrame(qrows).to_csv(out/'03c_order_quartiles.csv',index=False)
    meta={'scope':'TRAINING Stage A only',
          'sequence_basis':'first_timestamp sorted within each code group; not verified fab calendar or chronological independent batches',
          'review_rule':'Pre-existing MRR > 1000 data-review flag; not a quality specification',
          'sensitivity_view':'Four flagged rows not plotted/included in ordinary-scale summary; raw/processed data and review file retain them',
          'spc_decision':'No valid SPC control limits or Cp/Cpk are established by this diagnostic',
          'data_groups':summary.to_dict(orient='records')}
    (out/'03c_phase1_notes.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    for item in rows:
        print(f"{item['code_group']}: {item['ordinary_scale_sensitivity_rows']} ordinary-scale rows + "
              f"{item['review_flagged_rows_retained_elsewhere']} review flags | median {item['median_mrr']:.3f} | "
              f"lag-1 correlation {item['lag1_correlation_in_recorded_order']:.3f} | "
              f"quartile median range {item['quartile_medians_range']:.3f}")
    print('PASS: all 1,166 Training Stage A wafer-stage rows accounted for; no Test/Validation used.')
    print(f'Saved 4 reports and 2 charts to {out.resolve()}')
    print('No SPC control or engineering specification limits calculated; no source data modified.')

if __name__=='__main__':
    main()
