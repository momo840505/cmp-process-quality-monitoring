"""Step 03B — Training-only exploration of Stage A's two MRR groups.

Input: Step 02B cleaned sensor events and wafer-stage quality table.
Outputs: training-only wafer/code table, aggregate tables and figures.

Descriptive 90/135/1000 bins are EDA labels, NOT process specs, defects,
SPC limits, or valid labels to train supervised models without validation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Save plots without a graphical desktop.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SENSORS = [
    'PRESSURIZED_CHAMBER_PRESSURE',
    'MAIN_OUTER_AIR_BAG_PRESSURE',
    'SLURRY_FLOW_LINE_A',
    'WAFER_ROTATION',
]


def assign_mrr_band(v: float) -> str:
    if v < 90:
        return 'MRR < 90 (descriptive)'
    if v < 135:
        return '90 <= MRR < 135 (gap)'
    if v < 1000:
        return '135 <= MRR < 1000 (descriptive)'
    return 'MRR >= 1000 (review only)'


def categorize_codes(raw_codes: tuple[int, ...]) -> str:
    codes = set(raw_codes)
    if codes and codes.issubset({1, 2, 3}):
        return 'Codes 1-3'
    if codes and codes.issubset({4, 5, 6}):
        return 'Codes 4-6'
    return 'Mixed/other codes'


def main() -> None:
    p = argparse.ArgumentParser(description='Stage A process-code / MRR association, training only')
    p.add_argument('--project-root', type=Path, default=ROOT)
    args = p.parse_args()
    root = args.project_root.resolve()
    proc = root / 'outputs' / 'processed'
    audit = root / 'outputs' / 'analysis'
    figs = audit / 'figures'
    figs.mkdir(parents=True, exist_ok=True)

    quality_path = proc / '02b_wafer_stage_quality.csv'
    sensors_path = proc / '02b_sensor_events_training.csv.gz'
    for f in [quality_path, sensors_path]:
        if not f.is_file():
            raise FileNotFoundError(f'Missing input: {f}')

    quality = pd.read_csv(quality_path)
    train_a = quality.loc[(quality['split'] == 'training') & (quality['STAGE'] == 'A')].copy()
    if train_a.empty:
        raise RuntimeError('Training Stage A not found')
    if train_a.duplicated(['WAFER_ID', 'STAGE']).any():
        raise ValueError('Duplicate wafer-stage labels: stop and investigate')
    if train_a['AVG_REMOVAL_RATE'].isna().any():
        raise ValueError('Missing MRR labels: stop and investigate')

    sensors = pd.read_csv(sensors_path)
    sensors = sensors.loc[(sensors['split'] == 'training') & (sensors['STAGE'] == 'A')].copy()
    if sensors.empty:
        raise RuntimeError('No Training Stage A sensor events found')
    if sensors['WAFER_ID'].isna().any() or sensors['MACHINE_DATA'].isna().any():
        raise ValueError('Missing wafer or MACHINE_DATA codes')
    if not set(SENSORS).issubset(sensors.columns):
        raise ValueError(f'Missing sensor columns: {set(SENSORS) - set(sensors.columns)}')

    # One row per wafer-stage: derive identifiers from sensor data only.
    # MACHINE_DATA is an anonymized code, not a confirmed physical recipe.
    grp = sensors.groupby('WAFER_ID', sort=False)
    code_sets = grp['MACHINE_DATA'].agg(lambda col: tuple(sorted({int(x) for x in col})))
    chamber_sets = grp['CHAMBER'].agg(lambda col: tuple(sorted({int(x) for x in col})))
    features = pd.DataFrame(index=code_sets.index)
    features['machine_data_codes'] = code_sets.map(lambda x: ','.join(map(str, x)))
    features['chamber_codes'] = chamber_sets.map(lambda x: ','.join(map(str, x)))
    features['code_group'] = code_sets.map(categorize_codes)
    features['sensor_events_used'] = grp.size()
    for sensor in SENSORS:
        features[f'{sensor}_nonzero_fraction'] = grp[sensor].agg(lambda x: float((x != 0).mean()))
        features[f'{sensor}_median'] = grp[sensor].median()

    result = train_a.merge(features, how='left', left_on='WAFER_ID', right_index=True, validate='one_to_one')
    if result['code_group'].isna().any():
        raise ValueError('Some training wafer-stage quality rows have no sensor data')
    if set(sensors['WAFER_ID']) != set(train_a['WAFER_ID']):
        raise ValueError('Unmatched wafer IDs between stage A labels and sensors')
    if not (result['sensor_events_used'] == result['clean_sensor_rows']).all():
        raise ValueError('Clean sensor row count does not match quality inventory')
    result['mrr_band_descriptive_only'] = result['AVG_REMOVAL_RATE'].map(assign_mrr_band)

    out_wafer = audit / '03b_stageA_wafer_code_mrr.csv'
    result.sort_values(['first_timestamp', 'WAFER_ID']).to_csv(out_wafer, index=False)

    crosstab = pd.crosstab(result['code_group'], result['mrr_band_descriptive_only'])
    order_bands = [
        'MRR < 90 (descriptive)',
        '90 <= MRR < 135 (gap)',
        '135 <= MRR < 1000 (descriptive)',
        'MRR >= 1000 (review only)',
    ]
    crosstab = crosstab.reindex(columns=order_bands, fill_value=0).sort_index()
    crosstab.to_csv(audit / '03b_code_group_vs_mrr_band.csv')

    comparison_rows = []
    for groupname, part in result.groupby('code_group', sort=True):
        for sensor in SENSORS:
            active = part[f'{sensor}_nonzero_fraction']
            comparison_rows.append({
                'code_group': groupname,
                'sensor': sensor,
                'wafer_stage_n': int(len(part)),
                'median_nonzero_fraction': float(active.median()),
                'p25_nonzero_fraction': float(active.quantile(.25)),
                'p75_nonzero_fraction': float(active.quantile(.75)),
                'note': 'Nonzero fraction does not certify machine activity or sensor validity',
            })
    pd.DataFrame(comparison_rows).to_csv(audit / '03b_sensor_activity_by_group.csv', index=False)

    # Figure 1: MRR distribution by code group, with review >1000 out of view.
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    chartgroups = ['Codes 1-3', 'Codes 4-6', 'Mixed/other codes']
    values, names = [], []
    for groupname in chartgroups:
        v = result.loc[(result['code_group'] == groupname) & (result['AVG_REMOVAL_RATE'] < 1000), 'AVG_REMOVAL_RATE']
        if len(v):
            values.append(v.to_numpy())
            names.append(f'{groupname}\n(n={len(v)})')
    ax.boxplot(values, tick_labels=names, showfliers=True, patch_artist=False)
    ax.set_ylim(40, 180)
    ax.set_ylabel('Average removal rate (dataset scale)')
    ax.set_title('Stage A: MRR grouped by sensor MACHINE_DATA codes (TRAIN ONLY)')
    ax.grid(axis='y', alpha=.25)
    fig.text(.5, .02, 'Four MRR >= 1000 review records not displayed; no defect/specification limits are assumed.', ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .04, 1, 1])
    fig.savefig(figs / '03b_stageA_mrr_by_code_group.png', dpi=170)
    plt.close(fig)

    # Figure 2: overlay both modes against available record order.
    ordered = result.sort_values(['first_timestamp', 'WAFER_ID']).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11, 5.6))
    for name, label in [('Codes 1-3','Codes 1–3'), ('Codes 4-6','Codes 4–6'), ('Mixed/other codes','Mixed/other')]:
        subset = ordered.loc[(ordered['code_group'] == name) & (ordered['AVG_REMOVAL_RATE'] < 1000)]
        if not subset.empty:
            ax.scatter(subset.index, subset['AVG_REMOVAL_RATE'], s=10, alpha=.60, label=f'{label} (n={len(subset)})')
    ax.set_ylim(40,180)
    ax.set_xlabel('Record order by earliest sensor timestamp (not confirmed calendar time)')
    ax.set_ylabel('Average removal rate (dataset scale)')
    ax.set_title('Stage A: MRR by machine-code group and recorded order (TRAIN ONLY)')
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(figs / '03b_stageA_record_order_by_code_group.png', dpi=170)
    plt.close(fig)

    summary = {
        'scope': 'Training Stage A only',
        'stage_A_wafer_stage_count': int(len(result)),
        'stage_A_sensor_event_count': int(len(sensors)),
        'counts_by_code_group': {str(k):int(v) for k,v in result['code_group'].value_counts().items()},
        'counts_by_MRR_descriptive_band': {str(k):int(v) for k,v in result['mrr_band_descriptive_only'].value_counts().items()},
        'cross_tab_by_code_group_and_mrr_band': {str(idx):{str(k):int(v) for k,v in row.items()} for idx,row in crosstab.to_dict(orient='index').items()},
        'notes': [
            'MACHINE_DATA codes are anonymous codes, not confirmed physical tool or recipe identities.',
            'All 90/135/1000 thresholds only describe the Training sample; do not use as quality specs, SPC limits, or predictive labels.',
            'MRR>1000 is a review flag, not verified measurement error or failure.',
            'Zero sensor readings may indicate off-periods, inactive channels, missingness encoded as zero, or another unknown mechanism.',
            'No Test/Validation data used to choose groups, inspect MRR, or fit thresholds.',
            'Do not use MRR-derived group labels as input features in an MRR prediction model.',
        ],
    }
    with (audit/'03b_investigation_summary.json').open('w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print('PASS: Stage A training quality and sensor tables reconciled')
    print(f"Training Stage A wafer-stages: {len(result):,}; sensor events: {len(sensors):,}")
    print('\nMRR bands by MACHINE_DATA code group (EDA only):')
    print(crosstab.to_string())
    print(f'\nSaved 4 reports and 2 figures to: {audit}')
    print('No raw/clean source rows were modified. No SPC/specification limits were computed.')


if __name__ == '__main__':
    main()
