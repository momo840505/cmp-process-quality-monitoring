"""Step 05C: descriptive, code-specific positive-signal magnitude screening.

Usage (project root):
    .\\.venv\\Scripts\\python.exe .\\src\\05c_sensor_magnitude_screen.py

Input: outputs/analysis/05b_wafer_code_activity_screening.csv
No MRR / Test / Validation is accessed. No raw files are changed.
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

SENSORS = {
    'chamber_pressure': ('PRESSURIZED_CHAMBER_PRESSURE', 'pressure'),
    'retainer_pressure': ('RETAINER_RING_PRESSURE', 'pressure'),
    'slurry_flow_A': ('SLURRY_FLOW_LINE_A', 'slurry'),
    'slurry_flow_C': ('SLURRY_FLOW_LINE_C', 'slurry'),
    'wafer_rotation': ('WAFER_ROTATION', 'motion'),
    'stage_rotation': ('STAGE_ROTATION', 'motion'),
}
REF_MIN = 60             # minimum eligible wafer-channel reference points
DISTINCT_MIN = 10        # avoid nearly constant discrete profiles
MIN_NONZERO_COUNT = 10   # >=10 nonzero events per wafer-channel
MIN_NONZERO_FRACTION = 0.10
Q_LOW, Q_HIGH = 0.01, 0.99
# A frozen diagnostic setting, not an engineering/fault threshold.


def check_required(df: pd.DataFrame) -> None:
    required = ['WAFER_ID','STAGE','MACHINE_DATA','code_group','events_in_code',
                'reference_or_later','complete_signature','has_timestamp_conflict',
                'eligible_for_reference','first_timestamp']
    for stem, _ in SENSORS.values():
        required.extend([f'{stem}__nonzero_fraction',f'{stem}__median_nonzero'])
    missing = sorted(set(required)-set(df))
    if missing:
        raise ValueError('Missing Step05B columns: '+', '.join(missing))
    if len(df)!=3496 or df.WAFER_ID.nunique()!=1166 or set(df.STAGE.astype(str))!={'A'}:
        raise ValueError('Unexpected Step05B input dimensions or Stage; do not silently proceed.')
    if df[['WAFER_ID','STAGE','MACHINE_DATA']].duplicated().any():
        raise ValueError('Duplicate wafer/stage/code keys in Step05B input.')
    if set(df.reference_or_later.unique())!={'reference','later'}:
        raise ValueError('Unexpected reference/later labels.')
    if not set(df.MACHINE_DATA.unique())==set(range(1,7)):
        raise ValueError('Unexpected MACHINE_DATA codes.')
    if df.groupby('WAFER_ID').reference_or_later.nunique().gt(1).any():
        raise ValueError('Wafer appears in reference and later at once.')


def build_profiles_and_screen(df: pd.DataFrame):
    profiles=[]
    # Copy all original Step05B fields; no source mutation.
    screened=df.copy()
    for label, (stem, family) in SENSORS.items():
        med=f'{stem}__median_nonzero'; frac=f'{stem}__nonzero_fraction'
        n_nonzero=screened.events_in_code.astype(float)*screened[frac].astype(float)
        enough=(n_nonzero >= MIN_NONZERO_COUNT-1e-8) & (screened[frac]>=MIN_NONZERO_FRACTION) & screened[med].notna()
        eligible_name=f'05c__{label}__sufficient_active_samples'
        flag_name=f'05c__{label}__outside_ref'
        usable_name=f'05c__{label}__reference_usable'
        screened[eligible_name]=enough
        screened[flag_name]=False
        screened[usable_name]=False
        for code, subset in screened.groupby('MACHINE_DATA', sort=True):
            ref=subset[(subset.reference_or_later=='reference') & subset.eligible_for_reference & subset[eligible_name]]
            v=ref[med].astype(float)
            distinct=int(v.nunique())
            q01=float(v.quantile(Q_LOW)) if len(v) else np.nan
            q50=float(v.median()) if len(v) else np.nan
            q99=float(v.quantile(Q_HIGH)) if len(v) else np.nan
            # a flat quantile interval is unsuitable for alarm-like screening
            relative_spread=(q99-q01)/max(1.,abs(q50)) if np.isfinite(q50) else np.nan
            usable=(len(v)>=REF_MIN and distinct>=DISTINCT_MIN
                    and np.isfinite(relative_spread) and relative_spread>0.01)
            profiles.append({'MACHINE_DATA':code,'code_group':subset.code_group.iloc[0],
                             'signal':label,'signal_family':family,
                             'source_median_nonzero':med,'reference_available':len(v),
                             'reference_unique_values':distinct,'q01':q01,'q50':q50,'q99':q99,
                             'relative_q99_q01_spread':relative_spread,
                             'reference_usable':bool(usable),
                             'reason_if_not_usable': '' if usable else 'too_few_active_reference_rows_or_nearly_constant'})
            idx=subset.index
            screened.loc[idx,usable_name]=bool(usable)
            if usable:
                subset_ok=idx[screened.loc[idx,eligible_name].to_numpy()]
                screened.loc[subset_ok,flag_name] = ((screened.loc[subset_ok,med]<q01)|
                                                       (screened.loc[subset_ok,med]>q99)).to_numpy()
        screened[flag_name]=screened[flag_name].astype(bool)
    profile_df=pd.DataFrame(profiles)
    # One family may contain two sensors. Count independent families, not sensor counts.
    for family in ['pressure','slurry','motion']:
        cols=[f'05c__{name}__outside_ref' for name,(_,fam) in SENSORS.items() if fam==family]
        screened[f'05c__{family}_family_flag']=screened[cols].any(axis=1)
    family_cols=[f'05c__{x}_family_flag' for x in ['pressure','slurry','motion']]
    screened['05c__family_count']=screened[family_cols].sum(axis=1).astype(int)
    screened['05c__later_any_magnitude_deviation']=(screened.reference_or_later=='later')&(screened['05c__family_count']>=1)
    screened['05c__later_two_family_candidate']=(screened.reference_or_later=='later')&(screened['05c__family_count']>=2)
    screened['05c__requires_data_quality_review']=(~screened.complete_signature)|screened.has_timestamp_conflict
    return profile_df,screened


def plot_summary(summary: pd.DataFrame,out: Path) -> None:
    fig,ax=plt.subplots(figsize=(10,4.6))
    x=np.arange(len(summary));width=.36
    ax.bar(x-width/2, summary['later_any_family'],width,label='At least 1 magnitude family outside reference')
    ax.bar(x+width/2, summary['later_two_or_more_families'],width,label='At least 2 families (review candidate)')
    ax.set_xticks(x,[f'Code {int(c)}' for c in summary.MACHINE_DATA]);ax.set_ylabel('Later wafer-code observations')
    ax.set_title('TRAIN Stage A: sensor magnitude deviations (screening, not faults)')
    ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(out,dpi=160);plt.close(fig)


def plot_profile(profiles: pd.DataFrame,out: Path) -> None:
    # Show eligibility not scaled magnitudes, since each sensor has different physical scale.
    p=profiles.pivot(index='signal',columns='MACHINE_DATA',values='reference_usable').astype(float)
    fig,ax=plt.subplots(figsize=(10,4.5))
    im=ax.imshow(p.values,vmin=0,vmax=1,cmap='Greys',aspect='auto')
    ax.set_xticks(np.arange(len(p.columns)),[f'Code {x}' for x in p.columns]);ax.set_yticks(np.arange(len(p.index)),p.index)
    for i in range(len(p.index)):
        for j in range(len(p.columns)):
            ax.text(j,i,'usable' if p.values[i,j] else 'skip',ha='center',va='center',
                    color='white' if p.values[i,j] else 'black',fontsize=9)
    ax.set_title('Reference eligibility by code and positive-signal magnitude')
    fig.tight_layout();fig.savefig(out,dpi=160);plt.close(fig)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',default='outputs/analysis/05b_wafer_code_activity_screening.csv')
    ap.add_argument('--output',default='outputs/analysis')
    args=ap.parse_args()
    inp=Path(args.input);out=Path(args.output);fig_dir=out/'figures'
    if not inp.exists():raise FileNotFoundError(f'Run Step05B first: {inp}')
    df=pd.read_csv(inp)
    check_required(df)
    profiles,screen=build_profiles_and_screen(df)
    assert len(screen)==len(df)
    assert set(screen.reference_or_later)=={'reference','later'}
    assert screen.loc[screen.reference_or_later=='reference','05c__later_any_magnitude_deviation'].sum()==0
    assert screen.loc[screen.reference_or_later=='reference','05c__later_two_family_candidate'].sum()==0
    summary=[]
    for code,sub in screen.groupby('MACHINE_DATA',sort=True):
        later=sub[sub.reference_or_later=='later']
        summary.append({'MACHINE_DATA':int(code),'code_group':sub.code_group.iloc[0],
                        'reference_rows':int((sub.reference_or_later=='reference').sum()),
                        'later_rows':len(later),
                        'usable_magnitude_features':int(profiles.query('MACHINE_DATA == @code').reference_usable.sum()),
                        'later_any_family':int(later['05c__later_any_magnitude_deviation'].sum()),
                        'later_two_or_more_families':int(later['05c__later_two_family_candidate'].sum()),
                        'later_two_family_with_QA_flag':int((later['05c__later_two_family_candidate']&later['05c__requires_data_quality_review']).sum())})
    summary=pd.DataFrame(summary)
    assert int(summary.later_rows.sum())==1404
    assert int(summary.reference_rows.sum())==2092
    out.mkdir(exist_ok=True,parents=True);fig_dir.mkdir(exist_ok=True,parents=True)
    profiles.to_csv(out/'05c_magnitude_reference_profiles.csv',index=False)
    screen.to_csv(out/'05c_wafer_code_magnitude_screening.csv',index=False)
    summary.to_csv(out/'05c_summary_by_code.csv',index=False)
    screen.loc[screen['05c__later_two_family_candidate']].to_csv(out/'05c_later_magnitude_candidates.csv',index=False)
    plot_summary(summary,fig_dir/'05c_later_magnitude_candidate_counts.png')
    plot_profile(profiles,fig_dir/'05c_magnitude_reference_eligibility.png')
    notes={
        'scope':'TRAIN Stage A wafer-code features from Step05B only; no MRR, Test, Validation or defect labels read',
        'units':'Original PHM numeric recording scale; anonymous code-conditioned distributions, not calibrated specifications',
        'reference_split':'Reuses Step05B within-group wafer-level 60/40 reference/later; never re-split rows',
        'positive_median_logic':'Use *__median_nonzero only when >=10 positive/nonzero recorded events and nonzero fraction >=0.10. No inference about zero meaning.',
        'reference_calibration':'Per MACHINE_DATA, only Step05B reference rows marked eligible_for_reference; q01 and q99 of eligible positive-signal medians.',
        'reference_eligibility':{'min_reference_rows':REF_MIN,'min_distinct_reference_values':DISTINCT_MIN,'min_relative_quantile_spread':.01,
                                 'min_nonzero_events_per_wafer_code':MIN_NONZERO_COUNT,'min_nonzero_fraction':MIN_NONZERO_FRACTION},
        'signal_families':{n:{'sensor':s,'family':f} for n,(s,f) in SENSORS.items()},
        'candidate':'At least two distinct families (pressure/slurry/motion) with median_nonzero outside 1st/99th percentile reference.',
        'multiple_comparison_warning':'Six signals across six codes create multiple opportunities to flag. No validated false alarm probability.',
        'selection_warning':'Reference is not proven healthy or stable, and q01/q99 are chosen illustrations, not calibrated process or fault alarms.',
        'visibility_warning':'Insufficient active records are NA/not-screened, not automatically normal. Screening eligibility must be retained.',
        'sequencing_warning':'Order is first available sensor timestamp, not verified true run-to-run chronology; wafer codes assessed separately.',
        'no_mutation':True,
        'counts':summary.to_dict(orient='records'),
    }
    (out/'05c_method_notes.json').write_text(json.dumps(notes,indent=2,ensure_ascii=False),encoding='utf-8')
    print('PASS: 3,496 TRAIN Stage A wafer-code rows; reused Step05B fixed reference/later membership; no outcome labels read.')
    for _,r in summary.iterrows():
        print(f"Code {int(r.MACHINE_DATA)}: usable magnitude features {int(r.usable_magnitude_features)}/6 | later {int(r.later_rows)} | any-family {int(r.later_any_family)} | 2+ family candidates {int(r.later_two_or_more_families)}")
    print(f'Saved 5 tables/notes and 2 figures to {out}')
    print('REFERENCE ONLY; NOT a certified fault alarm, quality label, or process specification. No source files modified.')

if __name__=='__main__':main()
