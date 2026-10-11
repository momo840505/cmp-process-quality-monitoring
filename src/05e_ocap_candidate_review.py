"""Step 05E: read-only exploratory FDC candidate review (Training Stage A).

Do not interpret sensor deviations as faults or MRR within q05-q95 as product pass.
No thresholds or candidates are recalibrated here.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EVIDENCE_SIGNALS = {
    'chamber_pressure': 'Chamber pressure',
    'retainer_pressure': 'Retainer pressure',
    'slurry_flow_A': 'Slurry flow A',
    'slurry_flow_C': 'Slurry flow C',
    'wafer_rotation': 'Wafer rotation',
    'stage_rotation': 'Stage rotation',
}
KEY = ['WAFER_ID', 'STAGE', 'MACHINE_DATA']


def require(condition: bool, explanation: str):
    if not condition:
        raise ValueError(explanation)


def parse_evidence(raw: str) -> list[dict]:
    data = json.loads(raw)
    require(isinstance(data, list) and len(data) >= 2, 'Candidate evidence must be a list of >=2 signals.')
    family_seen = set()
    for d in data:
        require(all(x in d for x in ('signal','family','observed_median_nonzero','reference_q01','reference_q99','direction')),
                'Missing signal evidence fields.')
        require(d['signal'] in EVIDENCE_SIGNALS, 'Unknown source signal (do not silently accept).')
        a, b, v = (float(d[k]) for k in ('reference_q01','reference_q99','observed_median_nonzero'))
        require(np.isfinite([a,b,v]).all() and a < b, 'Invalid frozen reference range or observed value.')
        require((d['direction']=='below' and v < a) or (d['direction']=='above' and v > b),
                'Frozen trigger direction/value inconsistent.')
        require(d['family'] in ('pressure','slurry','motion'), 'Unknown sensor family.')
        family_seen.add(d['family'])
    require(len(family_seen)>=2, 'At least two sensor families must be represented.')
    return data


def compare_profile(e: dict, code: int, profiles: pd.DataFrame):
    matches = profiles[(profiles.MACHINE_DATA==code)&(profiles.signal==e['signal'])]
    require(len(matches)==1 and bool(matches.iloc[0]['reference_usable']),
            f'Missing eligible frozen profile for code {code} / {e["signal"]}')
    z=matches.iloc[0]
    require(np.isclose(float(z.q01),float(e['reference_q01']),rtol=1e-6,atol=1e-7)
            and np.isclose(float(z.q99),float(e['reference_q99']),rtol=1e-6,atol=1e-7),
            f'Frozen 05C q01/q99 mismatch for code {code} / {e["signal"]}')
    require(str(z.signal_family)==str(e['family']), 'Signal family differs from frozen 05C profile.')


def load_and_validate(folder: Path):
    files=['05c_later_magnitude_candidates.csv','05c_magnitude_reference_profiles.csv',
           '05d_candidate_case_review.csv','05d_later_wafer_quality_link.csv',
           '05d_group_comparison.csv']
    for fn in files:
        require((folder/fn).exists(), f'Missing {folder/fn}; complete Step 05C and 05D first.')
    chosen=pd.read_csv(folder/files[0]); profiles=pd.read_csv(folder/files[1]);
    cases=pd.read_csv(folder/files[2]); later=pd.read_csv(folder/files[3]); comparison=pd.read_csv(folder/files[4])
    require(len(cases)==11 and len(later)==468, 'This version expects verified Step05D 11 candidates/468 later wafers.')
    require(not cases.duplicated(KEY).any(), 'Duplicate candidate wafer-code key.')
    require(not later.duplicated(['WAFER_ID','STAGE']).any(), 'Duplicate wafer-stage later quality key.')
    a=set(map(tuple,cases[KEY].itertuples(index=False,name=None)))
    b=set(map(tuple,chosen[KEY].itertuples(index=False,name=None)))
    require(a==b and len(a)==11, 'Frozen 05C candidate membership mismatches 05D; stop before review.')
    require((cases['05c__family_count']>=2).all(), 'A frozen candidate has fewer than 2 families.')
    require((~cases['05c__requires_data_quality_review']).all(), 'QA-flagged candidate needs manual review before this casebook.')
    require(cases.WAFER_ID.nunique()==11, '11 candidate IDs are not unique.')
    joined=cases.merge(later[['WAFER_ID','STAGE','code_group','AVG_REMOVAL_RATE','outside_reference_q05_q95_descriptive']],
                       on=['WAFER_ID','STAGE'],how='left',validate='one_to_one',suffixes=('','_later'))
    require(joined['AVG_REMOVAL_RATE_later'].notna().all(), 'Candidate MRR not found in later outcomes.')
    require(np.allclose(joined['AVG_REMOVAL_RATE'],joined['AVG_REMOVAL_RATE_later']), 'MRR differs across 05D files.')
    require((joined.code_group==joined.code_group_later).all(), 'Group mismatch.')
    require((joined['outside_reference_q05_q95_descriptive']==joined['outside_reference_q05_q95_descriptive_later']).all(), 'MRR reference outcome mismatch.')
    require(not joined['outside_reference_q05_q95_descriptive'].any(), 'Unexpected outside-MRR candidate; inspect rather than hide.')
    require(comparison.later_wafers.sum()==468, 'Comparison count mismatch.')
    evidence=[]
    rows=[]
    checklist=[]
    for _,r in cases.sort_values(['code_group','MACHINE_DATA','WAFER_ID']).iterrows():
        k={'WAFER_ID':int(r.WAFER_ID),'STAGE':str(r.STAGE),'MACHINE_DATA':int(r.MACHINE_DATA),'code_group':r.code_group}
        ee=parse_evidence(r.frozen_sensor_evidence_json)
        for e in ee:
            compare_profile(e,int(r.MACHINE_DATA),profiles)
            a,b,x=[float(e[y]) for y in ['reference_q01','reference_q99','observed_median_nonzero']]
            evidence.append({**k,'signal':e['signal'],'family':e['family'],'direction':e['direction'],
                             'observed_median_nonzero':x,'reference_q01':a,'reference_q99':b,
                             'relative_to_reference_width':(x-a)/(b-a)})
        families=sorted({e['family'] for e in ee})
        require(len(families)==int(r['05c__family_count']), 'Frozen family count differs from evidence.')
        readable='; '.join(f"{EVIDENCE_SIGNALS[e['signal']]} {e['direction']} Q01/Q99 "
                           f"({float(e['observed_median_nonzero']):.2f} vs {float(e['reference_q01']):.2f}–{float(e['reference_q99']):.2f})"
                           for e in ee)
        rows.append({**k,'frozen_signal_family_count':len(families),'triggered_families':'/'.join(families),
                     'triggered_sensor_count':len(ee),'sensor_evidence':readable,
                     'measured_MRR':float(r.AVG_REMOVAL_RATE),
                     'reference_MRR_median':float(r.reference_median),
                     'reference_MRR_q05':float(r.reference_q05),'reference_MRR_q95':float(r.reference_q95),
                     'outside_reference_MRR_q05_q95_descriptive':bool(r.outside_reference_q05_q95_descriptive),
                     'has_recorded_timestamp_conflict':bool(r.has_timestamp_conflict),
                     'review_status':'Unverified sensor-pattern deviation; requires engineer assessment',
                     'do_not_infer':'Not verified defect; not verified tool fault; not proven root cause'})
        checklist.append({**k,'step_1':'Verify wafer-code assignment, timestamps, sensor continuity and measurement validity',
                          'step_2':'Review pressure trace/settings and pressure sensors for this code (if available)',
                          'step_3':'Review slurry flow trace, supplies and calibration for this code (if available)',
                          'step_4':'Compare earlier wafers and equipment/consumable log only if genuine run history is available',
                          'step_5':'Compare metrology with formal factory specifications ONLY when provided; do not infer pass/fail from dataset q05/q95',
                          'step_6':'Record triage outcome, engineer disposition and any authorised OCAP according to site SOP',
                          'action_status':'Suggested triage checklist only; no stop/hold/parameter change was executed'})
    return cases, later, pd.DataFrame(rows),pd.DataFrame(evidence),pd.DataFrame(checklist)


def draw_matrix(cases_df:pd.DataFrame,evidence_df:pd.DataFrame,path:Path):
    case_keys=[f"{int(r.WAFER_ID)} / C{int(r.MACHINE_DATA)}" for _,r in cases_df.iterrows()]
    signals=list(EVIDENCE_SIGNALS)
    lookup={(f"{int(r.WAFER_ID)} / C{int(r.MACHINE_DATA)}",r.signal):r.direction for _,r in evidence_df.iterrows()}
    arr=np.zeros((len(case_keys),len(signals)))
    for i,w in enumerate(case_keys):
        for j,s in enumerate(signals):
            arr[i,j]=(-1 if lookup.get((w,s))=='below' else 1 if lookup.get((w,s))=='above' else 0)
    from matplotlib.colors import ListedColormap,BoundaryNorm
    fig,ax=plt.subplots(figsize=(11,6.3))
    cmap=ListedColormap(['#457b9d','#f0f0f0','#db804b']);norm=BoundaryNorm([-1.5,-0.5,0.5,1.5],cmap.N)
    ax.imshow(arr,cmap=cmap,norm=norm,aspect='auto')
    ax.set_xticks(range(len(signals)),[EVIDENCE_SIGNALS[s] for s in signals],rotation=28,ha='right')
    ax.set_yticks(range(len(case_keys)),case_keys)
    ax.set_xlabel('Frozen Step 05C signal triggers (each code has its own Q01–Q99)')
    ax.set_ylabel('Wafer ID / anonymous machine code')
    ax.set_title('Step 05E | Frozen FDC review candidates (Training Stage A)')
    for i in range(len(case_keys)):
        for j in range(len(signals)):
            ax.text(j,i,('Below' if arr[i,j]==-1 else 'Above' if arr[i,j]==1 else '—'),
                    ha='center',va='center',fontsize=7.5,color='white' if arr[i,j]!=0 else '#555555')
    fig.text(0.02,0.01,'Below/Above = signal median outside frozen exploratory reference; NOT physical fault or product defect.',fontsize=9)
    fig.tight_layout(rect=[0,0.045,1,1]);fig.savefig(path,dpi=160,bbox_inches='tight');plt.close(fig)


def draw_case(case_row:pd.Series,evidence_df:pd.DataFrame,path:Path):
    e=evidence_df[(evidence_df.WAFER_ID==int(case_row.WAFER_ID)) & (evidence_df.MACHINE_DATA==int(case_row.MACHINE_DATA))]
    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(10,6.1),gridspec_kw={'height_ratios':[2,1]})
    ax1.axvspan(0,1,color='0.88',alpha=.6,label='Code reference Q01 to Q99 (scaled)')
    y=list(range(len(e)))
    z=e.relative_to_reference_width.to_numpy(dtype=float)
    ax1.scatter(z,y,s=65,marker='D')
    for j,(_,r) in enumerate(e.iterrows()):
        ax1.text(z[j],j+0.15, f"{r.observed_median_nonzero:.2f}",ha='center',fontsize=9)
    ax1.axvline(0,color='0.4',linestyle='--');ax1.axvline(1,color='0.4',linestyle='--')
    ax1.set_yticks(y,[EVIDENCE_SIGNALS[x] for x in e.signal])
    ax1.invert_yaxis()
    ax1.set_xlabel('Scaled within each sensor: (observed - Q01) / (Q99 - Q01)')
    ax1.set_title(f'Case {int(case_row.WAFER_ID)} / Code {int(case_row.MACHINE_DATA)}: sensor evidence')
    ax1.legend(loc='upper right',fontsize=8)
    lo,hi,med,v=(float(case_row[x]) for x in ['reference_MRR_q05','reference_MRR_q95','reference_MRR_median','measured_MRR'])
    ax2.axvspan(lo,hi,color='0.88',alpha=.7,label='Reference MRR Q05–Q95, descriptive')
    ax2.axvline(med,color='0.25',linestyle='--',label=f'Reference median {med:.2f}')
    ax2.scatter([v],[0],marker='x',s=100,label=f'Observed MRR {v:.2f}')
    ax2.set_ylim(-0.5,.5);ax2.set_yticks([]);ax2.set_xlabel('MRR (published dataset scale)');ax2.legend(loc='upper center',fontsize=8,ncol=3)
    fig.text(0.03,0.01,'Sensor trigger + MRR inside reference interval does NOT prove normal quality or root cause.',fontsize=9)
    fig.tight_layout(rect=[0,0.045,1,1]);fig.savefig(path,dpi=160,bbox_inches='tight');plt.close(fig)


def run(project_root:Path) -> dict:
    data_dir=project_root/'outputs'/'analysis';fig_dir=data_dir/'figures'
    cases,later,casebook,evidence,checklist=load_and_validate(data_dir)
    fig_dir.mkdir(parents=True,exist_ok=True)
    casebook.to_csv(data_dir/'05e_candidate_casebook.csv',index=False,encoding='utf-8-sig')
    evidence.to_csv(data_dir/'05e_signal_evidence_long.csv',index=False,encoding='utf-8-sig')
    checklist.to_csv(data_dir/'05e_ocap_triage_checklist.csv',index=False,encoding='utf-8-sig')
    c=Counter(casebook.MACHINE_DATA.astype(int).tolist())
    summary={
      'scope':'Training Stage A; read-only frozen Step05C candidates + Step05D observed MRR',
      'n_later_unique_wafers':int(len(later)),
      'frozen_candidate_wafer_codes':int(len(casebook)),
      'frozen_candidate_unique_wafers':int(casebook.WAFER_ID.nunique()),
      'candidate_codes':{str(k):int(v) for k,v in sorted(c.items())},
      'candidate_families':'Pressure + slurry, no motion family triggers in these 11 candidates',
      'frozen_triggered_signal_rows':int(len(evidence)),
      'candidate_MRR_outside_reference_q05_q95':int(casebook.outside_reference_MRR_q05_q95_descriptive.sum()),
      'noncandidate_MRR_outside_reference_q05_q95':int(later[(~later.sensor_candidate) & later.outside_reference_q05_q95_descriptive].shape[0]),
      'quality_label_used_to_select_candidates':False,
      'threshold_recalibration':False,
      'limits':['No actual fault labels or process USL/LSL',
                'Sensor reference Q01/Q99 and MRR Q05/Q95 are descriptive, not validated OCAP limits',
                'Proposed OCAP is a human review checklist, not an actual corrective-action execution',
                'Anonymous codes do not identify physical tools or recipes',
                'Candidate sample size 11 is too small for reliable defect or root-cause estimates'],
      'example_wafer_id':33494120,
      'input_verified_against':'05C candidate list, 05C reference profiles, 05D case + 05D later results'
    }
    require(summary['candidate_MRR_outside_reference_q05_q95']==0,'Unexpected candidate label distribution')
    require(summary['noncandidate_MRR_outside_reference_q05_q95']==77,'Unexpected comparison counts')
    (data_dir/'05e_review_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    draw_matrix(casebook,evidence,fig_dir/'05e_frozen_trigger_matrix.png')
    sample=casebook[casebook.WAFER_ID==summary['example_wafer_id']]
    require(len(sample)==1,'Fixed example wafer not found')
    draw_case(sample.iloc[0],evidence,fig_dir/'05e_example_wafer_evidence.png')
    print(f"PASS: {len(later)} later Training wafers; {len(casebook)} frozen candidates; {len(evidence)} sensor deviations")
    print('Sensor candidate MRR outside reference Q05-Q95: 0; non-candidate: 77 (DESCRIPTIVE, not pass/fail)')
    print(f'Candidate codes: {dict(sorted(c.items()))}; unchanged 05C membership, no sensor thresholds recalibrated')
    print('Saved 4 reports and 2 charts to:',data_dir)
    print('OCAP suggestions are manual review checklist ONLY. Source unchanged. Test/Validation not read.')
    return summary


def main():
    p=argparse.ArgumentParser(description='Step05E: FDC candidate evidence + suggested OCAP review (non-operational)')
    p.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[1])
    args=p.parse_args()
    run(args.project_root)


if __name__=='__main__':main()
