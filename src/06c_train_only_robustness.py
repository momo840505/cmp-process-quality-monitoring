"""Step 06C: TRAIN-only sensitivity checks for the fixed Step 06B VM model.

Reads 06A feature/catalog and 03B timestamp inventory. Never uses Valid/Test
rows for fitting, scoring, threshold design, ablation, or selection.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

Y = 'AVG_REMOVAL_RATE'
GROUPS = {'Codes 1-3': (1, 2, 3), 'Codes 4-6': (4, 5, 6)}
RNG_SEEDS = (19, 71)
FOLDS = 3


def input_checks(features: pd.DataFrame, catalog: pd.DataFrame, timeline: pd.DataFrame):
    required = {'WAFER_ID','vm_partition','code_family','code_signature',Y,'mrr_review_gt_1000'}
    if not required <= set(features.columns):
        raise ValueError(f'Missing feature-table fields: {required - set(features.columns)}')
    if not features.WAFER_ID.is_unique or not timeline.WAFER_ID.is_unique:
        raise ValueError('Duplicate WAFER_ID in features/timestamp inventory')
    if not set(features.WAFER_ID).issubset(set(timeline.WAFER_ID)):
        raise ValueError('Missing timeline IDs')
    if 'first_timestamp' not in timeline.columns:
        raise ValueError('Missing first_timestamp')
    feature_names = catalog['feature_name'].astype(str).tolist()
    if len(feature_names)!=133 or len(set(feature_names))!=133:
        raise ValueError('Unexpected feature catalog length/duplicates')
    if not set(feature_names)<=set(features.columns):
        raise ValueError('Missing catalog features')
    forbidden = {Y,'WAFER_ID','vm_partition','code_family','mrr_review_gt_1000','first_timestamp','last_timestamp', 'has_recorded_time_conflict'}
    if forbidden & set(feature_names):
        raise ValueError('Potential target/ID/metadata leakage')
    counts = features.vm_partition.value_counts().to_dict()
    if counts!={'train':817,'valid':175,'test':174}:
        raise ValueError('Unexpected 06A partitions: '+str(counts))
    return feature_names


def feature_sets(feature_names: list[str], codes: tuple[int, ...]) -> dict[str, list[str]]:
    group = [c for c in feature_names if c.startswith(tuple(f'code{i}__' for i in codes))]
    uses = [c for c in group if '__USAGE_OF_' in c]
    no_uses = [c for c in group if '__USAGE_OF_' not in c]
    if len(group)!=66 or len(uses)!=12 or len(no_uses)!=54:
        raise ValueError(f'Feature set mismatch {len(group)} / {len(uses)} / {len(no_uses)}')
    return {'full_66':group, 'no_usage_54':no_uses, 'usage_only_12':uses}


def hgb_fixed() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(loss='absolute_error', learning_rate=0.05,
        max_iter=140, max_leaf_nodes=12, min_samples_leaf=15,
        l2_regularization=5.0, random_state=42, early_stopping=False)


def metrics(y,p,flagged):
    y=np.asarray(y,float); p=np.asarray(p,float); flagged=np.asarray(flagged,bool)
    if len(y)==0 or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('Non-finite/empty metrics')
    def one(a,b):
        return {'n':len(a),'mae':float(mean_absolute_error(a,b)),
                'rmse':float(np.sqrt(mean_squared_error(a,b))),
                'r2':float(r2_score(a,b)) if len(a)>=2 else None}
    all_m=one(y,p)
    ordinary=one(y[~flagged],p[~flagged]) if (~flagged).sum()>=2 else dict(n=int((~flagged).sum()),mae=None,rmse=None,r2=None)
    return {**{f'all_{k}':v for k,v in all_m.items()},**{f'ordinary_{k}':v for k,v in ordinary.items()},'review_rows':int(flagged.sum())}


def measure_fold(train, hold, cols, variant):
    ytrain=train[Y].to_numpy(float); yhold=hold[Y].to_numpy(float)
    if variant=='median_baseline':
        pred=np.repeat(np.median(ytrain),len(hold))
    else:
        model=hgb_fixed()
        model.fit(train[cols].astype(float),ytrain)
        pred=model.predict(hold[cols].astype(float))
    return metrics(yhold,pred,hold['mrr_review_gt_1000'].astype(bool).to_numpy())


def evaluate_train_only(features, catalog, timeline):
    names=input_checks(features,catalog,timeline)
    # IMPORTANT: take train rows BEFORE touching targets, fitting, or making rankings.
    train_only=features.loc[features.vm_partition.eq('train')].copy()
    times=timeline[['WAFER_ID','first_timestamp']].copy()
    train_only=train_only.merge(times,how='left',on='WAFER_ID',validate='one_to_one')
    if train_only['first_timestamp'].isna().any():
        raise ValueError('Missing timestamp in train rows')
    rows=[]
    per_group={}
    for g,codes in GROUPS.items():
        d=train_only.loc[train_only.code_family.eq(g)].copy().sort_values('WAFER_ID').reset_index(drop=True)
        if not d.code_signature.astype(str).eq(','.join(map(str,codes))).all():
            raise ValueError('Signature mismatch')
        per_group[g]=len(d)
        fsets=feature_sets(names,codes)
        if len(d)<90:
            raise ValueError('Group too small')
        for seed in RNG_SEEDS:
            kf=KFold(n_splits=FOLDS,shuffle=True,random_state=seed)
            for fold,(ix_train,ix_hold) in enumerate(kf.split(d),1):
                a=d.iloc[ix_train]; b=d.iloc[ix_hold]
                if set(a.WAFER_ID)&set(b.WAFER_ID):
                    raise AssertionError('ID overlap')
                for variant in ['median_baseline',*fsets]:
                    m=measure_fold(a,b, fsets.get(variant,[]),variant)
                    rows.append({'group':g,'design':'train_only_3fold_repeated','seed':seed,'fold':fold,
                                 'variant':variant,'train_n':len(a),'holdout_n':len(b),**m})
        # Supplemental chronological-like stress test -- timestamp STARTS, not verified run completion.
        ordered=d.sort_values(['first_timestamp','WAFER_ID']).reset_index(drop=True)
        boundary=int(len(ordered)*0.75)
        a=ordered.iloc[:boundary];b=ordered.iloc[boundary:]
        if len(a)<100 or len(b)<40:
            raise ValueError('Too few chronological-like rows')
        for variant in ['median_baseline',*fsets]:
            m=measure_fold(a,b,fsets.get(variant,[]),variant)
            rows.append({'group':g,'design':'train_only_timestamp_start_75_25','seed':np.nan,'fold':np.nan,
                         'variant':variant,'train_n':len(a),'holdout_n':len(b),**m})
    d=pd.DataFrame(rows)
    # Expose average fold metrics; averaging RMSE is descriptive, NOT pooled RMSE.
    cv=(d[d.design.eq('train_only_3fold_repeated')]
        .groupby(['group','variant'],as_index=False).agg(folds=('all_mae','count'),
        all_mae_mean=('all_mae','mean'),all_mae_sd=('all_mae','std'),
        all_rmse_mean=('all_rmse','mean'),ordinary_mae_mean=('ordinary_mae','mean'),
        ordinary_rmse_mean=('ordinary_rmse','mean'),folds_with_review=('review_rows',lambda s:int((s>0).sum()))))
    assert set(d['design'].unique())=={'train_only_3fold_repeated','train_only_timestamp_start_75_25'}
    assert len(d)==(2*2*3*4+2*4)
    return d,cv,per_group


def plot_robustness(cv, time, figs):
    figs.mkdir(parents=True,exist_ok=True)
    for g in GROUPS:
        c=cv[cv.group.eq(g)].set_index('variant')
        t=time[time.group.eq(g)].set_index('variant')
        labels=['Baseline','HGB full','HGB no usage','HGB usage only']
        variants=['median_baseline','full_66','no_usage_54','usage_only_12']
        v1=c.loc[variants,'ordinary_mae_mean'].to_numpy()
        v2=t.loc[variants,'ordinary_mae'].to_numpy()
        x=np.arange(len(labels))
        fig,ax=plt.subplots(figsize=(9.6,5.6))
        ax.bar(x-.18,v1,width=.36,label='Repeated 3-fold: mean ordinary MAE')
        ax.bar(x+.18,v2,width=.36,label='Timestamp-start holdout: ordinary MAE')
        ax.set_xticks(x,labels,rotation=13)
        ax.set_ylabel('MAE (ordinary-scale supplemental view; lower is better)')
        ax.set_title(f'Step 06C TRAIN-only robustness / {g}')
        ax.legend()
        fig.text(.5,.01,'Training-only folds; ordinary values are NOT primary metrics. All >1000 observations retained in fits and primary scores.',ha='center',fontsize=8)
        fig.tight_layout(rect=[0,.055,1,1]);fig.savefig(figs/f'06c_{"codes1_3" if g=="Codes 1-3" else "codes4_6"}_robustness.png',dpi=140);plt.close(fig)


def execute(base:Path):
    out=base/'outputs'/'analysis'
    features=pd.read_csv(out/'06a_stageA_vm_features.csv',low_memory=False)
    catalog=pd.read_csv(out/'06a_feature_catalog.csv')
    timeline=pd.read_csv(out/'03b_stageA_wafer_code_mrr.csv',usecols=['WAFER_ID','first_timestamp'])
    folds,cv,counts=evaluate_train_only(features,catalog,timeline)
    time=folds[folds.design.eq('train_only_timestamp_start_75_25')].copy()
    folds.to_csv(out/'06c_train_only_folds.csv',index=False,encoding='utf-8-sig')
    cv.to_csv(out/'06c_repeated_cv_summary.csv',index=False,encoding='utf-8-sig')
    time.to_csv(out/'06c_timestamp_start_holdout.csv',index=False,encoding='utf-8-sig')
    plot_robustness(cv,time,out/'figures')
    notes={'scope':'Original competition TRAIN / Stage A / INTERNAL train partition ONLY; internal valid/test and competition test/validation never enter any Step06C fit or score',
      'groups':counts,'repeated_cv':{'folds':3,'repeat_seeds':list(RNG_SEEDS),'holdout_entity':'WAFER_ID (unique per row)','fixed_model_hyperparameters':'exact Step06B HGB_absolute_error'},
      'timestamp_stress':'Within INTERNAL train, sort first_timestamp; fit earliest 75%, assess latest 25%; NOT verified chronological factory run, interval overlaps persist',
      'variants':{'median_baseline':'Training-partition subset median only', 'full_66':'All 66 per-group process features', 'no_usage_54':'Exclude 12 USAGE_OF_* fields; remaining 54 sensor/activity/presence fields', 'usage_only_12':'Only 12 USAGE_OF_* feature summaries; consumable/time-proxy stress check'},
      'important_limits':['Different from published external competition Test','MRR>1000 retained in all model fits and primary all-row metrics; ordinary view explicitly supplemental','Repeated-fold mean RMSE is descriptive, not equivalent to pooled RMSE','Original VM splits were random by unique wafer, may not measure future drift','Time proxies such as consumable usage may create optimistic apparent performance across nearby conditions','No model selection or threshold adjustment using the internal valid/test partitions','If non-consumable model collapses while usage-only performs well, treat model as relying heavily on possible machine age/drift proxy','No evidence of actual product defect labels, physical process adjustments or confirmed root cause']}
    (out/'06c_robustness_notes.json').write_text(json.dumps(notes,ensure_ascii=False,indent=2),encoding='utf-8')
    print('PASS: 816 internal TRAIN rows across two complete code groups; 1 incomplete record excluded from fixed-group models')
    print('Repeated 3-fold x 2 seeds: 48 fold/variant results; timestamp-start 75/25: 8 results')
    print('\nRepeated CV ordinary MAE (supplemental, complete metrics in CSV):')
    print(cv[['group','variant','ordinary_mae_mean','all_mae_mean']].to_string(index=False,float_format=lambda x:f'{x:.3f}'))
    print('\nTimestamp-start holdout ordinary MAE:')
    print(time[['group','variant','holdout_n','ordinary_mae','all_mae']].to_string(index=False,float_format=lambda x:f'{x:.3f}'))
    print('\nNo VALID/TEST rows used. Models not tuned; source data unchanged. Saved 4 reports and 2 figures.')
    return folds,cv,time

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args()
    execute(args.project)
