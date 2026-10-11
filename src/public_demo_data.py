"""Validate aggregate-only public showcase data; never ingest PHM raw/wafer rows."""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd


def load_public_data(root: Path):
    root=Path(root)
    summary=json.loads((root/'public_data/summary.json').read_text(encoding='utf-8'))
    sim=pd.read_csv(root/'public_data/simulated_runs.csv')
    validate_public(summary,sim)
    return summary,sim


def validate_public(s,df):
    if s['schema_version'] != 1 or s['data_scope']['official_test_used'] is not False:
        raise ValueError('Invalid demo metadata or external test claim')
    if set(s['spc']) != {'Codes 1-3','Codes 4-6'}:
        raise ValueError('Missing SPC code groups')
    if len(df)!=120 or df.run.tolist()!=list(range(1,121)):
        raise ValueError('Simulated run sequence invalid')
    required={'run','open_mrr','ewma_mrr','open_setting','ewma_setting'}
    if set(df.columns)!=required or df.isna().any().any():
        raise ValueError('Simulated data schema invalid')
    for col in ('open_mrr','ewma_mrr','open_setting','ewma_setting'):
        if not np.isfinite(df[col]).all():raise ValueError('Nonfinite simulation value')
    r=s['r2r']
    for name,col in [('mae_open','open_mrr'),('mae_ewma','ewma_mrr')]:
        actual=float(np.abs(df[col]-r['target']).mean())
        if not np.isclose(actual,r[name],atol=1e-4):
            raise ValueError('R2R published aggregate disagrees with simulation')
    pct=100*(1-r['mae_ewma']/r['mae_open'])
    if not np.isclose(pct,r['reduction_pct'],atol=0.01):raise ValueError('R2R improvement inconsistent')
    if not np.allclose(df['open_setting'],50):raise ValueError('Open loop setpoint changed')
    if not df.ewma_setting.between(42,58).all():raise ValueError('EWMA setting out of assumed range')
    if len(s['fdc']['by_code'])!=6 or sum(x['two_or_more'] for x in s['fdc']['by_code'])!=11:
        raise ValueError('FDC counts inconsistent')
    if s['fdc']['sensor_trigger_records']!=sum(s['fdc']['signal_type_counts'].values()):
        raise ValueError('FDC trigger evidence count inconsistent')
    for code in s['spc']:
        st=s['spc'][code]
        if not (0<=st['i_chart_flags']<=st['monitor_n'] and 0<=st['ar1_flags']<=st['monitor_n']):
            raise ValueError('SPC group invalid')
        v=s['vm'][code]
        if not np.isclose(v['improvement_pct'],100*(1-v['model_mae']/v['baseline_mae']),atol=1e-6):
            raise ValueError('VM gain inconsistent')
