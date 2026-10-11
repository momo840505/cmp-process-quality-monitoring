"""Read-only validation utilities for CMP Process QE dashboard.

All input files are PRECOMPUTED results. This module never retrains models,
re-runs the final test, or executes the R2R controller.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DashboardSources:
    spc: pd.DataFrame
    spc_summary: pd.DataFrame
    ar1: pd.DataFrame
    fdc_cases: pd.DataFrame
    fdc_signals: pd.DataFrame
    fdc_summary: pd.DataFrame
    vm_metrics: pd.DataFrame
    vm_predictions: pd.DataFrame
    open_runs: pd.DataFrame
    closed_runs: pd.DataFrame
    actions: pd.DataFrame
    r2r_comparison: pd.DataFrame
    r2r_manifest: dict
    vm_manifest: dict


FILES = {
    'spc': 'outputs/analysis/04a_training_chart_points.csv',
    'spc_summary': 'outputs/analysis/04a_provisional_chart_summary.csv',
    'ar1': 'outputs/analysis/04b_reference_only_AR1_summary.csv',
    'fdc_cases': 'outputs/analysis/05e_candidate_casebook.csv',
    'fdc_signals': 'outputs/analysis/05e_signal_evidence_long.csv',
    'fdc_summary': 'outputs/analysis/05c_summary_by_code.csv',
    'vm_metrics': 'outputs/analysis/06d_internal_test_metrics.csv',
    'vm_predictions': 'outputs/analysis/06d_internal_test_predictions.csv',
    'open_runs': 'outputs/simulation/07a_open_loop_simulated_runs.csv',
    'closed_runs': 'outputs/simulation/07c_closed_loop_simulated_runs.csv',
    'actions': 'outputs/simulation/07c_feedback_actions.csv',
    'r2r_comparison': 'outputs/simulation/07c_open_vs_closed_comparison.csv',
}
MANIFESTS = {
    'r2r_manifest': 'outputs/simulation/07c_method_and_integrity.json',
    'vm_manifest': 'outputs/analysis/06d_final_model_manifest.json',
}


def read_sources(project_root: Path) -> DashboardSources:
    project_root = Path(project_root)
    missing = [str(project_root / relative) for relative in list(FILES.values()) + list(MANIFESTS.values()) if not (project_root / relative).is_file()]
    if missing:
        raise FileNotFoundError('Missing Step 04/05/06/07 output files:\n' + '\n'.join(missing))
    csv = {name: pd.read_csv(project_root / path) for name, path in FILES.items()}
    js = {name: json.loads((project_root / path).read_text(encoding='utf-8')) for name, path in MANIFESTS.items()}
    sources = DashboardSources(**csv, **js)
    validate_sources(sources)
    return sources


def _require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f'{name} missing columns: {missing}')


def _close(actual: float, expected: float, label: str) -> None:
    if not np.isclose(float(actual), float(expected), atol=1e-5, rtol=1e-6):
        raise ValueError(f'{label}: recomputed {actual:.8f} disagrees with stored {expected:.8f}')


def validate_sources(d: DashboardSources) -> None:
    """Validate precomputed outputs; not statistical certification or safety validation."""
    _require_columns(d.spc, ['code_group', 'WAFER_ID', 'order_within_group', 'AVG_REMOVAL_RATE', 'zone', 'outside_provisional_3sigma', 'reference_CL_provisional', 'reference_UCL_provisional', 'reference_LCL_provisional'], '04A SPC')
    _require_columns(d.spc_summary, ['code_group', 'candidate_chart_rows', 'monitor_outside_3sigma_count'], '04A SPC summary')
    _require_columns(d.ar1, ['code_group', 'I_chart_monitor_flags', 'AR1_monitor_flags'], '04B AR1')
    _require_columns(d.fdc_cases, ['WAFER_ID', 'MACHINE_DATA', 'measured_MRR', 'frozen_signal_family_count'], '05E FDC cases')
    _require_columns(d.fdc_signals, ['WAFER_ID', 'MACHINE_DATA', 'signal', 'observed_median_nonzero', 'reference_q01', 'reference_q99', 'direction'], '05E FDC signals')
    _require_columns(d.fdc_summary, ['MACHINE_DATA', 'later_rows', 'later_two_or_more_families'], '05C FDC summary')
    _require_columns(d.vm_metrics, ['group', 'model', 'test_rows', 'all_mae', 'all_rmse', 'all_r2'], '06D VM metrics')
    _require_columns(d.vm_predictions, ['WAFER_ID', 'code_family', 'partition', 'model', 'actual_MRR', 'predicted_MRR'], '06D VM predictions')
    _require_columns(d.open_runs, ['run', 'setpoint_applied', 'measured_MRR_simulated', 'target_MRR_simulated', 'exogenous_drift_hidden_from_controller', 'process_noise_hidden_from_controller', 'metrology_noise_hidden_from_controller'], '07A open loop')
    _require_columns(d.closed_runs, ['run', 'setpoint_applied', 'measured_MRR_simulated', 'target_MRR_simulated', 'exogenous_drift_hidden_from_controller', 'process_noise_hidden_from_controller', 'metrology_noise_hidden_from_controller'], '07C closed loop')
    _require_columns(d.actions, ['completed_run', 'measured_mrr_completed_run', 'setpoint_applied_completed_run', 'setpoint_for_next_run', 'step_change_for_next_run'], '07C actions')
    _require_columns(d.r2r_comparison, ['metric', 'open_loop', 'closed_loop'], '07C comparison')
    if len(d.open_runs) != 120 or len(d.closed_runs) != 120 or len(d.actions) != 120:
        raise ValueError('R2R paired experiment must have 120 rows for each input')
    if not np.array_equal(d.open_runs.run.to_numpy(), np.arange(1, 121)) or not np.array_equal(d.closed_runs.run.to_numpy(), np.arange(1, 121)):
        raise ValueError('R2R run numbers not sequential 1-120')
    for col in ['exogenous_drift_hidden_from_controller', 'process_noise_hidden_from_controller', 'metrology_noise_hidden_from_controller']:
        if not np.allclose(d.open_runs[col], d.closed_runs[col], atol=1e-8):
            raise ValueError(f'R2R open/closed unequal simulated disturbances: {col}')
    if not np.allclose(d.closed_runs.setpoint_applied.to_numpy()[1:], d.actions.setpoint_for_next_run.to_numpy()[:-1], atol=1e-6):
        raise ValueError('R2R next-run-only action causality does not match trace')
    if not np.allclose(d.actions.measured_mrr_completed_run, d.closed_runs.measured_MRR_simulated, atol=1e-6):
        raise ValueError('R2R feedback actions do not match completed measurements')
    for mode, data in [('open_loop', d.open_runs), ('closed_loop', d.closed_runs)]:
        delta = data.measured_MRR_simulated - data.target_MRR_simulated
        actual = {'MAE_from_target': float(delta.abs().mean()), 'RMSE_from_target': float(np.sqrt(np.mean(delta ** 2))), 'last30_mean_signed_error': float(delta.tail(30).mean())}
        for metric, value in actual.items():
            entry = d.r2r_comparison.loc[d.r2r_comparison.metric == metric]
            if len(entry) != 1:
                raise ValueError(f'Missing R2R comparison row for {metric}')
            _close(value, entry.iloc[0]['open_loop' if mode == 'open_loop' else 'closed_loop'], f'{mode} {metric}')
    if d.r2r_manifest.get('paired_noise_drift_equal') is not True or d.r2r_manifest.get('no_posthoc_parameter_tuning') is not True:
        raise ValueError('Expected frozen paired R2R protocol not recorded')
    if len(d.fdc_cases) != 11 or len(d.fdc_signals) != 27:
        raise ValueError('FDC frozen casebook/evidence count differs from accepted 05E (11/27)')
    if d.fdc_cases.WAFER_ID.duplicated().any():
        raise ValueError('FDC casebook has duplicated wafer IDs')
    if int(d.fdc_summary.later_two_or_more_families.sum()) != 11:
        raise ValueError('05C and 05E candidate totals differ')
    if not d.fdc_signals.WAFER_ID.isin(d.fdc_cases.WAFER_ID).all():
        raise ValueError('FDC evidence refers to unknown case')
    if d.vm_predictions.duplicated(['WAFER_ID', 'code_family', 'model']).any():
        raise ValueError('VM repeated wafer/model combination')
    if set(d.vm_predictions.partition) != {'test'}:
        raise ValueError('VM predictions unexpectedly contain non-test rows')
    for _, row in d.vm_metrics.iterrows():
        subset = d.vm_predictions.loc[(d.vm_predictions.code_family == row['group']) & (d.vm_predictions.model == row['model'])]
        if len(subset) != int(row.test_rows):
            raise ValueError('VM test row count disagrees with predictions')
        e = subset.predicted_MRR - subset.actual_MRR
        _close(np.abs(e).mean(), row.all_mae, 'VM all_mae')
        _close(np.sqrt(np.mean(e**2)), row.all_rmse, 'VM all_rmse')
        _close(1-np.sum(e**2)/np.sum((subset.actual_MRR-subset.actual_MRR.mean())**2), row.all_r2, 'VM all_r2')
    if d.vm_manifest.get('status', '').startswith('ONE-TIME') is False:
        raise ValueError('Expected one-time final internal Test manifest')
    for _, row in d.spc_summary.iterrows():
        sub = d.spc[d.spc.code_group == row.code_group]
        if len(sub) != int(row.candidate_chart_rows):
            raise ValueError('SPC row count mismatch')
        flagged = sub.loc[sub.zone == 'monitor', 'outside_provisional_3sigma'].fillna(False).astype(bool).sum()
        if int(flagged) != int(row.monitor_outside_3sigma_count):
            raise ValueError('SPC later warning count mismatch')


def relative_mae_gain(d: DashboardSources, group: str) -> float:
    sub = d.vm_metrics.loc[d.vm_metrics.group == group].set_index('model')
    baseline = float(sub.loc['training_group_median', 'all_mae'])
    model = float(sub.loc['HGB_absolute_error', 'all_mae'])
    return (baseline-model) / baseline * 100


def r2r_relative_gain(d: DashboardSources, metric: str = 'MAE_from_target') -> float:
    row = d.r2r_comparison.loc[d.r2r_comparison.metric == metric].iloc[0]
    return 100 * (1 - float(row.closed_loop)/float(row.open_loop))
