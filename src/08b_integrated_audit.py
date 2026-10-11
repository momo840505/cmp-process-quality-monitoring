"""Step 08B: read-only end-to-end audit, no model fit and no simulation rerun.

Run from project root:
  .\\.venv\\Scripts\\python.exe .\\src\\08b_integrated_audit.py
Reports are newly created under outputs/audit and never overwrite model results.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Check:
    name: str
    status: str  # PASS / WARN / FAIL
    detail: str


class Audit:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.checks: list[Check] = []

    def check(self, name: str, success: bool, detail: str, *, warn: bool = False) -> None:
        self.checks.append(Check(name, 'PASS' if success else ('WARN' if warn else 'FAIL'), detail))

    def fail(self, name: str, detail: str) -> None:
        self.checks.append(Check(name, 'FAIL', detail))

    def require(self, rel: str, name: str) -> Path | None:
        file = self.root / rel
        self.check(name, file.is_file(), f'{rel} {"exists" if file.is_file() else "MISSING"}')
        return file if file.is_file() else None

    def json(self, rel: str, name: str) -> dict | None:
        f=self.require(rel,name+' / file')
        if f is None: return None
        try:
            data=json.loads(f.read_text(encoding='utf-8-sig'))
            self.check(name, isinstance(data,dict), 'JSON is readable and a dict')
            return data if isinstance(data,dict) else None
        except Exception as e:
            self.fail(name, f'JSON parsing: {type(e).__name__}: {e}')
            return None

    def csv(self, rel: str, name: str) -> pd.DataFrame | None:
        f=self.require(rel,name+' / file')
        if f is None: return None
        try:
            df=pd.read_csv(f)
            self.check(name, True, f'{len(df):,} rows, {len(df.columns)} columns')
            return df
        except Exception as e:
            self.fail(name, f'CSV parsing: {type(e).__name__}: {e}')
            return None


def digest(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def audit_data(a: Audit) -> None:
    inventory=a.json('outputs/audit/01_dataset_summary.json','01 inventory')
    cleaning=a.json('outputs/audit/02b_cleaning_summary.json','02B cleaning')
    quality=a.csv('outputs/processed/02b_wafer_stage_quality.csv','02B wafer-stage table')
    if inventory and cleaning:
        for split in ('training','test','validation'):
            try:
                s=inventory[split];c=cleaning[split]
                valid=(s['sensor_rows']==c['raw_sensor_rows']==c['derived_sensor_rows']+c['exact_duplicate_rows_removed_from_derived_view']
                    and s['label_rows']==c['quality_rows'] and s['header_only_files']==c['header_only_files'])
                a.check(f'01→02B {split} conservation',valid,
                    f'raw={c["raw_sensor_rows"]}, derived={c["derived_sensor_rows"]}, exact_dedup={c["exact_duplicate_rows_removed_from_derived_view"]}, label={c["quality_rows"]}')
            except (KeyError,TypeError,ValueError) as e:
                a.fail(f'01→02B {split} conservation',str(e))
    if quality is not None:
        needed={'split','WAFER_ID','STAGE','AVG_REMOVAL_RATE'}
        if not needed.issubset(quality.columns):
            a.fail('02B quality key uniqueness',f'Missing {sorted(needed-set(quality.columns))}')
        else:
            counts=quality.groupby('split').size().to_dict()
            unique=not quality.duplicated(['split','WAFER_ID','STAGE']).any()
            a.check('02B label counts',len(quality)==2829 and counts=={'training':1981,'test':424,'validation':424},f'total={len(quality)}, per_split={counts}')
            a.check('02B quality key uniqueness',unique,'No duplicated split/WAFER_ID/STAGE keys' if unique else 'DUPLICATED wafer-stage keys')
    stage=a.json('outputs/analysis/03b_investigation_summary.json','03B process group')
    if stage:
        a.check('03B sensor group count',stage.get('stage_A_wafer_stage_count')==1166 and
                stage.get('counts_by_code_group',{}).get('Codes 1-3')==368 and stage.get('counts_by_code_group',{}).get('Codes 4-6')==798,
                'Stage A 1,166; Codes 1–3: 368; Codes 4–6: 798')
    vm_a=a.json('outputs/analysis/06a_feature_audit.json','06A features')
    vm_table=a.csv('outputs/analysis/06a_stageA_vm_features.csv','06A wafer features')
    catalog=a.csv('outputs/analysis/06a_feature_catalog.csv','06A feature catalog')
    if vm_table is not None and vm_a:
        a.check('06A single wafer per row',len(vm_table)==1166 and not vm_table.WAFER_ID.duplicated().any(),f'rows={len(vm_table)}, duplicate wafer IDs={int(vm_table.WAFER_ID.duplicated().sum())}')
        a.check('06A split conservation',vm_a.get('partition_counts')=={'train':817,'valid':175,'test':174},f'{vm_a.get("partition_counts")}')
    if catalog is not None:
        a.check('06A expected feature count',len(catalog)==133,f'{len(catalog)} catalog rows; catalog inclusion still checked separately')


def audit_models(a: Audit) -> None:
    manifest=a.json('outputs/analysis/06d_final_model_manifest.json','06D manifest')
    finalized=a.json('outputs/analysis/06d_INTERNAL_TEST_FINALIZED.json','06D finalized marker')
    if manifest:
        a.check('06D one-time model description','ONE-TIME' in manifest.get('status','') and 'Train+Valid' in manifest.get('refit',''),str(manifest.get('status')))
        models=manifest.get('model_info',{})
        for group in ['Codes 1-3','Codes 4-6']:
            m=models.get(group,{})
            feat=m.get('features',[])
            no_leak=not any(re.search(r'(WAFER_ID|AVG_REMOVAL_RATE|actual_MRR|mrr_review|partition|first_timestamp)',str(x),re.I) for x in feat)
            a.check('06D frozen feature list '+group,len(feat)==66 and len(set(feat))==66 and no_leak,f'{len(feat)} features; forbidden names={not no_leak}')
    if finalized:
        a.check('06D final Test lock',finalized.get('do_not_rerun_for_better_metrics') is True, str(finalized.get('status')))
    try:
        # Import Step08A to re-validate outputs (does not fit models or run simulations).
        sys.path.insert(0,str(a.root))
        from src.dashboard_data import read_sources
        d=read_sources(a.root)
        a.check('08A complete dashboard data validation',True,'SPC/FDC/VM/R2R values recalculated and matched existing outputs')
        a.check('06D unique Test wafers',len(d.vm_predictions)==348 and
                len(d.vm_predictions.drop_duplicates(['WAFER_ID','code_family']))==174,
                '174 internal test wafers; 2 model predictions per wafer')
        a.check('05E FDC cases vs signals',len(d.fdc_cases)==11 and len(d.fdc_signals)==27,'11 candidate wafers / 27 sensor evidence records')
        a.check('07C R2R conservation and causality', len(d.closed_runs)==120 and len(d.actions)==120,
                '120 runs; exact paired noise/drift and t→t+1 feedback checked by 08A validator')
    except Exception as e:
        a.fail('08A complete dashboard data validation',f'{type(e).__name__}: {e}')


def audit_frozen(a: Audit) -> None:
    ctrl=a.json('outputs/simulation/07b_controller_manifest.json','07B frozen controller')
    sim=a.json('outputs/simulation/07c_method_and_integrity.json','07C paired protocol')
    done=a.json('outputs/simulation/07c_FINALIZED.json','07C finalized marker')
    if ctrl and sim:
        src=a.root/'src/r2r_controller.py'
        if not src.exists():
            a.fail('07B controller source frozen','Missing src/r2r_controller.py')
        else:
            actual=digest(src)
            frozen=ctrl.get('controller_source_sha256')
            a.check('07B controller SHA-256 unchanged',actual==frozen,f'actual={actual[:12]}..., frozen={str(frozen)[:12]}...')
        a.check('07B→07C controller parameters frozen',sim.get('controller_config_frozen_from_07b')==ctrl.get('controller_config'),
                f'07B/07C configs equal: {sim.get("controller_config_frozen_from_07b")==ctrl.get("controller_config")}')
        a.check('07C no future signals',sim.get('paired_noise_drift_equal') is True and sim.get('simulator_true_gain_is_hidden_from_controller') is True and sim.get('no_posthoc_parameter_tuning') is True,
                'Paired disturbances, hidden plant gain, frozen settings affirmed by saved protocol')
        open_file=a.root/'outputs/simulation/07a_open_loop_simulated_runs.csv'
        expected=ctrl.get('open_loop_input_audit',{}).get('sha256_step07a_csv')
        if open_file.exists() and expected:
            a.check('07A input CSV unmodified',digest(open_file)==expected,'07A SHA-256 checked against 07B frozen manifest')
    if done:
        a.check('07C final lock','FINALIZED' in done.get('status','') and done.get('no_real_fab_metrics') is True,str(done.get('status')))


def audit_github(a: Audit) -> None:
    required=['dashboard.py','src/dashboard_data.py','README.md','.gitignore','requirements.txt','requirements_dashboard.txt']
    for rel in required:
        exists=(a.root/rel).exists()
        a.check('Github readiness: '+rel,exists,'present' if exists else 'missing / need preparation',warn=True)
    ignores=(a.root/'.gitignore')
    if ignores.exists():
        content=ignores.read_text(encoding='utf-8',errors='ignore').lower()
        for label,pattern in [('.venv','venv'),('cache','__pycache__'),('secret env','.env'),('data files','data/'),('model results','outputs/')]:
            a.check(f'Git ignore: {label}',pattern in content,
                    ('looks covered' if pattern in content else 'check .gitignore before GitHub push'),warn=True)
    # No network, no changes to index or working tree. Git check is advisory.
    try:
        root=subprocess.run(['git','-C',str(a.root),'rev-parse','--show-toplevel'],capture_output=True,text=True,timeout=4)
        if root.returncode!=0:
            a.check('Git staged risk scan',False,'No Git repository yet; run before first push',warn=True)
            return
        staged=subprocess.run(['git','-C',str(a.root),'ls-files','--cached','-z'],capture_output=True,timeout=8)
        if staged.returncode!=0:
            a.check('Git staged risk scan',False,'git ls-files failed',warn=True)
            return
        files=[s.decode('utf-8',errors='replace') for s in staged.stdout.split(b'\x00') if s]
        risk=[]
        pats=[r'(^|/)\.venv/',r'(^|/)\.env($|\.)',r'(^|/)data/raw/',r'(^|/)outputs/models/',r'\.zip$',r'\.joblib$',r'\.pkl$',r'(^|/)__pycache__/',r'\.(nc|xlsx|pdf)$']
        for name in files:
            if any(re.search(p,name,re.I) for p in pats):risk.append(name)
        a.check('Git staged risk scan',len(risk)==0,
                'No obvious restricted/large files staged' if not risk else 'REVIEW tracked files: '+', '.join(risk[:20]),warn=True)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        a.check('Git staged risk scan',False,'Git unavailable or timed out',warn=True)


def run_audit(root: Path) -> list[Check]:
    a=Audit(root)
    audit_data(a)
    audit_models(a)
    audit_frozen(a)
    audit_github(a)
    return a.checks


def write_reports(root: Path,checks: list[Check]) -> tuple[Path,Path]:
    out=root/'outputs/audit'
    out.mkdir(parents=True,exist_ok=True)
    counts={s:sum(c.status==s for c in checks) for s in ('PASS','WARN','FAIL')}
    payload={'stage':'08B read-only integration audit','scope':'Existing outputs only: no model retrain, no R2R rerun, no raw data edits',
             'summary':counts,'checks':[vars(c) for c in checks],
             'interpretation':'PASS means programmatic consistency, not verified semiconductor process safety or factory deployment.'}
    json_path=out/'08b_integration_audit.json'
    md_path=out/'08b_integration_audit.md'
    json_path.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=['# Step 08B｜專案整合驗收','',f'**檢查結果：通過 {counts["PASS"]}、待處理 {counts["WARN"]}、失敗 {counts["FAIL"]}**','',
           '本報告只核對既有輸出，不重新訓練、不重新測 Test、不重新執行 R2R。','',
           '| 狀態 | 檢查 | 結果與處理說明 |','|---|---|---|']
    for c in checks:
        lines.append('| '+ ' | '.join([c.status,c.name.replace('|','/'),c.detail.replace('|','/').replace('\n',' ')])+' |')
    lines.extend(['','## 對外展示前必須說明的限制','',
       '- PHM 2016 CMP 是公開研究資料；匿名 Code 不等於已確認設備或配方。','- SPC 為探索性示範界限，不代表廠內核准管制限或不良率。',
       '- FDC 的 11 片晶圓是待查候選，不代表故障分類或已確認根因。','- Virtual Metrology 是原始 Training 內部 Test，不是官方外部 Test；不能依最終分數重新選模型。',
       '- R2R 是獨立人工模擬，不是 PHM／Micron 真實改善結果。','',
       '## GitHub 提交原則','',
       '- 優先上傳可執行程式、測試、README、資料來源與下載指引，以及少量經挑選的輸出圖表。',
       '- 請勿直接提交 .venv、環境變數、私密資訊、PHM 原始資料、所有輸出 CSV、joblib 模型或大型壓縮檔。',
       '- 本檢查的 WARN 是提醒你做公開前決策，不會修改 .gitignore 或 Git index。',''])
    md_path.write_text('\n'.join(lines),encoding='utf-8')
    return json_path,md_path


def main() -> int:
    checks=run_audit(ROOT)
    a,b=write_reports(ROOT,checks)
    n={s:sum(c.status==s for c in checks) for s in ('PASS','WARN','FAIL')}
    print(f'Step 08B integration audit: PASS={n["PASS"]}, WARN={n["WARN"]}, FAIL={n["FAIL"]}')
    for c in checks:
        if c.status!='PASS':print(f'  [{c.status}] {c.name}: {c.detail}')
    print('Report JSON:',a)
    print('Report Markdown:',b)
    print('No Test re-evaluation beyond score reconciliation; no model fit; no R2R rerun; no source changes.')
    return 1 if n['FAIL'] else 0

if __name__=='__main__':
    raise SystemExit(main())
