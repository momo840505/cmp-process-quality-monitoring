"""Step 06B: group-specific, training-only VM baseline versus robust tree model.

Inputs: outputs/analysis/06a_stageA_vm_features.csv, 06a_feature_catalog.csv.
Outputs: validation-only reports in outputs/analysis; no test-row evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TARGET = "AVG_REMOVAL_RATE"
PART = "vm_partition"
GROUP = "code_family"
REVIEW = "mrr_review_gt_1000"
ALLOWED_GROUPS = {"Codes 1-3": (1, 2, 3), "Codes 4-6": (4, 5, 6)}
POSSIBLE_PARTS = {"train", "valid", "test"}
SEED = 42


def group_columns(catalog: pd.DataFrame, codes: tuple[int, ...]) -> list[str]:
    """Choose only numeric sensor-derived features from documented allowlist."""
    names = catalog["feature_name"].astype(str).tolist()
    prefix = tuple(f"code{x}__" for x in codes)
    result = [c for c in names if c.startswith(prefix)]
    if not result:
        raise ValueError("No group-specific features in allowlist")
    return result


def validate_inputs(df: pd.DataFrame, catalog: pd.DataFrame) -> dict:
    if df.empty or catalog.empty:
        raise ValueError("Input files cannot be empty")
    must = {"WAFER_ID", "code_family", "code_signature", "has_complete_code_signature", "mrr_review_gt_1000", PART, TARGET}
    if not must.issubset(df.columns):
        raise ValueError(f"Missing required columns: {sorted(must-set(df.columns))}")
    if not {"feature_name", "english_definition"}.issubset(catalog.columns):
        raise ValueError("Malformed feature catalog")
    if not df["WAFER_ID"].is_unique:
        raise ValueError("Duplicate WAFER_ID across partitions")
    if set(df[PART].dropna().unique()) != POSSIBLE_PARTS:
        raise ValueError("Unexpected partition names")
    if df[TARGET].isna().any():
        raise ValueError("Missing MRR target")
    feature_names = catalog.feature_name.astype(str).tolist()
    if len(feature_names) != len(set(feature_names)) or not set(feature_names).issubset(df.columns):
        raise ValueError("Duplicate/missing catalog features")
    forbidden = {"WAFER_ID", TARGET, REVIEW, GROUP, PART, "code_signature", "first_timestamp", "last_timestamp"}
    if forbidden & set(feature_names):
        raise ValueError(f"Forbidden feature(s): {forbidden & set(feature_names)}")
    expected = {"Codes 1-3": {1,2,3}, "Codes 4-6": {4,5,6}}
    counts = {}
    for g,codes in expected.items():
        rows = df[df[GROUP].eq(g)]
        if len(rows)==0:
            raise ValueError(f"No {g} rows")
        signatures = rows.code_signature.astype(str).map(lambda s: set(int(n) for n in s.split(',')))
        if not signatures.map(lambda s: s == codes).all():
            raise ValueError(f"Code signature mismatch in {g}")
        if not rows["has_complete_code_signature"].astype(bool).all():
            raise ValueError(f"Incomplete code signatures inside {g}")
        cols = group_columns(catalog, tuple(sorted(codes)))
        if any(not pd.api.types.is_numeric_dtype(rows[c]) for c in cols):
            raise ValueError(f"Non-numeric chosen feature in {g}")
        if any(np.isinf(rows[cols].to_numpy(dtype=float)).ravel()):
            raise ValueError(f"Infinite feature in {g}")
        counts[g] = {k: int((rows[PART] == k).sum()) for k in ['train','valid','test']}
    incomplete = df[~df[GROUP].isin(ALLOWED_GROUPS)]
    return {"partition_counts": df[PART].value_counts().to_dict(),
            "code_group_counts": counts,
            "excluded_incomplete_rows": int(len(incomplete)),
            "excluded_incomplete_IDs": incomplete.WAFER_ID.astype(str).tolist(),
            "catalog_feature_count": len(feature_names)}


def ordinary_metric_view(y_true: np.ndarray, y_pred: np.ndarray, flagged: np.ndarray) -> dict:
    # Pure reporting only; this does NOT change training or model selection.
    keep = ~np.asarray(flagged, dtype=bool)
    if not np.any(keep):
        return {"n": 0, "mae": None, "rmse": None, "r2": None}
    return score(y_true[keep], y_pred[keep])


def score(y: np.ndarray, pred: np.ndarray) -> dict:
    if len(y)!=len(pred) or len(y)==0 or not np.isfinite(pred).all():
        raise ValueError("Invalid scoring inputs")
    return {"n": int(len(y)),
            "mae": round(float(mean_absolute_error(y,pred)),6),
            "rmse": round(float(np.sqrt(mean_squared_error(y,pred))),6),
            "r2": round(float(r2_score(y,pred)),6) if len(y)>1 else None}


def fit_models(train: pd.DataFrame, valid: pd.DataFrame, cols: list[str]):
    # Group median baseline is robust even when rare huge MRRs are present.
    y_train = train[TARGET].to_numpy(dtype=float)
    median = float(np.median(y_train))
    base_pred = np.full(len(valid), median)
    # This regressor can ingest missing numeric features directly, no imputation
    # fitted using validation information. Absolute error loss helps guard against
    # the 3 large training MRR review observations (all remain in training).
    model = HistGradientBoostingRegressor(
        loss="absolute_error", learning_rate=0.05, max_iter=140,
        max_leaf_nodes=12, min_samples_leaf=15, l2_regularization=5.0,
        random_state=SEED, early_stopping=False,
    )
    x_train = train[cols].astype(float)
    x_valid = valid[cols].astype(float)
    model.fit(x_train, y_train)
    model_pred = model.predict(x_valid)
    return median, base_pred, model_pred, model


def plot_validation(pred: pd.DataFrame, out: Path) -> None:
    """Display only ordinary-scale points; all extreme rows stay in scoring."""
    path = out / "figures"
    path.mkdir(parents=True,exist_ok=True)
    for group in ALLOWED_GROUPS:
        d = pred[(pred.code_family==group)&(~pred.mrr_review_gt_1000)].copy()
        fig, ax=plt.subplots(figsize=(9.0,5.7))
        for model, label,marker in [("training_group_median","Training median baseline","o"),
                                   ("HGB_absolute_error","HGB model","x")]:
            a=d[d.model==model]
            ax.scatter(a.actual_MRR,a.predicted_MRR,label=label,alpha=0.68,s=30,marker=marker)
        lo=min(float(d.actual_MRR.min()),float(d.predicted_MRR.min()))
        hi=max(float(d.actual_MRR.max()),float(d.predicted_MRR.max()))
        pad=(hi-lo)*0.06
        ax.plot([lo-pad,hi+pad],[lo-pad,hi+pad],linestyle="--",color="gray",label="Ideal prediction")
        ax.set(xlabel="Observed MRR (published dataset scale)",
               ylabel="Predicted MRR (same scale)",
               title=f"Step 06B validation: {group} (ordinary-scale sensitivity view)")
        ax.legend(loc="best")
        fig.text(.5,.01,"All review-flagged MRR > 1000 INCLUDED in primary metrics but excluded from this plot only.",ha="center",fontsize=8)
        fig.tight_layout(rect=[0,.04,1,1])
        fig.savefig(path / f"06b_{'codes1_3' if group=='Codes 1-3' else 'codes4_6'}_validation_scatter.png",dpi=150)
        plt.close(fig)


def execute(base: Path):
    inp = base/"outputs"/"analysis"
    out = inp
    df = pd.read_csv(inp/"06a_stageA_vm_features.csv", low_memory=False)
    catalog = pd.read_csv(inp/"06a_feature_catalog.csv")
    audit = validate_inputs(df, catalog)
    # No Test row enters model fitting, calibration, scoring or model selection.
    tv = df[df[PART].isin(["train","valid"])].copy()
    predictions = []
    summary = []
    chosen_cols = {}
    for g,codes in ALLOWED_GROUPS.items():
        d = tv.loc[tv[GROUP].eq(g)]
        train = d.loc[d[PART].eq("train")].copy()
        valid = d.loc[d[PART].eq("valid")].copy()
        if len(train) < 100 or len(valid)<30:
            raise ValueError(f"Insufficient group data {g}")
        cols = group_columns(catalog,codes)
        chosen_cols[g] = cols
        baseline, base_pred, hgb_pred, model = fit_models(train, valid, cols)
        y = valid[TARGET].to_numpy(dtype=float)
        review = valid[REVIEW].astype(bool).to_numpy()
        for name,pred in [("training_group_median",base_pred),("HGB_absolute_error",hgb_pred)]:
            s_full=score(y,pred)
            s_ordinary=ordinary_metric_view(y,pred,review)
            summary.append({"group":g,"model":name,"train_rows":len(train),"valid_rows":len(valid),
                            "train_review_gt1000":int(train[REVIEW].astype(bool).sum()),
                            "valid_review_gt1000":int(review.sum()),
                            "feature_count":0 if name=="training_group_median" else len(cols),
                            "train_group_median":baseline,
                            **{f"all_{k}":v for k,v in s_full.items()},
                            **{f"ordinary_{k}":v for k,v in s_ordinary.items()}})
            predictions.extend({"WAFER_ID":wid,"code_family":g,"partition":"valid",
                                "actual_MRR":actual,"model":name,"predicted_MRR":round(float(yhat),6),
                                "absolute_error":round(float(abs(actual-yhat)),6),
                                "mrr_review_gt_1000":bool(flag)}
                for wid,actual,yhat,flag in zip(valid.WAFER_ID,y,pred,review))
        print(f"{g}: TRAIN {len(train)} / VALID {len(valid)}, {len(cols)} features; "
              f"train median {baseline:.3f}; validation review >1000: {int(review.sum())}")
    pred=pd.DataFrame(predictions)
    metrics=pd.DataFrame(summary)
    if len(pred)!=2*sum(audit["code_group_counts"][g]["valid"] for g in ALLOWED_GROUPS):
        raise ValueError("Prediction row count mismatch")
    if set(pred.model.unique())!={"training_group_median","HGB_absolute_error"}:
        raise ValueError("Missing model results")
    out.mkdir(parents=True,exist_ok=True)
    metrics.to_csv(out/"06b_validation_metrics.csv",index=False,encoding="utf-8-sig")
    plot_validation(pred, out)
    pred.to_csv(out/"06b_validation_predictions.csv",index=False,encoding="utf-8-sig")
    notes={"status":"VALIDATION-ONLY fixed-model comparison, NOT an external test result",
           "input_training_stage_A_rows":len(df),"reference_partitions":{"train":int((df[PART]=="train").sum()),"valid":int((df[PART]=="valid").sum()),"test_untouched":int((df[PART]=="test").sum())},
           "method":"Per anonymous code family: training-group MRR median versus preset HistGradientBoostingRegressor(loss=absolute_error, no early stopping). No use of MRR as feature.",
           "allowlist_by_group":chosen_cols,
           "training_outliers":"All 3 >1000 Training rows kept in both group models; no outcome-based trimming.",
           "validation_review":"1 >1000 Valid row kept in ALL-row primary MAE/RMSE/R2; ordinary-only view reported as secondary sensitivity, NOT for picking model.",
           "incomplete_code_rows":audit["excluded_incomplete_rows"],
           "important_limits":["All features summarize COMPLETE sensor trajectories: post-process, pre-metrology quality prediction; not live FDC.","Review MRR >1000 is NOT confirmed sensor error or product failure.","Training/Valid wafer-exclusive random split, not verified future fab chronological holdout.","Original PHM competition Test/Validation unused; internal Test untouched for final evaluation.","Valid can be used for model comparison; report it separately from future Test; do not tune on Test.","No engineering MRR units, target or specification limits are assumed.","Model metrics are predictive errors, not validated fab yield improvement."]}
    (out/"06b_model_protocol.json").write_text(json.dumps(notes,ensure_ascii=False,indent=2),encoding="utf-8")
    print("\nVALIDATION RESULTS (primary all rows first; ordinary only is sensitivity):")
    print(metrics[["group","model","all_mae","all_rmse","all_r2","ordinary_mae","ordinary_rmse","ordinary_r2"]].to_string(index=False))
    print("\nPASS: frozen train/valid comparison; internal TEST excluded from evaluation; files in outputs/analysis")
    return metrics,pred,notes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project",type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args()
    execute(args.project)
