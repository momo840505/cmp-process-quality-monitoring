"""Step 06D: frozen, one-time INTERNAL test of Step 06B's full-66 HGB.

All choices were frozen BEFORE looking at internal TEST outcomes. Preflight prints
schema/split checks without evaluating test targets. The evaluation is one-shot,
with an on-disk completion guard to discourage reruns and cherry-picking.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import tempfile

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TARGET = "AVG_REMOVAL_RATE"
PART = "vm_partition"
GROUP = "code_family"
REVIEW = "mrr_review_gt_1000"
SEED = 42
GROUP_CODES = {"Codes 1-3": (1, 2, 3), "Codes 4-6": (4, 5, 6)}
MODEL_PARAMS = {
    "loss": "absolute_error",
    "learning_rate": 0.05,
    "max_iter": 140,
    "max_leaf_nodes": 12,
    "min_samples_leaf": 15,
    "l2_regularization": 5.0,
    "random_state": 42,
    "early_stopping": False,
}


def fixed_group_columns(catalog: pd.DataFrame, codes: tuple[int, ...]) -> list[str]:
    """Allowlist ONLY documented code-specific sensor features."""
    if not {"feature_name", "english_definition"}.issubset(catalog.columns):
        raise ValueError("Missing feature catalog column(s)")
    names = catalog.feature_name.astype(str).tolist()
    if len(names) != len(set(names)):
        raise ValueError("Duplicate feature catalog entries")
    forbidden = ("WAFER_ID", "AVG_REMOVAL_RATE", "MRR", "TARGET", "TIMESTAMP",
                 "code_signature", "vm_partition", "code_family",
                 "mrr_review_gt_1000", "has_recorded_time_conflict")
    if any(x in names for x in forbidden):
        raise ValueError("Forbidden metadata/target in model feature allowlist")
    pref = tuple(f"code{k}__" for k in codes)
    cols = [n for n in names if n.startswith(pref)]
    if len(cols) != 66:
        raise ValueError(f"Expected the Step 06B frozen 66 features, found {len(cols)}")
    if any(n.lower().find("mrr") >= 0 or "timestamp" in n.lower() for n in cols):
        raise ValueError("Target or timestamp leaked through feature column names")
    return cols


def validate_structure(df: pd.DataFrame, catalog: pd.DataFrame) -> dict:
    need = {"WAFER_ID", PART, GROUP, "code_signature", "has_complete_code_signature",
            REVIEW, TARGET}
    if need-set(df.columns):
        raise ValueError(f"Missing data columns: {sorted(need-set(df.columns))}")
    if len(df) != 1166 or not df.WAFER_ID.is_unique:
        raise ValueError("Expected exactly 1166 distinct Training Stage A wafer rows")
    counts = df[PART].value_counts().to_dict()
    if counts != {"train": 817, "valid": 175, "test": 174}:
        raise ValueError(f"Unexpected original fixed partition: {counts}")
    if set(df[PART]) != {"train", "valid", "test"}:
        raise ValueError("Unexpected partition values")
    if TARGET in catalog.feature_name.values:
        raise ValueError("Target in feature catalog")
    if not set(catalog.feature_name).issubset(df.columns):
        raise ValueError("Feature catalog doesn't match input table")
    out = {}
    for group, codes in GROUP_CODES.items():
        g = df.loc[df[GROUP].eq(group)].copy()
        feature_cols = fixed_group_columns(catalog, codes)
        exp_sig = ",".join(map(str, codes))
        if not g.code_signature.astype(str).eq(exp_sig).all() or not g.has_complete_code_signature.astype(bool).all():
            raise ValueError(f"Bad code signature in {group}")
        if not all(pd.api.types.is_numeric_dtype(g[col]) for col in feature_cols):
            raise ValueError(f"Non-numeric predictor in {group}")
        x = g[feature_cols].to_numpy(dtype=float)
        if np.isinf(x).any():
            raise ValueError(f"Nonfinite feature in {group}")
        n = g[PART].value_counts().to_dict()
        # Fixed counts established in 06A/06B before inspecting internal Test.
        expected = {"Codes 1-3": {"train": 258, "valid": 55, "test": 55},
                    "Codes 4-6": {"train": 558, "valid": 120, "test": 119}}[group]
        if n != expected:
            raise ValueError(f"Unexpected {group} partitions: {n} != {expected}")
        if g.loc[g[PART].isin(["train","valid"]), TARGET].isna().any():
            raise ValueError("Train/valid MRR missing")
        out[group] = {"group_counts": n, "columns": feature_cols}
    incomplete = df.loc[~df[GROUP].isin(GROUP_CODES)]
    if len(incomplete) != 1 or bool(incomplete.iloc[0].has_complete_code_signature):
        raise ValueError("Unexpected count/type of incomplete group rows")
    if incomplete.iloc[0][PART] != "train":
        raise ValueError("Incomplete record partition differs from Step 06A")
    return out


def train_frozen(train_valid: pd.DataFrame, cols: list[str]):
    """Never touches any internal Test targets or test outcomes."""
    if len(train_valid) < 50 or train_valid[TARGET].isna().any():
        raise ValueError("Invalid train+valid")
    if not train_valid[PART].isin(["train", "valid"]).all():
        raise ValueError("Training data includes internal Test rows")
    median = float(np.median(train_valid[TARGET].to_numpy(dtype=float)))
    model = HistGradientBoostingRegressor(**MODEL_PARAMS)
    model.fit(train_valid[cols].astype(float), train_valid[TARGET].to_numpy(dtype=float))
    return median, model


def predict_from_features(model, baseline: float, test_features: pd.DataFrame, cols: list[str]) -> dict:
    """test_features MUST NOT include true MRR; only allowlisted X columns used."""
    if TARGET in test_features.columns or REVIEW in test_features.columns:
        raise ValueError("Use ONLY test feature columns for prediction")
    if test_features.empty:
        raise ValueError("Test feature rows missing")
    x = test_features.loc[:, cols].astype(float)
    return {"training_group_median": np.full(len(x), baseline, dtype=float),
            "HGB_absolute_error": model.predict(x)}


def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    if len(y) < 2 or len(pred) != len(y) or not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError("Invalid metric input")
    return {"n": len(y),
            "mae": float(mean_absolute_error(y, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y, pred))),
            "r2": float(r2_score(y, pred))}


def sensitivity_metrics(y: np.ndarray, pred: np.ndarray, review: np.ndarray) -> dict:
    # DESCRIPTION ONLY: do not select, fit, or trim a model using this target flag.
    keep = ~review.astype(bool)
    if sum(keep) < 2:
        return {"n": int(sum(keep)), "mae": None, "rmse": None, "r2": None}
    return metrics(y[keep], pred[keep])


def plot_predictions(preds: pd.DataFrame, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    for group in GROUP_CODES:
        d = preds[preds[GROUP].eq(group)]
        ordinary = d.loc[~d[REVIEW]].copy()
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
        # ALL rows including extreme target on the left; ordinary zoom on the right.
        for ax, data, title in [(axes[0], d, "Full internal Test"),
                                (axes[1], ordinary, "Ordinary-scale view (descriptive)")]:
            for model_name, marker in [("training_group_median", "o"), ("HGB_absolute_error", "x")]:
                m = data[data.model.eq(model_name)]
                if len(m):
                    ax.scatter(m.actual_MRR, m.predicted_MRR, label=model_name, marker=marker, alpha=.72)
            if len(data):
                lo = min(data.actual_MRR.min(), data.predicted_MRR.min())
                hi = max(data.actual_MRR.max(), data.predicted_MRR.max())
                ax.plot([lo, hi], [lo, hi], linestyle="--", color="gray", linewidth=1.2)
            ax.set(title=title, xlabel="Observed MRR (published scale)", ylabel="Predicted MRR")
            ax.grid(alpha=.17)
        axes[0].legend(loc="best", fontsize=8)
        fig.suptitle(f"Step 06D / {group}: frozen INTERNAL TEST (not external competition Test)")
        fig.text(.5, .01, "All test rows included in PRIMARY metrics. Review-only zoom is not a second test selection.", ha="center", fontsize=8)
        fig.tight_layout(rect=[0,.04,1,.95])
        suffix = "codes1_3" if group=="Codes 1-3" else "codes4_6"
        fig.savefig(fig_dir/f"06d_{suffix}_test_predictions.png", dpi=150)
        plt.close(fig)


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def run(project: Path, preflight: bool = False) -> tuple | dict:
    analysis = project/"outputs"/"analysis"
    model_dir = project/"outputs"/"models"
    inp = analysis/"06a_stageA_vm_features.csv"
    catpath = analysis/"06a_feature_catalog.csv"
    if not inp.is_file() or not catpath.is_file():
        raise FileNotFoundError("Please finish Step 06A before running Step 06D")
    if not (analysis/"06b_model_protocol.json").exists() or not (analysis/"06c_robustness_notes.json").exists():
        raise FileNotFoundError("Please complete and save Steps 06B and 06C first")
    guard = analysis/"06d_INTERNAL_TEST_FINALIZED.json"
    if guard.exists() and not preflight:
        raise FileExistsError("Step 06D already finalized. Do not rerun and choose a favorable score. See existing results.")
    d = pd.read_csv(inp, low_memory=False)
    cat = pd.read_csv(catpath)
    v = validate_structure(d, cat)
    # No test outcomes examined in preflight: test y is never scored/summarized.
    if preflight:
        print("PASS preflight: 1,166 original TRAIN Stage A wafers; partitions 817/175/174")
        for group, meta in v.items():
            print(f"{group}: 66 frozen sensor features | ref=train+valid "
                  f"{meta['group_counts']['train']+meta['group_counts']['valid']} | held-out test {meta['group_counts']['test']}")
        print("No TEST scores, metrics, flags, or outcomes were inspected in this preflight.")
        return v
    if (analysis/"06d_internal_test_metrics.csv").exists():
        raise FileExistsError("Final test metrics file already exists; refuse to overwrite")
    rows=[]
    pred_rows=[]
    stored={}
    metadata={}
    for group, meta in v.items():
        cols=meta["columns"]
        g=d.loc[d[GROUP].eq(group)]
        ref=g.loc[g[PART].isin(["train", "valid"])].copy()
        test=g.loc[g[PART].eq("test")].copy()
        if set(ref.WAFER_ID) & set(test.WAFER_ID):
            raise ValueError("Wafer ID overlap between train+valid and TEST")
        median, model=train_frozen(ref, cols)
        # Only pass X columns into the prediction function.
        yhat=predict_from_features(model, median, test.loc[:, cols].copy(), cols)
        # This is the first and only time the held-out test target is used.
        actual=test[TARGET].to_numpy(dtype=float)
        review=test[REVIEW].astype(bool).to_numpy()
        if not np.isfinite(actual).all():
            raise ValueError("Missing/invalid test target values")
        for method,pred in yhat.items():
            full=metrics(actual,pred)
            supplemental=sensitivity_metrics(actual,pred,review)
            rows.append({"group":group,"model":method,"fit_rows_train_plus_valid":len(ref),
                        "test_rows":len(test),"feature_count":len(cols) if method=="HGB_absolute_error" else 0,
                        "reference_median":median,"test_review_gt1000":int(review.sum()),
                        **{f"all_{k}":x for k,x in full.items()},
                        **{f"ordinary_{k}":x for k,x in supplemental.items()}})
            for wid,y,yp,is_rev in zip(test.WAFER_ID,actual,pred,review):
                pred_rows.append({"WAFER_ID":int(wid),GROUP:group,"partition":"test",
                                  "model":method,"actual_MRR":float(y),"predicted_MRR":float(yp),
                                  "absolute_error":float(abs(y-yp)),REVIEW:bool(is_rev)})
        stored[group]=model
        metadata[group]={"features":cols,"median":median,"ref_rows":len(ref),"test_rows":len(test)}
        print(f"{group}: train+valid {len(ref)} | internal test {len(test)} | 66 frozen features")
    results=pd.DataFrame(rows)
    pred_df=pd.DataFrame(pred_rows)
    if len(pred_df)!=2*(55+119) or results.shape[0]!=4:
        raise AssertionError("Final results row count inconsistent with frozen protocol")
    # Write everything to a staging directory. The guard is saved LAST.
    analysis.mkdir(parents=True,exist_ok=True)
    model_dir.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="step06d_stage_",dir=analysis) as tmpdir:
        tmp=Path(tmpdir)
        results.to_csv(tmp/"06d_internal_test_metrics.csv",index=False,encoding="utf-8-sig")
        pred_df.to_csv(tmp/"06d_internal_test_predictions.csv",index=False,encoding="utf-8-sig")
        plot_predictions(pred_df, tmp/"figures")
        for group, model in stored.items():
            fn="06d_hgb_codes1_3.joblib" if group=="Codes 1-3" else "06d_hgb_codes4_6.joblib"
            joblib.dump(model,tmp/fn)
        notes={
            "status":"ONE-TIME FINAL INTERNAL TEST, not PHM external competition Test",
            "decision_frozen_before_test":"Full 66-feature HGB_absolute_error per complete code group; compare train+valid median baseline; params unchanged from Step 06B",
            "refit":"Train+Valid combined (no hyperparameter tuning); Test excluded from fitting",
            "hyperparameters":MODEL_PARAMS,
            "internal_partition_counts":{"train":817,"valid":175,"test":174},
            "excluded_from_group_models":"One incomplete code4-only TRAIN wafer, retained in Step 06A table",
            "model_info":metadata,
            "training_data_sha256":sha256(inp),
            "feature_catalog_sha256":sha256(catpath),
            "python_version":platform.python_version(),
            "sklearn_version":sklearn.__version__,
            "limits":[
                "Test is one-time holdout from original competition TRAIN Stage A, NOT external validation.",
                "Split randomized by unique WAFER_ID, not a proven production chronology; potential drift/distribution shift.",
                "Four >1000 MRR review cases retained in train+valid fits; Test all-row metrics retain all its rows.",
                "Review flag is TARGET-derived and ONLY used after prediction for descriptive sensitivity; never a feature.",
                "MRR has no engineering USL/LSL or validated unit/target; MAE/RMSE/R2 are not factory yield or Cp/Cpk.",
                "Full process sensor trajectory is used; prediction occurs after process, prior to metrology, not live early-warning FDC.",
                "No actual defect/failure labels; never claim real fab deployment or confirmed root causes.",
                "Do not adapt thresholds or model hyperparameters based on this held-out TEST score."
            ]
        }
        (tmp/"06d_final_model_manifest.json").write_text(json.dumps(notes,ensure_ascii=False,indent=2),encoding="utf-8")
        for filename in ["06d_internal_test_metrics.csv","06d_internal_test_predictions.csv","06d_final_model_manifest.json"]:
            (tmp/filename).replace(analysis/filename)
        final_figdir=analysis/"figures"
        final_figdir.mkdir(exist_ok=True,parents=True)
        for f in (tmp/"figures").glob("*.png"):
            f.replace(final_figdir/f.name)
        for f in tmp.glob("*.joblib"):
            f.replace(model_dir/f.name)
    # Write marker LAST. No test score-driven decision or retry recommended.
    guard.write_text(json.dumps({"status":"ONE-TIME INTERNAL TEST COMPLETED",
                                 "results":"06d_internal_test_metrics.csv",
                                 "method":"Precommitted full66 HGB and group median (train+valid fits)",
                                 "do_not_rerun_for_better_metrics":True},ensure_ascii=False,indent=2),encoding="utf-8")
    print("\nFINAL INTERNAL TEST SCORES (all rows PRIMARY):")
    print(results[["group","model","test_rows","all_mae","all_rmse","all_r2","ordinary_mae"]].to_string(index=False,float_format=lambda x:f"{x:.4f}"))
    print("\nPASS: one-time internal Test finalized; no more tuning or retry for a preferred score.")
    print(f"Reports: {analysis}\nModels: {model_dir}")
    return results,pred_df


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--preflight",action="store_true",help="Check schema/splits without evaluating held-out Test targets")
    args=parser.parse_args()
    run(args.project, preflight=args.preflight)

if __name__=="__main__":
    main()
