"""Step 07C: one-time paired open-loop versus frozen EWMA R2R simulation.

SIMULATION ONLY. NOT PHM process data, actual equipment control or deployed SPC.
The simulated plant receives fixed 07A disturbances and *applied* recipe;
the frozen 07B controller only receives completed run's measurement and recipe.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import inspect

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from r2r_controller import ControllerConfig, R2REWMAController

OPEN_CSV = "07a_open_loop_simulated_runs.csv"
OPEN_JSON = "07a_simulation_scenario.json"
CTRL_JSON = "07b_controller_manifest.json"
CONTROL_SOURCE = Path(__file__).with_name("r2r_controller.py")
CONTROL_COLUMNS = ["run", "setpoint_applied", "target_MRR_simulated",
                   "actual_MRR_simulated", "measured_MRR_simulated",
                   "measured_error_from_target",
                   "exogenous_drift_hidden_from_controller",
                   "process_noise_hidden_from_controller",
                   "metrology_noise_hidden_from_controller"]
PREFIX = "07c_"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verified_inputs(folder: Path):
    """Verify identical previously published 07A inputs and frozen 07B controller.

    Does not calculate closed-loop outcomes; appropriate for --preflight.
    """
    paths = [folder / OPEN_CSV, folder / OPEN_JSON, folder / CTRL_JSON, CONTROL_SOURCE]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing required file: {path}")
    manifest = json.loads((folder / CTRL_JSON).read_text(encoding="utf-8"))
    scenario = json.loads((folder / OPEN_JSON).read_text(encoding="utf-8"))
    check = manifest["open_loop_input_audit"]
    if sha256_file(CONTROL_SOURCE) != manifest["controller_source_sha256"]:
        raise ValueError("Frozen 07B r2r_controller.py source checksum mismatch: STOP")
    if sha256_file(folder / OPEN_CSV) != check["sha256_step07a_csv"]:
        raise ValueError("Step07A CSV changed after controller freeze: STOP")
    if sha256_file(folder / OPEN_JSON) != check["sha256_step07a_json"]:
        raise ValueError("Step07A scenario changed after controller freeze: STOP")
    cfg = ControllerConfig(**manifest["controller_config"])
    if asdict(cfg) != manifest["controller_config"]:
        raise ValueError("Frozen 07B controller configuration is inconsistent")
    if set(inspect.signature(R2REWMAController.observe_completed_run).parameters) != {
            "self", "run", "measured_mrr", "applied_setpoint"}:
        raise ValueError("Controller interface differs from Step07B's fixed contract")

    df = pd.read_csv(folder / OPEN_CSV)
    if list(df.columns) != CONTROL_COLUMNS:
        raise ValueError("Unexpected Step07A schema")
    if len(df) != 120 or not np.array_equal(df["run"].to_numpy(), np.arange(1, 121)):
        raise ValueError("Step07A must have 120 consecutive simulation runs")
    if df.isna().any().any() or not np.isfinite(df.to_numpy(dtype=float)).all():
        raise ValueError("Step07A has missing or nonfinite values")
    if scenario["n_runs"] != 120 or scenario["feedback_control"] != "OFF (fixed setpoint); controller to be added later":
        raise ValueError("Not the expected original open-loop scenario")
    target = float(scenario["target_MRR"])
    nominal = float(scenario["nominal_setpoint"])
    true_gain = float(scenario["hypothetical_true_gain_MRR_per_setpoint_unit"])
    if abs(target - cfg.target_mrr) > 1e-10 or abs(nominal - cfg.nominal_setpoint) > 1e-10:
        raise ValueError("07A plant and frozen controller target/nominal mismatch")
    if abs(true_gain - cfg.estimated_gain) < 1e-12:
        raise ValueError("Expected separate plant gain and assumed controller gain")
    if not np.allclose(df["setpoint_applied"], nominal, atol=1e-12, rtol=0):
        raise ValueError("Step07A does not maintain fixed recipe")
    if not np.allclose(df["target_MRR_simulated"], target, atol=1e-12, rtol=0):
        raise ValueError("Target changed in Step07A")
    actual = (target + true_gain * (df["setpoint_applied"].to_numpy() - nominal)
              + df["exogenous_drift_hidden_from_controller"].to_numpy()
              + df["process_noise_hidden_from_controller"].to_numpy())
    measured = actual + df["metrology_noise_hidden_from_controller"].to_numpy()
    if not np.allclose(actual, df["actual_MRR_simulated"].to_numpy(), atol=5e-7, rtol=0):
        raise ValueError("Step07A plant equation inconsistent")
    if not np.allclose(measured, df["measured_MRR_simulated"].to_numpy(), atol=5e-7, rtol=0):
        raise ValueError("Step07A measured MRR inconsistent")
    if not np.allclose(measured - target, df["measured_error_from_target"].to_numpy(), atol=5e-7, rtol=0):
        raise ValueError("Step07A error column inconsistent")
    observed_mae = float(np.mean(np.abs(measured - target)))
    if abs(observed_mae - float(check["open_loop_mae_verified"])) > 1e-5:
        raise ValueError("Step07A MAE changed")
    return df, scenario, manifest, cfg


def simulate_with_frozen_controller(open_loop: pd.DataFrame, *, true_gain: float,
                                    cfg: ControllerConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use the same disturbances/noises, but a causal feedback recipe.

    IMPORTANT: hidden noise and the plant's true gain are confined to this
    simulator function. Controller.observe_completed_run receives only the
    completed run number, already measured MRR, and applied setpoint.
    """
    controller = R2REWMAController(cfg)
    records, actions = [], []
    for row in open_loop.itertuples(index=False):
        run = int(row.run)
        recipe = float(controller.next_setpoint)
        drift = float(row.exogenous_drift_hidden_from_controller)
        process_noise = float(row.process_noise_hidden_from_controller)
        metrology_noise = float(row.metrology_noise_hidden_from_controller)
        # Simulator's assumed physical response. Not passed to controller.
        actual = (cfg.target_mrr + true_gain * (recipe - cfg.nominal_setpoint)
                  + drift + process_noise)
        measured = actual + metrology_noise
        records.append({
            "run": run, "setpoint_applied": recipe,
            "target_MRR_simulated": cfg.target_mrr,
            "actual_MRR_simulated": actual, "measured_MRR_simulated": measured,
            "measured_error_from_target": measured - cfg.target_mrr,
            "exogenous_drift_hidden_from_controller": drift,
            "process_noise_hidden_from_controller": process_noise,
            "metrology_noise_hidden_from_controller": metrology_noise,
        })
        # Update happens AFTER the current run finishes. Recommendation is for run+1.
        evt = controller.observe_completed_run(
            run=run, measured_mrr=measured, applied_setpoint=recipe)
        actions.append(evt)
    return pd.DataFrame.from_records(records, columns=CONTROL_COLUMNS), pd.DataFrame(actions)


def assert_paired_replay(open_loop: pd.DataFrame, controlled: pd.DataFrame, cfg: ControllerConfig, true_gain: float):
    if len(open_loop) != len(controlled):
        raise ValueError("Simulation lengths differ")
    for c in ["run", "target_MRR_simulated", "exogenous_drift_hidden_from_controller",
              "process_noise_hidden_from_controller", "metrology_noise_hidden_from_controller"]:
        if not np.allclose(open_loop[c], controlled[c], rtol=0, atol=1e-12):
            raise ValueError(f"Unpaired scenario! {c} changed")
    if abs(controlled.loc[0, "setpoint_applied"] - cfg.nominal_setpoint) > 1e-12:
        raise ValueError("First recipe must be fixed before any measurement")
    # Only the assumed recipe effect can differ between the same-world runs.
    expected_delta = true_gain * (controlled["setpoint_applied"] - open_loop["setpoint_applied"])
    for c in ["actual_MRR_simulated", "measured_MRR_simulated"]:
        if not np.allclose(controlled[c] - open_loop[c], expected_delta, atol=1e-6, rtol=0):
            raise ValueError(f"Differences in {c} are not from recipe change alone")
    if not controlled["setpoint_applied"].between(cfg.min_setpoint - 1e-10, cfg.max_setpoint + 1e-10).all():
        raise ValueError("Setpoint bounds violated")
    diffs = controlled["setpoint_applied"].diff().dropna()
    if (diffs.abs() > cfg.max_step_change + 1e-10).any():
        raise ValueError("Maximum adjustment step violated")


def summarize(df: pd.DataFrame, *, case: str, cfg: ControllerConfig) -> dict:
    measured = df["measured_MRR_simulated"].to_numpy(dtype=float)
    e = measured - cfg.target_mrr
    recipes = df["setpoint_applied"].to_numpy(dtype=float)
    changes = np.diff(recipes)
    return {
        "scenario": case,
        "n_runs": len(df),
        "target_MRR": cfg.target_mrr,
        "measured_MRR_mean": float(np.mean(measured)),
        "mean_signed_error": float(np.mean(e)),
        "MAE_from_target": float(np.mean(np.abs(e))),
        "RMSE_from_target": float(np.sqrt(np.mean(np.square(e)))),
        "last30_mean_signed_error": float(np.mean(e[-30:])),
        "last30_MAE_from_target": float(np.mean(np.abs(e[-30:]))),
        "setpoint_change_count": int(np.count_nonzero(np.abs(changes) > 1e-10)),
        "total_absolute_adjustment": float(np.sum(np.abs(changes))),
        "maximum_single_adjustment": float(np.max(np.abs(changes))) if len(changes) else 0.0,
        "minimum_applied_setpoint": float(np.min(recipes)),
        "maximum_applied_setpoint": float(np.max(recipes)),
        "setpoint_offset_from_nominal_mean_absolute": float(np.mean(np.abs(recipes - cfg.nominal_setpoint))),
    }


def make_comparison(open_metrics: dict, closed_metrics: dict) -> dict:
    """Computed reporting only. No choices, tuning or controller update."""
    metrics = ["MAE_from_target", "RMSE_from_target", "mean_signed_error",
               "last30_mean_signed_error", "last30_MAE_from_target",
               "setpoint_change_count", "total_absolute_adjustment",
               "maximum_single_adjustment", "setpoint_offset_from_nominal_mean_absolute"]
    out = []
    for key in metrics:
        a, b = float(open_metrics[key]), float(closed_metrics[key])
        out.append({"metric": key, "open_loop": a, "closed_loop": b,
                    "closed_minus_open": b - a,
                    "relative_reduction_pct": (100 * (a - b) / a if a > 0 and key in
                        {"MAE_from_target", "RMSE_from_target", "last30_MAE_from_target"} else float("nan"))})
    return {"metrics": out}


def plot_comparison(open_loop: pd.DataFrame, controlled: pd.DataFrame, cfg: ControllerConfig, outdir: Path) -> None:
    figdir = outdir / "figures"
    figdir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(open_loop.run, open_loop.measured_MRR_simulated, lw=1.2, label="Open-loop measured MRR")
    ax.plot(controlled.run, controlled.measured_MRR_simulated, lw=1.2, label="EWMA feedback measured MRR")
    ax.axhline(cfg.target_mrr, color="black", ls="--", lw=1.3, label="Hypothetical target")
    ax.axvline(60.5, ls=":", color="gray", alpha=0.7, label="Artificial disturbance step")
    ax.set(title="Step 07C | Paired 120-run MRR (SIMULATION ONLY)", xlabel="Simulated run (not fab production order)",
           ylabel="Hypothetical measured MRR (arbitrary units)")
    ax.grid(alpha=0.2); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(figdir / "07c_open_vs_closed_mrr.png", dpi=165); plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(open_loop.run, open_loop.setpoint_applied, label="Open-loop fixed setpoint", lw=1.6)
    ax.step(controlled.run, controlled.setpoint_applied, where="post", label="EWMA applied setpoint", lw=1.4)
    ax.axhline(cfg.min_setpoint, color="gray", ls=":", alpha=.6, label="Assumed setpoint bounds")
    ax.axhline(cfg.max_setpoint, color="gray", ls=":", alpha=.6)
    ax.set(title="Step 07C | Applied setpoint after previous-run feedback (SIMULATION ONLY)",
           xlabel="Simulated run", ylabel="Hypothetical setpoint units")
    ax.grid(alpha=0.2); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(figdir / "07c_open_vs_closed_setpoints.png", dpi=165); plt.close(fig)


def _no_existing_outputs(folder: Path):
    """Refuse to overwrite a previously seen outcome, even if marker missing."""
    names = ["07c_closed_loop_simulated_runs.csv", "07c_open_vs_closed_comparison.csv",
             "07c_feedback_actions.csv", "07c_method_and_integrity.json",
             "07c_FINALIZED.json"]
    names += ["figures/07c_open_vs_closed_mrr.png", "figures/07c_open_vs_closed_setpoints.png"]
    already = [str(folder / x) for x in names if (folder / x).exists()]
    if already:
        raise FileExistsError("Step07C may only run once. Existing output(s): " + ", ".join(already))


def main() -> None:
    parser = argparse.ArgumentParser(description="Once-only frozen EWMA demonstration")
    parser.add_argument("--simulation-dir", type=Path, default=Path("outputs/simulation"))
    parser.add_argument("--preflight", action="store_true", help="Verify inputs & code; NO closed-loop performance")
    args = parser.parse_args()
    folder = args.simulation_dir
    open_loop, scenario, manifest, cfg = verified_inputs(folder)
    print("PASS: 07A original 120-run simulation and 07B controller/config SHA256 hashes match frozen manifest")
    print(f"Controller: EWMA alpha={cfg.ewma_alpha:.2f}, estimated gain={cfg.estimated_gain:.2f}, "
          f"max step={cfg.max_step_change:.2f}, bounds=[{cfg.min_setpoint:g}, {cfg.max_setpoint:g}]")
    if args.preflight:
        print("PREFLIGHT ONLY: 07C closed-loop outcomes NOT computed or inspected.")
        return
    _no_existing_outputs(folder)
    true_gain = float(scenario["hypothetical_true_gain_MRR_per_setpoint_unit"])
    controlled, actions = simulate_with_frozen_controller(open_loop, true_gain=true_gain, cfg=cfg)
    assert_paired_replay(open_loop, controlled, cfg, true_gain)
    if not np.allclose(controlled.setpoint_applied.iloc[1:].to_numpy(),
                       actions.setpoint_for_next_run.iloc[:-1].to_numpy(), rtol=0, atol=1e-9):
        raise ValueError("Controller recommendation applied on incorrect run")
    om = summarize(open_loop, case="open_loop_fixed_recipe", cfg=cfg)
    cm = summarize(controlled, case="closed_loop_frozen_ewma", cfg=cfg)
    if abs(om["MAE_from_target"] - float(scenario["metrics"]["MAE_from_target"])) > 1e-5:
        raise ValueError("Open-loop metrics changed")
    if abs(om["RMSE_from_target"] - float(scenario["metrics"]["RMSE_from_target"])) > 1e-5:
        raise ValueError("Open-loop RMSE changed")
    if len(actions) != 120:
        raise ValueError("Missing control trace")
    summary = pd.DataFrame(make_comparison(om, cm)["metrics"])
    notes = {
        "status": "PAIRED CLOSED-LOOP SIMULATION; ARTIFICIAL 120 RUNS; NOT A FAB RESULT",
        "controller_code_SHA256_verified": manifest["controller_source_sha256"],
        "07a_input_SHA256_verified": manifest["open_loop_input_audit"],
        "controller_config_frozen_from_07b": asdict(cfg),
        "simulator_true_gain_is_hidden_from_controller": True,
        "causal_protocol": "Run t uses recipe chosen after t-1; controller only sees measured MRR of finished t and applied recipe",
        "paired_noise_drift_equal": True,
        "no_posthoc_parameter_tuning": True,
        "metrics_definition": "Against hypothetical target 75, based on simulated MEASURED MRR; not yield, Cp/Cpk, official OOC",
        "limits": [
            "Controller and plant are fictional; no actual Micron/PHM process settings or quality targets",
            "Identical artificial drift/noise replay controls paired comparison, not prospective factory validation",
            "Current model assumes immediate metrology after each run and fixed positive process gain",
            "The controller cannot see hidden drift, process/metrology noise, true plant gain or future measurements",
            "Setpoint movement is not cost-free; report number/magnitude of changes and constrained actions",
            "No claims about actual tool safety, device yield or process capability",
        ],
        "open_loop_metrics": om,
        "closed_loop_metrics": cm,
        "constrained_suggestions_count_including_run120_unused": int(actions.recipe_was_constrained.sum()),
        "first_119_constrained_suggestions_applied_next_run": int(actions.recipe_was_constrained.iloc[:-1].sum()),
    }
    folder.mkdir(parents=True, exist_ok=True)
    controlled.to_csv(folder / "07c_closed_loop_simulated_runs.csv", index=False, float_format="%.8f")
    actions.to_csv(folder / "07c_feedback_actions.csv", index=False, float_format="%.8f")
    summary.to_csv(folder / "07c_open_vs_closed_comparison.csv", index=False, float_format="%.8f")
    (folder / "07c_method_and_integrity.json").write_text(json.dumps(notes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plot_comparison(open_loop, controlled, cfg, folder)
    finalized = {
        "status": "07C FINALIZED: ONE FROZEN PAIRED EVALUATION; DO NOT TUNE AND RE-RUN FOR BETTER RESULT",
        "controller_source_SHA256": manifest["controller_source_sha256"],
        "outputs": ["07c_open_vs_closed_comparison.csv", "07c_closed_loop_simulated_runs.csv", "07c_feedback_actions.csv"],
        "no_real_fab_metrics": True,
    }
    (folder / "07c_FINALIZED.json").write_text(json.dumps(finalized, indent=2) + "\n", encoding="utf-8")
    print(f"OPEN   : MAE={om['MAE_from_target']:.3f}, RMSE={om['RMSE_from_target']:.3f}, last30 bias={om['last30_mean_signed_error']:+.3f}")
    print(f"EWMA   : MAE={cm['MAE_from_target']:.3f}, RMSE={cm['RMSE_from_target']:.3f}, last30 bias={cm['last30_mean_signed_error']:+.3f}")
    print(f"Control cost: {cm['setpoint_change_count']} applied changes, total abs movement={cm['total_absolute_adjustment']:.3f}")
    print(f"PASS: paired disturbances/noises match exactly; frozen one-run-delayed controller; results FINALIZED in {folder.resolve()}")
    print("SIMULATION ONLY. No real manufacturing improvement demonstrated.")


if __name__ == "__main__":
    main()
