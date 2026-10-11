"""Step07B: freeze and demonstrate a CAUSAL hypothetical R2R controller.

This script DOES NOT run the proposed controller over 07A's 120-run plant.
That closed-loop comparison belongs to the separate, predeclared Step07C.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from r2r_controller import ControllerConfig, R2REWMAController

CONTROLLER_SOURCE = Path(__file__).with_name("r2r_controller.py")
OPEN_CSV = "07a_open_loop_simulated_runs.csv"
OPEN_JSON = "07a_simulation_scenario.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_open_loop(folder: Path) -> dict:
    """Only audits Step07A input integrity; NOT used to tune the controller."""
    csv_path = folder / OPEN_CSV
    json_path = folder / OPEN_JSON
    if not csv_path.is_file() or not json_path.is_file():
        raise FileNotFoundError(f"Missing Step07A input in {folder}: run Step 07A first")
    df = pd.read_csv(csv_path)
    scen = json.loads(json_path.read_text(encoding="utf-8"))
    cols = {"run", "setpoint_applied", "target_MRR_simulated", "actual_MRR_simulated",
            "measured_MRR_simulated", "measured_error_from_target",
            "exogenous_drift_hidden_from_controller", "process_noise_hidden_from_controller",
            "metrology_noise_hidden_from_controller"}
    if set(df.columns) != cols:
        raise ValueError("Step07A sensor/simulator schema changed")
    if len(df) != 120 or not np.array_equal(df.run.values, np.arange(1, 121)):
        raise ValueError("Step07A must contain exactly runs 1..120")
    if df.isna().any().any() or not np.isfinite(df.select_dtypes(include="number").to_numpy()).all():
        raise ValueError("Step07A contains NaN/Inf")
    if not (df.setpoint_applied == 50.0).all() or not (df.target_MRR_simulated == 75.0).all():
        raise ValueError("Step07A does not represent expected FIXED, no-control scenario")
    true_gain = float(scen["hypothetical_true_gain_MRR_per_setpoint_unit"])
    p_calc = df.target_MRR_simulated + true_gain * (df.setpoint_applied - 50.0) + df.exogenous_drift_hidden_from_controller + df.process_noise_hidden_from_controller
    y_calc = df.actual_MRR_simulated + df.metrology_noise_hidden_from_controller
    if not np.allclose(df.actual_MRR_simulated, p_calc, atol=5e-7, rtol=0) or not np.allclose(df.measured_MRR_simulated, y_calc, atol=5e-7, rtol=0):
        raise ValueError("Step07A equations do not reconcile")
    e = df.measured_MRR_simulated.values - 75.0
    calc = {"MAE_from_target": float(np.mean(np.abs(e))),
            "RMSE_from_target": float(np.sqrt(np.mean(e ** 2))),
            "last_30_mean_signed_error": float(np.mean(e[-30:]))}
    for key, value in calc.items():
        if abs(value - float(scen["metrics"][key])) > 1e-5:
            raise ValueError("Step07A reported metrics do not reconcile")
    return {
        "file_rows_verified": len(df),
        "equations_verified": True,
        "fixed_setpoint_verified": True,
        "open_loop_mae_verified": calc["MAE_from_target"],
        "sha256_step07a_csv": sha256_file(csv_path),
        "sha256_step07a_json": sha256_file(json_path),
    }


def produce_toy_demo(config: ControllerConfig) -> pd.DataFrame:
    """Arbitrary illustration: supplied measurements are not from a plant model.

    No 'closed loop improvement' can be inferred from this standalone example.
    """
    given_measurements = (78.0, 77.1, 76.8, 74.6, 75.2, 75.8)
    controller = R2REWMAController(config)
    events = []
    for run, measured in enumerate(given_measurements, start=1):
        applied = controller.next_setpoint
        event = controller.observe_completed_run(run=run, measured_mrr=measured,
                                                  applied_setpoint=applied)
        events.append(event)
    return pd.DataFrame(events)


def write_frozen_manifest(path: Path, expected: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != expected:
            raise RuntimeError("A DIFFERENT frozen controller manifest exists. Do not silently replace it.")
        return
    path.write_text(json.dumps(expected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def plot_toy(df: pd.DataFrame, config: ControllerConfig, dest: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    t = df.completed_run.values
    axes[0].plot(t, df.measured_mrr_completed_run, marker="o", label="Illustrative given measurement")
    axes[0].axhline(config.target_mrr, linestyle="--", label="Hypothetical target")
    axes[0].set_ylabel("Made-up MRR")
    axes[0].legend(fontsize=9)
    axes[1].step(t, df.setpoint_applied_completed_run, where="mid", marker="o", label="Applied to run t")
    axes[1].step(t+1, df.setpoint_for_next_run, where="mid", marker="s", label="Planned for run t+1")
    axes[1].axhline(config.nominal_setpoint, linestyle="--", color="gray", alpha=.7)
    axes[1].set_xlabel("Toy completed run index (NOT the Step07A plant)")
    axes[1].set_ylabel("Hypothetical setpoint")
    axes[1].legend(fontsize=9)
    for ax in axes:
        ax.grid(alpha=.25)
    fig.suptitle("Step 07B: one-run delayed controller logic only; NOT an efficacy test")
    fig.tight_layout(rect=(0, 0, 1, .96))
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulation-dir", type=Path, default=Path("outputs") / "simulation")
    args = parser.parse_args()
    folder = args.simulation_dir
    verified = audit_open_loop(folder)
    config = ControllerConfig()
    # Controller gain is a PRE-CHOSEN hypothetical calibration assumption, NOT
    # read from the hidden true-gain parameter inside the Step07A scenario.
    assert config.estimated_gain != 0.65
    demo = produce_toy_demo(config)
    folder.mkdir(parents=True, exist_ok=True)
    manifest = {
        "classification": "SIMULATED / EDUCATIONAL / NO real fab controller",
        "step": "07B: freeze the controller BEFORE closed-loop evaluation in 07C",
        "controller_kind": "EWMA disturbance compensation; 1-run measurement delay",
        "controller_config": asdict(config),
        "controller_source_sha256": sha256_file(CONTROLLER_SOURCE),
        "open_loop_input_audit": verified,
        "controller_can_observe": ["completed_run", "measured_mrr_completed_run", "setpoint_applied_completed_run"],
        "controller_must_not_observe": ["exogenous_drift_hidden_from_controller", "process_noise_hidden_from_controller", "metrology_noise_hidden_from_controller", "actual_MRR_simulated", "future_measured_mrr"],
        "toy_demo_warning": "Toy input measurements are prewritten examples, NOT responses to controller actions and NOT an effectiveness test.",
        "future_07c_protocol": "Freeze config and code hash, reuse identical Step07A hidden disturbance/noise streams, compare controlled and open-loop MAE/RMSE, final-30 bias, and count/control costs. No gain tuning after observing 07C results.",
        "provenance_warning": "07A drift/noise and any performance outcome are artificial, not PHM data, Micron MRR, or safety-validated limits.",
    }
    write_frozen_manifest(folder / "07b_controller_manifest.json", manifest)
    demo.to_csv(folder / "07b_toy_controller_trace.csv", index=False, float_format="%.8f")
    plot_toy(demo, config, folder / "figures" / "07b_one_step_feedback_logic.png")
    assert (abs(demo.step_change_for_next_run) <= config.max_step_change + 1e-8).all()
    assert demo.setpoint_for_next_run.between(config.min_setpoint, config.max_setpoint).all()
    assert len(demo) == 6
    print("PASS: Step07A original 120-run simulation audited (fixed setpoint and equations reconciled)")
    print("Frozen educational R2R controller: EWMA alpha=0.35, assumed gain=0.60, step<=1.50, bounds=[42, 58]")
    print("Toy 6-run causal demonstration complete; the applied recipe changes ONLY on NEXT run")
    print("No closed-loop performance evaluated; NO 07A drift/noise was given to the controller")
    print(f"Saved manifest, toy trace, and a figure under: {folder.resolve()}")


if __name__ == "__main__":
    main()
