"""Simulate a fictional CMP process under a fixed recipe (R2R project, phase 07A).

This is NOT the PHM 2016 process and NOT a reconstruction of its parameters.
The hypothetical material-removal-rate units and model parameters are illustrative.
Run this before testing any feedback controller in Step 07B/07C.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEED = 20261011
N_RUNS = 120
TARGET = 75.0
BASE_SETPOINT = 50.0
TRUE_GAIN = 0.65
PROCESS_NOISE_SD = 0.65
MEASUREMENT_NOISE_SD = 0.35


def create_exogenous(n_runs: int = N_RUNS, seed: int = SEED) -> pd.DataFrame:
    """Create exogenous drift and noises, independent of any future controller.

    These columns are available to the simulator/analyst, but a controller must
    NEVER use them directly. It will only observe delayed measured_MRR.
    """
    if n_runs < 4:
        raise ValueError("n_runs must be >= 4")
    rng = np.random.default_rng(seed)
    run = np.arange(1, n_runs + 1)
    drift = 0.025 * (run - 1) + 2.0 * (run > (n_runs // 2)) + 0.4 * np.sin(run / 8.0)
    return pd.DataFrame({
        "run": run,
        "exogenous_drift_hidden_from_controller": drift,
        "process_noise_hidden_from_controller": rng.normal(0.0, PROCESS_NOISE_SD, n_runs),
        "metrology_noise_hidden_from_controller": rng.normal(0.0, MEASUREMENT_NOISE_SD, n_runs),
    })


def simulate_open_loop(exogenous: pd.DataFrame) -> pd.DataFrame:
    """Compute hypothetical MRR under fixed controller input with zero feedback."""
    out = exogenous.copy()
    out["setpoint_applied"] = BASE_SETPOINT
    out["target_MRR_simulated"] = TARGET
    out["actual_MRR_simulated"] = (
        TARGET
        + TRUE_GAIN * (out["setpoint_applied"] - BASE_SETPOINT)
        + out["exogenous_drift_hidden_from_controller"]
        + out["process_noise_hidden_from_controller"]
    )
    out["measured_MRR_simulated"] = (
        out["actual_MRR_simulated"] + out["metrology_noise_hidden_from_controller"]
    )
    out["measured_error_from_target"] = out["measured_MRR_simulated"] - TARGET
    return out[
        ["run", "setpoint_applied", "target_MRR_simulated", "actual_MRR_simulated",
         "measured_MRR_simulated", "measured_error_from_target",
         "exogenous_drift_hidden_from_controller", "process_noise_hidden_from_controller",
         "metrology_noise_hidden_from_controller"]
    ]


def compute_metrics(df: pd.DataFrame) -> dict:
    delta = df["measured_error_from_target"].to_numpy(dtype=float)
    return {
        "n_runs": int(len(df)),
        "target_MRR_simulated": TARGET,
        "fixed_setpoint": BASE_SETPOINT,
        "mean_measured_MRR": float(df["measured_MRR_simulated"].mean()),
        "mean_signed_error": float(delta.mean()),
        "MAE_from_target": float(np.abs(delta).mean()),
        "RMSE_from_target": float(np.sqrt(np.mean(delta ** 2))),
        "last_30_mean_signed_error": float(delta[-min(30, len(delta)):].mean()),
        "setpoint_changes": int(np.count_nonzero(np.diff(df["setpoint_applied"]))),
    }


def write_outputs(output_dir: Path, df: pd.DataFrame, metrics: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    figs = output_dir / "figures"
    figs.mkdir(exist_ok=True)
    df.to_csv(output_dir / "07a_open_loop_simulated_runs.csv", index=False, float_format="%.8f")

    scenario = {
        "classification": "SIMULATED / HYPOTHETICAL; not PHM data or Micron operating conditions",
        "seed": SEED,
        "n_runs": int(len(df)),
        "target_MRR": TARGET,
        "nominal_setpoint": BASE_SETPOINT,
        "hypothetical_true_gain_MRR_per_setpoint_unit": TRUE_GAIN,
        "hypothetical_drift": "0.025*(run-1) + 2.0 after midpoint + 0.4*sin(run/8)",
        "process_noise_standard_deviation": PROCESS_NOISE_SD,
        "metrology_noise_standard_deviation": MEASUREMENT_NOISE_SD,
        "feedback_control": "OFF (fixed setpoint); controller to be added later",
        "measurement_timing": "End of each run. Later controller may use previous measured run only.",
        "units": "Arbitrary illustrative numerical scale; not calibrated engineering units or fab specs",
        "critical_warning": "Hidden disturbance and noise columns are for simulation analysis only. Never pass them to a proposed controller.",
        "comparability": "Use these identical disturbances/noise sequences in controlled-versus-open-loop comparison later.",
        "metrics": metrics,
    }
    (output_dir / "07a_simulation_scenario.json").write_text(
        json.dumps(scenario, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(df["run"], df["measured_MRR_simulated"], label="Simulated measured MRR", marker=".", markersize=3, linewidth=1)
    ax.axhline(TARGET, color="black", linestyle="--", linewidth=1.4, label="Hypothetical target (75)")
    ax.axvline(len(df) // 2 + 0.5, color="gray", linestyle=":", label="Designed drift step (midpoint)")
    ax.set_xlabel("Simulated run index (not fab chronology)")
    ax.set_ylabel("Simulated MRR (arbitrary units)")
    ax.set_title("Step 07A | Open loop: fixed recipe, no run-to-run feedback")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(figs / "07a_open_loop_quality_drift.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(df["run"], df["setpoint_applied"], linewidth=1.6, label="Applied setpoint")
    ax.set_xlabel("Simulated run index")
    ax.set_ylabel("Hypothetical setpoint units")
    ax.set_title("Step 07A | No control action: setpoint stays fixed")
    ax.set_ylim(BASE_SETPOINT - 2, BASE_SETPOINT + 2)
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figs / "07a_fixed_recipe.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "simulation")
    args = parser.parse_args()
    exog = create_exogenous()
    df = simulate_open_loop(exog)
    metrics = compute_metrics(df)
    assert len(df) == N_RUNS
    assert df["run"].is_unique
    assert (df["setpoint_applied"] == BASE_SETPOINT).all()
    assert df.isna().sum().sum() == 0
    write_outputs(args.output_dir, df, metrics)
    print("PASS: SIMULATION ONLY / 120 hypothetical runs; fixed setpoint; no feedback")
    print("No PHM observations, real tool settings, MRR spec limits, or ML predictions used.")
    print(f"Open-loop target={TARGET:.2f} | measured mean={metrics['mean_measured_MRR']:.3f} | "
          f"MAE={metrics['MAE_from_target']:.3f} | RMSE={metrics['RMSE_from_target']:.3f} | "
          f"last30 bias={metrics['last_30_mean_signed_error']:.3f}")
    print(f"Saved 2 reports + 2 figures to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
