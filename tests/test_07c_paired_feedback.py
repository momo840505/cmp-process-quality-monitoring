"""Step 07C unit tests use new FICTIONAL fixtures only, not user's 07A outcome stream."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
from r2r_controller import ControllerConfig, R2REWMAController

spec = importlib.util.spec_from_file_location("step07c_paired", SRC / "07c_compare_closed_loop.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def fake_open(n=120):
    """Different fictitious drift/noise scenario used solely for code tests."""
    r = np.arange(1, n + 1)
    drift = 0.01 * r + 0.5 * (r > n/2)
    process = np.sin(r) * 0.2
    metrology = np.cos(r) * 0.07
    target = 75.0
    actual = target + drift + process
    measured = actual + metrology
    return pd.DataFrame({
        "run": r, "setpoint_applied": 50.0, "target_MRR_simulated": target,
        "actual_MRR_simulated": actual, "measured_MRR_simulated": measured,
        "measured_error_from_target": measured - target,
        "exogenous_drift_hidden_from_controller": drift,
        "process_noise_hidden_from_controller": process,
        "metrology_noise_hidden_from_controller": metrology,
    })[mod.CONTROL_COLUMNS]


class TestPairedR2R(unittest.TestCase):
    def setUp(self):
        self.open = fake_open()
        self.cfg = ControllerConfig()

    def test_same_noise_and_frozen_one_run_feedback(self):
        controlled, actions = mod.simulate_with_frozen_controller(self.open, true_gain=.65, cfg=self.cfg)
        mod.assert_paired_replay(self.open, controlled, self.cfg, true_gain=.65)
        self.assertEqual(len(controlled), 120)
        self.assertEqual(len(actions), 120)
        self.assertAlmostEqual(controlled.setpoint_applied.iloc[0], 50.)
        np.testing.assert_allclose(controlled.setpoint_applied.to_numpy()[1:],
                                   actions.setpoint_for_next_run.to_numpy()[:-1], atol=1e-10)

    def test_no_future_information_leaks_backwards(self):
        controlled, _ = mod.simulate_with_frozen_controller(self.open, true_gain=.65, cfg=self.cfg)
        changed = self.open.copy()
        # At run=61 future disturbance is radically changed. Earlier RECIPES remain identical.
        changed.loc[changed.run == 61, "exogenous_drift_hidden_from_controller"] += 100
        controlled_changed, _ = mod.simulate_with_frozen_controller(changed, true_gain=.65, cfg=self.cfg)
        np.testing.assert_allclose(controlled.setpoint_applied.iloc[:61],
                                   controlled_changed.setpoint_applied.iloc[:61], atol=1e-10)
        # Only run 62 onwards can react to information from run 61.
        self.assertNotAlmostEqual(controlled.setpoint_applied.iloc[61], controlled_changed.setpoint_applied.iloc[61])

    def test_corrupted_paired_simulation_is_rejected(self):
        controlled, _ = mod.simulate_with_frozen_controller(self.open, true_gain=.65, cfg=self.cfg)
        controlled.loc[10, "process_noise_hidden_from_controller"] += .3
        with self.assertRaisesRegex(ValueError, "Unpaired"):
            mod.assert_paired_replay(self.open, controlled, self.cfg, .65)

    def test_ewma_rate_and_absolute_limits(self):
        out, action = mod.simulate_with_frozen_controller(self.open, true_gain=.65, cfg=self.cfg)
        self.assertTrue(out.setpoint_applied.between(42, 58).all())
        self.assertLessEqual(out.setpoint_applied.diff().abs().max(), 1.5 + 1e-12)
        self.assertTrue(action.setpoint_for_next_run.between(42, 58).all())

    def test_metric_values_and_adjustment_cost(self):
        closed, _ = mod.simulate_with_frozen_controller(self.open, true_gain=.65, cfg=self.cfg)
        m = mod.summarize(closed, case="closed", cfg=self.cfg)
        errs = closed.measured_MRR_simulated - 75
        self.assertAlmostEqual(m["MAE_from_target"], errs.abs().mean())
        self.assertAlmostEqual(m["RMSE_from_target"], np.sqrt((errs ** 2).mean()))
        self.assertAlmostEqual(m["last30_mean_signed_error"], errs.tail(30).mean())
        self.assertEqual(m["setpoint_change_count"], int(np.count_nonzero(closed.setpoint_applied.diff().fillna(0).abs() > 1e-10)))

    def test_frozen_input_hash_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            self.open.to_csv(folder / mod.OPEN_CSV, index=False, float_format="%.8f")
            # Match real 07A CSV rounding for baseline equation tolerance.
            sim = {
                "n_runs": 120, "target_MRR": 75., "nominal_setpoint": 50.,
                "hypothetical_true_gain_MRR_per_setpoint_unit": .65,
                "feedback_control": "OFF (fixed setpoint); controller to be added later",
                "metrics": {"MAE_from_target": mod.summarize(self.open, case="open", cfg=self.cfg)["MAE_from_target"]},
            }
            (folder / mod.OPEN_JSON).write_text(json.dumps(sim), encoding="utf-8")
            manifest = {
                "controller_config": asdict(self.cfg),
                "controller_source_sha256": mod.sha256_file(mod.CONTROL_SOURCE),
                "open_loop_input_audit": {
                    "sha256_step07a_csv": mod.sha256_file(folder / mod.OPEN_CSV),
                    "sha256_step07a_json": mod.sha256_file(folder / mod.OPEN_JSON),
                    "open_loop_mae_verified": sim["metrics"]["MAE_from_target"],
                },
            }
            (folder / mod.CTRL_JSON).write_text(json.dumps(manifest), encoding="utf-8")
            _df, _scenario, _manifest, _cfg = mod.verified_inputs(folder)
            self.assertEqual(len(_df), 120)
            (folder / "07c_FINALIZED.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                mod._no_existing_outputs(folder)
            (folder / "07c_FINALIZED.json").unlink()
            (folder / mod.OPEN_JSON).write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed"):
                mod.verified_inputs(folder)


if __name__ == "__main__":
    unittest.main()
