"""Tests for one-step causal educational R2R controller."""
from __future__ import annotations

import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from r2r_controller import ControllerConfig, R2REWMAController


class TestR2RController(unittest.TestCase):
    def test_first_run_is_fixed_and_change_only_next_run(self):
        con = R2REWMAController(ControllerConfig())
        self.assertEqual(con.initial_setpoint, 50.0)
        self.assertEqual(con.next_setpoint, 50.0)
        out = con.observe_completed_run(run=1, applied_setpoint=50.0, measured_mrr=78.0)
        self.assertEqual(out['completed_run'], 1)
        self.assertEqual(out['setpoint_applied_completed_run'], 50)
        self.assertEqual(out['setpoint_for_next_run'], 48.5)

    def test_expected_ewma_bias_math(self):
        con = R2REWMAController(ControllerConfig())
        evt = con.observe_completed_run(run=1, applied_setpoint=50.0, measured_mrr=78.0)
        self.assertAlmostEqual(evt['raw_bias_observation'], 3.0)
        self.assertAlmostEqual(evt['ewma_bias_estimate_after_measurement'], 1.05)
        self.assertAlmostEqual(evt['desired_setpoint_for_next_run'], 48.25)
        self.assertAlmostEqual(evt['setpoint_for_next_run'], 48.5)

    def test_rate_and_absolute_limits(self):
        con = R2REWMAController(ControllerConfig())
        for run in range(1, 35):
            applied = con.next_setpoint
            out = con.observe_completed_run(run=run, applied_setpoint=applied, measured_mrr=110.0)
            self.assertLessEqual(abs(out['step_change_for_next_run']), 1.5 + 1e-8)
            self.assertTrue(42.0 <= out['setpoint_for_next_run'] <= 58.0)
        self.assertAlmostEqual(con.next_setpoint, 42.0)

    def test_no_future_measurement_or_hidden_fields_in_interface(self):
        sig = inspect.signature(R2REWMAController.observe_completed_run)
        names = set(sig.parameters)
        self.assertEqual(names, {'self', 'run', 'measured_mrr', 'applied_setpoint'})
        self.assertNotIn('hidden_drift', names)
        con = R2REWMAController(ControllerConfig())
        with self.assertRaises(TypeError):
            con.observe_completed_run(run=1, measured_mrr=75.0, applied_setpoint=50.0, hidden_drift=10.0)

    def test_order_and_applied_recipe_must_match(self):
        con = R2REWMAController(ControllerConfig())
        with self.assertRaises(ValueError):
            con.observe_completed_run(run=2, measured_mrr=75.0, applied_setpoint=50.0)
        con.observe_completed_run(run=1, measured_mrr=75.0, applied_setpoint=50.0)
        with self.assertRaises(ValueError):
            con.observe_completed_run(run=2, measured_mrr=76.0, applied_setpoint=49.0)
        with self.assertRaises(ValueError):
            con.observe_completed_run(run=1, measured_mrr=75.0, applied_setpoint=50.0)

    def test_frozen_manifest_cannot_be_overwritten(self):
        import importlib.util
        modpath = PROJECT_ROOT / 'src' / '07b_design_r2r_controller.py'
        spec = importlib.util.spec_from_file_location('step07b_entry', modpath)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'manifest.json'
            mod.write_frozen_manifest(p, {'gain': 0.60})
            mod.write_frozen_manifest(p, {'gain': 0.60})
            with self.assertRaises(RuntimeError):
                mod.write_frozen_manifest(p, {'gain': 0.65})
            self.assertEqual(json.loads(p.read_text())['gain'], 0.60)


if __name__ == '__main__':
    unittest.main()
