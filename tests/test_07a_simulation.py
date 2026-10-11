"""Tests for a fictional run-to-run process plant, not a real fab model."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src" / "07a_simulate_open_loop.py"
spec = importlib.util.spec_from_file_location("r2r_07a", SRC)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

class TestOpenLoop(unittest.TestCase):
    def test_deterministic_exogenous(self):
        a = module.create_exogenous(30, seed=8)
        b = module.create_exogenous(30, seed=8)
        self.assertTrue(a.equals(b))

    def test_fixed_recipe_does_not_adjust(self):
        d = module.simulate_open_loop(module.create_exogenous())
        self.assertEqual(len(d), 120)
        self.assertEqual(d["setpoint_applied"].nunique(), 1)
        self.assertEqual(module.compute_metrics(d)["setpoint_changes"], 0)

    def test_plant_equations(self):
        d = module.simulate_open_loop(module.create_exogenous())
        predicted = (module.TARGET + module.TRUE_GAIN*(d["setpoint_applied"]-module.BASE_SETPOINT)
                     + d["exogenous_drift_hidden_from_controller"] + d["process_noise_hidden_from_controller"])
        np.testing.assert_allclose(d["actual_MRR_simulated"], predicted)
        np.testing.assert_allclose(d["measured_MRR_simulated"], d["actual_MRR_simulated"]+d["metrology_noise_hidden_from_controller"])

    def test_noise_independent_of_control(self):
        ex = module.create_exogenous()
        self.assertFalse(ex.isna().any().any())
        self.assertEqual(ex.run.nunique(), 120)
        self.assertGreater(ex.exogenous_drift_hidden_from_controller.iloc[-1],ex.exogenous_drift_hidden_from_controller.iloc[0])

if __name__ == "__main__":
    unittest.main()
