"""Read-only dashboard consistency tests; no Streamlit startup needed."""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from src.dashboard_data import read_sources, validate_sources, r2r_relative_gain, relative_mae_gain


class TestDashboardData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = read_sources(PROJECT)

    def test_r2r_paired_data_checked(self):
        self.assertEqual(len(self.d.open_runs), 120)
        self.assertEqual(len(self.d.closed_runs), 120)
        self.assertTrue(np.allclose(self.d.actions.setpoint_for_next_run[:-1], self.d.closed_runs.setpoint_applied[1:]))

    def test_r2r_reported_mae_reduction(self):
        self.assertAlmostEqual(r2r_relative_gain(self.d),74.6017,places=3)

    def test_vm_final_holdout_independent_samples(self):
        unique=self.d.vm_predictions.drop_duplicates(['WAFER_ID','code_family'])
        self.assertEqual(len(unique),174)
        self.assertEqual(len(self.d.vm_predictions),348)
        self.assertEqual(set(self.d.vm_predictions.partition),{'test'})

    def test_vm_improvement_against_baseline(self):
        self.assertAlmostEqual(relative_mae_gain(self.d,'Codes 1-3'),24.733,delta=0.1)
        self.assertAlmostEqual(relative_mae_gain(self.d,'Codes 4-6'),41.85,delta=0.1)

    def test_fdc_review_counts(self):
        self.assertEqual(len(self.d.fdc_cases),11)
        self.assertEqual(len(self.d.fdc_signals),27)
        self.assertEqual(int(self.d.fdc_summary.later_two_or_more_families.sum()),11)

    def test_validator_rejects_mismatched_noise(self):
        changed=self.d.closed_runs.copy()
        changed.loc[0,'process_noise_hidden_from_controller'] += 1
        from dataclasses import replace
        with self.assertRaisesRegex(ValueError,'unequal simulated disturbances'):
            validate_sources(replace(self.d,closed_runs=changed))

    def test_validator_rejects_corrupt_vm_prediction(self):
        changed=self.d.vm_predictions.copy()
        changed.loc[0,'predicted_MRR'] += 9
        from dataclasses import replace
        with self.assertRaisesRegex(ValueError,'VM all_mae'):
            validate_sources(replace(self.d,vm_predictions=changed))

if __name__=='__main__':
    unittest.main()
