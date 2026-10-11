"""Integration checks using the user's already-produced local Step 02B/05C outputs."""
import importlib.util
from pathlib import Path
import unittest

import pandas as pd

SRC = Path(__file__).resolve().parents[1] / 'src' / '05d_link_sensor_flags_to_quality.py'
spec = importlib.util.spec_from_file_location('step05d', SRC)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
ROOT = Path(__file__).resolve().parents[1]


class Step05DTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sensor = pd.read_csv(ROOT/'outputs/analysis/05c_wafer_code_magnitude_screening.csv')
        cls.candidates = pd.read_csv(ROOT/'outputs/analysis/05c_later_magnitude_candidates.csv')
        cls.quality = pd.read_csv(ROOT/'outputs/processed/02b_wafer_stage_quality.csv')

    def test_01_frozen_sensor_keys_match_candidate_file(self):
        linked, sensor = mod.construct_wafer_tables(self.sensor,self.quality,self.candidates)
        self.assertEqual(len(linked), 1166)
        self.assertEqual(len(sensor), 3496)
        self.assertEqual(int(linked.sensor_candidate.sum()),11)

    def test_02_one_outcome_per_wafer_not_per_code(self):
        linked,_ = mod.construct_wafer_tables(self.sensor,self.quality,self.candidates)
        later, stats = mod.build_comparison(linked)
        self.assertEqual(len(later),468)
        self.assertTrue(later.WAFER_ID.is_unique)
        self.assertEqual(stats.later_wafers.sum(),468)
        self.assertEqual(int(later.mrr_review_gt_1000.sum()),1)

    def test_03_sensor_flags_unchanged_if_only_quality_changes(self):
        original,_ = mod.construct_wafer_tables(self.sensor,self.quality,self.candidates)
        changed_quality = self.quality.copy()
        mask=(changed_quality.split == 'training') & (changed_quality.STAGE == 'A')
        changed_quality.loc[mask,'AVG_REMOVAL_RATE'] += 700
        shifted,_ = mod.construct_wafer_tables(self.sensor,changed_quality,self.candidates)
        pd.testing.assert_series_equal(
            original.sort_values('WAFER_ID').reset_index(drop=True).sensor_candidate,
            shifted.sort_values('WAFER_ID').reset_index(drop=True).sensor_candidate
        )

    def test_04_reference_mrr_not_changed_by_later_quality_changes(self):
        original,_ = mod.construct_wafer_tables(self.sensor,self.quality,self.candidates)
        l1, _ = mod.build_comparison(original)
        adjusted = original.copy()
        adjusted.loc[adjusted.reference_or_later=='later','AVG_REMOVAL_RATE']+=400
        l2, _ = mod.build_comparison(adjusted)
        a=l1.sort_values(['WAFER_ID','STAGE']).reset_index(drop=True)
        b=l2.sort_values(['WAFER_ID','STAGE']).reset_index(drop=True)
        for col in ['reference_median','reference_q05','reference_q95']:
            pd.testing.assert_series_equal(a[col],b[col])


if __name__ == '__main__':
    unittest.main()
