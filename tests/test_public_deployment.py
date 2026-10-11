"""Tests require only PUBLIC aggregate/simulated files; safe on fresh GitHub clone."""
import unittest
from pathlib import Path
import sys
import pandas as pd
import numpy as np
PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from src.public_demo_data import load_public_data,validate_public

class TestPublicDemo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s,cls.df=load_public_data(PROJECT)

    def test_public_payload_validated(self):
        self.assertEqual(len(self.df),120)
        self.assertFalse(self.s['data_scope']['official_test_used'])

    def test_no_raw_or_wafer_level_columns(self):
        self.assertNotIn('WAFER_ID',self.df.columns)
        self.assertNotIn('actual_MRR',self.df.columns)
        self.assertNotIn('predicted_MRR',self.df.columns)

    def test_final_vm_test_numbers(self):
        self.assertEqual(sum(d['test_n'] for d in self.s['vm'].values()),174)
        self.assertAlmostEqual(self.s['vm']['Codes 4-6']['model_mae'],3.001733,places=4)

    def test_spc_count_consistency(self):
        self.assertEqual(self.s['spc']['Codes 1-3']['i_chart_flags'],31)
        self.assertEqual(self.s['spc']['Codes 4-6']['ar1_flags'],4)

    def test_fdc_counts(self):
        self.assertEqual(self.s['fdc']['distinct_candidate_wafers'],11)
        self.assertEqual(sum(self.s['fdc']['signal_type_counts'].values()),27)

    def test_simulated_metrics(self):
        target=self.s['r2r']['target']
        self.assertAlmostEqual(np.abs(self.df.open_mrr-target).mean(),self.s['r2r']['mae_open'],places=5)
        self.assertAlmostEqual(np.abs(self.df.ewma_mrr-target).mean(),self.s['r2r']['mae_ewma'],places=5)

    def test_public_images_present(self):
        for name in ('spc_codes_1_3.png','spc_codes_4_6.png','vm_codes_1_3.png','vm_codes_4_6.png','fdc_candidates.png','r2r_comparison.png'):
            self.assertTrue((PROJECT/'docs/figures'/name).is_file(),name)

    def test_no_real_wafer_ids_in_public_json(self):
        raw=(PROJECT/'public_data/summary.json').read_text(encoding='utf-8')
        self.assertNotIn('WAFER_ID',raw)

    def test_reject_corrupted_simulation(self):
        bad=self.df.copy();bad.loc[0,'open_mrr']+=3
        with self.assertRaisesRegex(ValueError,'R2R published aggregate'):
            validate_public(self.s,bad)

if __name__=='__main__':unittest.main()
