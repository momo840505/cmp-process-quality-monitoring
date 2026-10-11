"""Fast Step06C tests: no model retraining needed."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('vm06c',PROJECT/'src'/'06c_train_only_robustness.py')
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)
DATA=PROJECT/'outputs'/'analysis'

class TestStep06C(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df=pd.read_csv(DATA/'06a_stageA_vm_features.csv',low_memory=False)
        cls.catalog=pd.read_csv(DATA/'06a_feature_catalog.csv')
        cls.timeline=pd.read_csv(DATA/'03b_stageA_wafer_code_mrr.csv', usecols=['WAFER_ID','first_timestamp'])

    def test_no_metadata_or_mrr_in_features(self):
        names=M.input_checks(self.df,self.catalog,self.timeline)
        for g,codes in M.GROUPS.items():
            variants=M.feature_sets(names,codes)
            self.assertEqual([len(variants[v]) for v in ['full_66','no_usage_54','usage_only_12']],[66,54,12])
            self.assertTrue(not any('AVG_REMOVAL_RATE' in k or 'WAFER_ID' in k for k in variants['full_66']))

    def test_partition_and_wafer_separation(self):
        self.assertTrue(self.df.WAFER_ID.is_unique)
        self.assertEqual(int((self.df.vm_partition=='train').sum()),817)
        for g in M.GROUPS:
            tr=self.df[(self.df.vm_partition=='train')&(self.df.code_family==g)]
            self.assertTrue(set(tr.WAFER_ID).isdisjoint(set(self.df[self.df.vm_partition!='train'].WAFER_ID)))

    def test_full_metrics_retains_review_extreme(self):
        r=M.metrics([70.,75.,4326.],[71.,73.,73.],[False,False,True])
        self.assertEqual(r['all_n'],3)
        self.assertEqual(r['ordinary_n'],2)
        self.assertGreater(r['all_mae'],r['ordinary_mae'])
        self.assertEqual(r['review_rows'],1)

    def test_time_inventory_and_no_target_driven_features(self):
        ids=set(self.timeline.WAFER_ID)
        self.assertTrue(set(self.df.WAFER_ID).issubset(ids))
        self.assertFalse(self.timeline.first_timestamp.isna().any())
        self.assertFalse(set(['AVG_REMOVAL_RATE','mrr_review_gt_1000','code_family','vm_partition','WAFER_ID']) & set(self.catalog.feature_name))

    def test_fixed_hgb_and_split(self):
        model=M.hgb_fixed()
        self.assertEqual(model.get_params()['max_iter'],140)
        self.assertEqual(model.get_params()['loss'],'absolute_error')
        self.assertEqual(M.RNG_SEEDS,(19,71))
        self.assertEqual(M.FOLDS,3)

if __name__=='__main__':
    unittest.main()
