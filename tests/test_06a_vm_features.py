"""Small synthetic checks, no external repo files required."""
import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src" / "06a_build_vm_features.py"
spec = importlib.util.spec_from_file_location("vm06a", SRC)
vm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vm)


def fixtures():
    rows = []
    qs = []
    for i in range(20):
        codes = (1, 2, 3) if i < 10 else (4, 5, 6)
        for c in codes:
            for j in range(4):
                x = dict(split="training", WAFER_ID=i, STAGE="A", MACHINE_DATA=c,
                         same_time_multiple_sensor_values=False)
                x.update({n: float(j+1) * (c + 1) for n in vm.SIGNALS})
                x.update({n: 100.0 + i + c for n in vm.CONSUMABLES})
                rows.append(x)
        qs.append(dict(split="training", WAFER_ID=i, STAGE="A", AVG_REMOVAL_RATE=70+i,
                       mrr_review_gt_1000=False))
    return pd.DataFrame(rows), pd.DataFrame(qs)

class TestVm06A(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.events,cls.labels=fixtures()
    def test_all_wafer_ids_unique_and_joined(self):
        result,_,audit=vm.make_features(self.events,self.labels)
        self.assertEqual(len(result),20)
        self.assertTrue(result.WAFER_ID.is_unique)
        self.assertEqual(set(result.WAFER_ID),set(self.labels.WAFER_ID))
    def test_no_target_in_feature_catalog(self):
        _,catalog,_=vm.make_features(self.events,self.labels)
        self.assertFalse(any("MRR" in s.upper() for s in catalog.feature_name))
    def test_target_change_does_not_change_sensor_features_or_partitions(self):
        a,cat,_=vm.make_features(self.events,self.labels)
        labs=self.labels.copy();labs.AVG_REMOVAL_RATE*=10
        b,_,_=vm.make_features(self.events,labs)
        pd.testing.assert_frame_equal(a[cat.feature_name],b[cat.feature_name])
        pd.testing.assert_series_equal(a.vm_partition,b.vm_partition)
    def test_partition_nonoverlap_and_reproducibility(self):
        a=vm.split_wafers(pd.DataFrame({"WAFER_ID":range(20),
                                        "code_family":["Codes 1-3"]*10+["Codes 4-6"]*10}))
        b=vm.split_wafers(pd.DataFrame({"WAFER_ID":range(20),
                                        "code_family":["Codes 1-3"]*10+["Codes 4-6"]*10}))
        pd.testing.assert_frame_equal(a,b)
        self.assertEqual(a.WAFER_ID.nunique(),len(a))
        self.assertEqual(set(a.vm_partition),{"train","valid","test"})
    def test_missing_code_not_imputed_from_other_codes(self):
        res,_,_=vm.make_features(self.events,self.labels)
        r=res.loc[res.WAFER_ID==0].iloc[0]
        self.assertTrue(np.isnan(r["code4__SLURRY_FLOW_LINE_A__median_nonzero"]))
        self.assertEqual(r["code4__present"],0)
        self.assertEqual(r["code1__SLURRY_FLOW_LINE_A__nonzero_fraction"],1.0)

if __name__ == "__main__": unittest.main()
