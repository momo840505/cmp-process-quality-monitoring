import importlib.util
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1]/"src"/"06b_vm_baseline_models.py"
spec=importlib.util.spec_from_file_location("step06b", SCRIPT)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class TestVM(unittest.TestCase):
    def test_group_column_allowlist(self):
        cat = pd.DataFrame({'feature_name':['code1__pressure','code4__pressure','mode_from_sensor','AVG_REMOVAL_RATE','WAFER_ID']})
        self.assertEqual(m.group_columns(cat,(1,2,3)), ['code1__pressure'])
        self.assertEqual(m.group_columns(cat,(4,5,6)), ['code4__pressure'])
    def test_target_not_allowed_in_catalog(self):
        d = pd.DataFrame({'WAFER_ID':[1,2,3],'code_family':['Codes 1-3']*3,
          'code_signature':['1,2,3']*3, 'has_complete_code_signature':[True]*3,
          'mrr_review_gt_1000':[False]*3,'vm_partition':['train','valid','test'],
          'AVG_REMOVAL_RATE':[2.,3.,4.], 'code1__pressure':[1.,2.,3.]})
        cat=pd.DataFrame({'feature_name':['code1__pressure','AVG_REMOVAL_RATE'],'english_definition':['sensor','target']})
        with self.assertRaises(ValueError):m.validate_inputs(d,cat)
    def test_full_metrics_include_extreme_validation_target(self):
        y=np.array([70.,4000.]);p=np.array([71.,71.]);flag=np.array([False,True])
        self.assertGreater(m.score(y,p)['mae'],1000.)
        self.assertEqual(m.ordinary_metric_view(y,p,flag)['n'],1)
        self.assertEqual(m.ordinary_metric_view(y,p,flag)['mae'],1)
    def test_valid_targets_do_not_enter_fit(self):
        tr=pd.DataFrame({'AVG_REMOVAL_RATE':[float(x) for x in range(100,160)],
            'code1__sensor':[float(x%7) for x in range(60)]})
        va=pd.DataFrame({'AVG_REMOVAL_RATE':[9999.,-9999.], 'code1__sensor':[1.,5.]})
        args=(tr,va,['code1__sensor'])
        median,base,pred,_=m.fit_models(*args)
        va.loc[:,'AVG_REMOVAL_RATE']=[1.,2.]
        median2,base2,pred2,_=m.fit_models(*args)
        self.assertEqual(median,median2)
        np.testing.assert_array_equal(base,base2)
        np.testing.assert_allclose(pred,pred2)
    def test_unique_wafer_partition_integrity(self):
        # Distinct WAFER_ID makes different partitions wafer-exclusive.
        good=pd.DataFrame({'WAFER_ID':[1,2,3], 'vm_partition':['train','valid','test']})
        self.assertFalse(good.WAFER_ID.duplicated().any())
        bad=pd.concat([good,good.iloc[[0]].assign(vm_partition='valid')],ignore_index=True)
        self.assertTrue(bad.WAFER_ID.duplicated().any())

if __name__=='__main__':unittest.main()
