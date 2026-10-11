import importlib.util
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

SCRIPT=Path(__file__).resolve().parents[1] / 'src' / '06d_final_internal_test.py'
spec=importlib.util.spec_from_file_location('step06d', SCRIPT)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class TestStep06D(unittest.TestCase):
    def test_params_frozen_from_step_06b(self):
        self.assertEqual(m.MODEL_PARAMS, dict(loss='absolute_error', learning_rate=0.05,
            max_iter=140, max_leaf_nodes=12, min_samples_leaf=15,
            l2_regularization=5.0, random_state=42, early_stopping=False))

    def test_feature_allowlist_and_target_exclusion(self):
        names=[f'code1__feat_{i}' for i in range(21)] + [f'code2__feat_{i}' for i in range(21)] + [f'code3__feat_{i}' for i in range(21)] + ['code1__present','code2__present','code3__present']
        names += ['mode_from_sensor']
        cat=pd.DataFrame({'feature_name':names,'english_definition':['sensor']*len(names)})
        self.assertEqual(len(m.fixed_group_columns(cat,(1,2,3))),66)
        cat.loc[len(cat)]=['AVG_REMOVAL_RATE','target']
        with self.assertRaises(ValueError):m.fixed_group_columns(cat,(1,2,3))

    def test_final_training_only_train_and_valid(self):
        d=pd.DataFrame({'vm_partition':['train']*40+['valid']*20,
                        'AVG_REMOVAL_RATE':np.r_[np.arange(40,dtype=float),np.arange(20,dtype=float)+42],
                        'code1__a':np.arange(60,dtype=float)})
        med,mod=m.train_frozen(d,['code1__a'])
        self.assertTrue(np.isfinite(med))
        self.assertTrue(mod.__class__.__name__.startswith('HistGradient'))
        changed=d.copy();changed.loc[0,'vm_partition']='test'
        with self.assertRaises(ValueError):m.train_frozen(changed,['code1__a'])

    def test_test_target_cannot_enter_predict(self):
        d=pd.DataFrame({'vm_partition':['train']*50+['valid']*10,
                        'AVG_REMOVAL_RATE':np.linspace(60,100,60),
                        'code1__sensor':np.arange(60,dtype=float)})
        med,mod=m.train_frozen(d,['code1__sensor'])
        x=pd.DataFrame({'code1__sensor':[0.,10.,20.]})
        a=m.predict_from_features(mod,med,x,['code1__sensor'])
        b=m.predict_from_features(mod,med,x,['code1__sensor'])
        np.testing.assert_array_equal(a['HGB_absolute_error'],b['HGB_absolute_error'])
        with self.assertRaises(ValueError):m.predict_from_features(mod,med,x.assign(AVG_REMOVAL_RATE=[1,2,3]),['code1__sensor'])

    def test_extreme_primary_metrics_and_ordinary_sensitivity(self):
        y=np.array([70.,74.,4200.]);p=np.array([71.,73.,74.]);rev=np.array([False,False,True])
        self.assertGreater(m.metrics(y,p)['mae'],1300.)
        self.assertEqual(m.sensitivity_metrics(y,p,rev)['n'],2)
        self.assertEqual(m.sensitivity_metrics(y,p,rev)['mae'],1.)

if __name__=='__main__':unittest.main()
