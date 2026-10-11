"""Run from project root: .\\.venv\\Scripts\\python.exe -m unittest discover -s tests -p 'test_05c_magnitude.py' -v"""
import importlib.util
import unittest
from pathlib import Path

import pandas as pd

MODULE_PATH = Path(__file__).resolve().parents[1] / 'src' / '05c_sensor_magnitude_screen.py'
spec = importlib.util.spec_from_file_location('step05c', MODULE_PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

INPUT = Path('outputs/analysis/05b_wafer_code_activity_screening.csv')

class TestStep05CMagnitude(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not INPUT.exists():
            raise FileNotFoundError('Run Step05B first; missing '+str(INPUT))
        cls.df = pd.read_csv(INPUT)
        mod.check_required(cls.df)
        cls.profiles,cls.result = mod.build_profiles_and_screen(cls.df)

    def test_group_and_row_counts(self):
        self.assertEqual(len(self.result),3496)
        self.assertEqual(self.df.WAFER_ID.nunique(),1166)
        self.assertEqual(self.result.reference_or_later.value_counts().to_dict(),{'reference':2092,'later':1404})

    def test_candidates_not_in_reference(self):
        bad = self.result[self.result['05c__later_two_family_candidate']]
        self.assertTrue((bad.reference_or_later=='later').all())
        self.assertTrue((bad['05c__family_count']>=2).all())
        self.assertTrue((bad[['05c__pressure_family_flag','05c__slurry_family_flag','05c__motion_family_flag']].sum(axis=1)>=2).all())

    def test_later_change_cannot_change_reference_bounds(self):
        altered = self.df.copy()
        mask = altered.reference_or_later.eq('later')
        altered.loc[mask,'SLURRY_FLOW_LINE_C__median_nonzero'] = 99999999.
        p2,_=mod.build_profiles_and_screen(altered)
        pd.testing.assert_frame_equal(self.profiles,p2)

    def test_reference_uses_reference_only(self):
        self.assertEqual(len(self.profiles),len(mod.SENSORS)*6)
        self.assertTrue((self.profiles.reference_available <= self.df.eligible_for_reference.sum()).all())
        self.assertTrue(self.profiles.loc[self.profiles.reference_usable,'reference_available'].ge(mod.REF_MIN).all())

if __name__=='__main__': unittest.main()
