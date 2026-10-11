import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location('case_review',Path(__file__).resolve().parents[1]/'src'/'05e_ocap_candidate_review.py')
mod=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)

class TestFrozenEvidence(unittest.TestCase):
    def test_two_different_families(self):
        s='[{"signal":"retainer_pressure","family":"pressure","observed_median_nonzero":5,"reference_q01":10,"reference_q99":20,"direction":"below"}, {"signal":"slurry_flow_A","family":"slurry","observed_median_nonzero":30,"reference_q01":10,"reference_q99":20,"direction":"above"}]'
        self.assertEqual(len(mod.parse_evidence(s)),2)
    def test_direction_conflict_rejected(self):
        s='[{"signal":"retainer_pressure","family":"pressure","observed_median_nonzero":25,"reference_q01":10,"reference_q99":20,"direction":"below"},{"signal":"slurry_flow_A","family":"slurry","observed_median_nonzero":30,"reference_q01":10,"reference_q99":20,"direction":"above"}]'
        with self.assertRaises(ValueError):mod.parse_evidence(s)
    def test_single_family_rejected(self):
        s='[{"signal":"chamber_pressure","family":"pressure","observed_median_nonzero":5,"reference_q01":10,"reference_q99":20,"direction":"below"},{"signal":"retainer_pressure","family":"pressure","observed_median_nonzero":30,"reference_q01":10,"reference_q99":20,"direction":"above"}]'
        with self.assertRaises(ValueError):mod.parse_evidence(s)
    def test_unknown_signal_rejected(self):
        s='[{"signal":"nonexistent","family":"pressure","observed_median_nonzero":5,"reference_q01":10,"reference_q99":20,"direction":"below"},{"signal":"slurry_flow_A","family":"slurry","observed_median_nonzero":30,"reference_q01":10,"reference_q99":20,"direction":"above"}]'
        with self.assertRaises(ValueError):mod.parse_evidence(s)

if __name__=='__main__':unittest.main()

# Read-only end-to-end integrity checks on the user's existing prior-step outputs.
class TestProjectInputs(unittest.TestCase):
    def setUp(self):
        self.folder = Path(__file__).resolve().parents[1]/'outputs'/'analysis'

    def test_prior_step_membership_and_quality_join(self):
        if not self.folder.exists():
            self.skipTest('Project analysis inputs not available in standalone template')
        _, later, casebook, evidence, _ = mod.load_and_validate(self.folder)
        self.assertEqual(len(later), 468)
        self.assertEqual(len(casebook), 11)
        self.assertEqual(len(evidence), 27)
        self.assertEqual(int(casebook.outside_reference_MRR_q05_q95_descriptive.sum()), 0)
