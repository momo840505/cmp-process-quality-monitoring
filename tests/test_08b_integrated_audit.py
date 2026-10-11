"""Unit tests for non-destructive Step 08B consistency checks."""
import sys
import tempfile
import unittest
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec

SCRIPT=Path(__file__).resolve().parents[1]/'src'/'08b_integrated_audit.py'
spec=spec_from_file_location('audit08b',SCRIPT)
module=module_from_spec(spec)
sys.modules[spec.name]=module
spec.loader.exec_module(module)

class TestAudit08B(unittest.TestCase):
    def test_sha256_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Path(tmp)/'x.txt';f.write_text('a',encoding='utf8')
            h=module.digest(f)
            self.assertEqual(h,module.digest(f));f.write_text('b',encoding='utf8')
            self.assertNotEqual(h,module.digest(f))

    def test_missing_file_reports_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=module.Audit(Path(tmp))
            self.assertIsNone(a.json('missing.json','missing'))
            self.assertEqual(a.checks[0].status,'FAIL')

    def test_invalid_json_reports_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'bad.json').write_text('{broken',encoding='utf8')
            a=module.Audit(root);self.assertIsNone(a.json('bad.json','invalid'))
            self.assertTrue(any(c.status=='FAIL' for c in a.checks))

    def test_markdown_json_report_only_new_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);src=root/'source.bin';src.write_bytes(b'original')
            checks=[module.Check('test','PASS','ok'),module.Check('git','WARN','pending')]
            json_path,md_path=module.write_reports(root,checks)
            self.assertTrue(json_path.exists() and md_path.exists())
            self.assertEqual(src.read_bytes(),b'original')
            self.assertEqual(__import__('json').loads(json_path.read_text())['summary'],{'PASS':1,'WARN':1,'FAIL':0})

    def test_git_check_warns_if_repository_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=module.Audit(Path(tmp));module.audit_github(a)
            self.assertTrue(any(c.status=='WARN' for c in a.checks))
            self.assertFalse(any(c.status=='FAIL' for c in a.checks))

if __name__=='__main__':unittest.main()
