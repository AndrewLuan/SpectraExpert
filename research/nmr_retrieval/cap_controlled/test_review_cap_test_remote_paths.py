"""功能概述：跨host远端canonical只在明确RR namespace适配；本地安全边界保持。
全部人工路径和数据，承接13metadata测试，不打开实际test/模型值。
"""
from pathlib import Path
import types,unittest
from unittest.mock import patch
import review_cap_test_results_remote_paths as r
import test_review_cap_test_metadata_results as previous
previous.r=r
previous.old.r=r
MetadataStoreTests=previous.MetadataStoreTests

class RemoteNamespaceTests(unittest.TestCase):
    def test_remote_path_not_resolved_against_local_host_filesystem(self):
        remote='/home/user/IRfingerprints/SpectrExpert/outputs/synthetic/queue.json'
        original=Path.resolve
        def mac_alias(p,*a,**k):
            if str(p).startswith('/home/user/'):return Path('/System/Volumes/Data'+str(p))
            return original(p,*a,**k)
        with patch.object(type(Path()),'resolve',mac_alias):
            self.assertNotEqual(str(Path(remote).resolve()),remote)
            self.assertEqual(r.canonical(remote),remote)
            self.assertEqual(str(r.RemotePath(remote).resolve()),remote)
    def test_lexical_alias_parent_and_duplicate_slash_reject(self):
        for value in ('relative/path','/home/user/../other','//home/user/RR','/home//user/RR','/home/user/./RR'):
            with self.assertRaises(ValueError):r.canonical(value)
    def test_adapter_changes_only_inherited_metadata_module_names(self):
        fake=types.SimpleNamespace(Path=Path,c=types.SimpleNamespace(Path=Path));seen=[]
        def delegate(*args):
            seen.append((fake.Path,fake.c.Path));self.assertIs(r.Path,Path);self.assertIs(Path,r.Path);return 'metadata-only'
        with patch.object(r,'_verify',delegate):self.assertEqual(r.verify(None,None,fake,None),'metadata-only')
        self.assertEqual(seen,[(r.RemotePath,r.RemotePath)]);self.assertIs(fake.Path,Path);self.assertIs(fake.c.Path,Path)
    def test_outside_explicit_remote_namespace_uses_real_resolve(self):
        original=Path.resolve;called=[]
        def observe(p,*a,**k):called.append(str(p));return original(p,*a,**k)
        with patch.object(type(Path()),'resolve',observe):r.RemotePath('/synthetic/local/path').resolve()
        self.assertEqual(called,['/synthetic/local/path'])

if __name__=='__main__':unittest.main()
