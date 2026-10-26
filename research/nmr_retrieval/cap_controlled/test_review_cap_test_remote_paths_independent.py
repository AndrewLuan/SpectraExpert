"""功能概述：非作者将原8读边界迁移到跨hostreader，额外核例外范围和异常后恢复。
只用人工路径/metadata；本地symlink/tensor检查仍走未patch的Path。
"""
from pathlib import Path
import types
import unittest
from unittest.mock import patch
import review_cap_test_results_remote_paths as r
import test_review_cap_test_metadata_results_independent as previous
previous.r=r
previous.previous.r=r

class RemoteReaderIndependentTests(previous.MetadataAdmissionIndependentTests):
    def parts(self,*args,**kwargs):
        values=super().parts(*args,**kwargs);stats=values[2]
        stats.Path=Path;stats.c.Path=Path
        return values
    def test_remote_prefix_sibling_has_no_lexical_resolve_exemption(self):
        called=[]
        def resolve(p,*args,**kwargs):
            called.append(str(p));return Path('/synthetic/local_alias')
        with patch.object(type(Path()),'resolve',resolve):
            got=r.RemotePath('/home/user/IRfingerprints/SpectrExpert_evil/file.json').resolve()
        self.assertEqual(str(got),'/synthetic/local_alias')
        self.assertEqual(len(called),1)
    def test_exception_restores_metadata_paths_and_never_replaces_local_Path(self):
        stats=types.SimpleNamespace(Path=Path,c=types.SimpleNamespace(Path=Path))
        def fail(*args):
            self.assertIs(stats.Path,r.RemotePath);self.assertIs(stats.c.Path,r.RemotePath)
            self.assertIs(r.Path,Path)
            raise ValueError('synthetic_stop_before_payload')
        with patch.object(r,'_verify',side_effect=fail),self.assertRaisesRegex(ValueError,'synthetic_stop'):
            r.verify(None,None,stats,None)
        self.assertIs(stats.Path,Path);self.assertIs(stats.c.Path,Path);self.assertIs(r.Path,Path)

if __name__=='__main__':unittest.main()
