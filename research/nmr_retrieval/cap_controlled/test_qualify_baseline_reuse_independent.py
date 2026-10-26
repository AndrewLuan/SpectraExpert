"""功能概述：四格baseline复用资格的独立纯JSON攻击测试。
复用作者规范化元数据fixture，但独立验证首读源码pin、无科学payload读取、
缺证据整体四格重评和路径准入。所有输入均为临时人工JSON，不读取旧test内容。
源码漂移使用临时复制品和__file__patch；不会改动被审源码实际字节。
"""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import qualify_baseline_reuse as q
from test_qualify_baseline_reuse import Fixture


class IndependentQualificationTests(unittest.TestCase):
    def test_empty_evidence_four_fresh_and_zero_payload_reads(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture=Fixture(tmp);request=fixture.build();request['historical']={}
            opened=[];original=Path.read_bytes
            def read(path):
                if path.suffix in ('.pt','.pth','.ckpt') or 'predictions' in path.name:
                    opened.append(str(path));raise AssertionError('payload_forbidden')
                return original(path)
            metadata=q.Metadata()
            with patch.object(Path,'read_bytes',read):
                result=q.qualify(request,metadata);metadata.finish()
            self.assertEqual(opened,[]);self.assertEqual(result['new_evaluation_jobs_count'],12)
            self.assertEqual([v['qualified'] for v in result['baseline_cells']],[False]*4)
            self.assertFalse(result['scientific_hashes_verified_by_inventory'])

    def test_one_role_frozen_state_mismatch_invalidates_all_four(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture=Fixture(tmp);fixture.state_roles['211']['no_aug']['tensor_sha256']['observation_state_sha256']='0'*64
            result=fixture.execute()
            self.assertEqual(result['new_evaluation_jobs_count'],12)
            self.assertTrue(all(v['qualified'] is False for v in result['baseline_cells']))

    def test_source_change_during_first_metadata_read_cannot_be_adopted_as_starting_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=(Path(tmp)/'temporary_qualifier.py').resolve();raw=Path(q.__file__).read_bytes();source.write_bytes(raw)
            changed=raw+b'\n# temporary simulated drift\n';newsha=hashlib.sha256(changed).hexdigest()
            with patch.object(q,'__file__',str(source)):
                fixture=Fixture(tmp)
                fixture.impl_mutation=lambda v:v['reviewed_sha256'].update({str(source):newsha})
                request=fixture.build();metadata=q.Metadata();original=metadata.load;first=[True]
                def load(ref):
                    if first[0]:first[0]=False;source.write_bytes(changed)
                    return original(ref)
                with patch.object(metadata,'load',side_effect=load),self.assertRaises(ValueError):
                    q.qualify(request,metadata);metadata.finish()

    def test_symlink_to_tensor_rejected_before_first_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            tensor=Path(tmp)/'features.pt';tensor.write_text('untouched sentinel')
            alias=Path(tmp)/'admitted_metadata.json';alias.symlink_to(tensor)
            metadata=q.Metadata();read=[];original=Path.read_bytes
            def audited(path):
                if path==tensor:read.append(str(path));raise AssertionError('tensor_read')
                return original(path)
            with patch.object(Path,'read_bytes',audited),self.assertRaises(ValueError):
                metadata.load(dict(path=str(alias),sha256='0'*64))
            self.assertEqual(read,[])


if __name__=='__main__':unittest.main()
