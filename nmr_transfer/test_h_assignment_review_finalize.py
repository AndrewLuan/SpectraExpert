"""功能概述：验证双审核验器拒绝覆盖缺口、源错配、同会话双计和未经裁决的分歧。
使用合成metadata及策略bytes，核查原谱/其他策略行保留、test拒绝、弱目标不会自动开启。
不读取研究谱、远程文件、模型或API。
"""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from .h_assignment_review_finalize import resolve,overlay_policy,content_sha
from .data import identity


class ReviewFinalizeTests(unittest.TestCase):
    def fixture(self):
        raw=dict(record_id='a',source_ref='synthetic',smiles='CC',nucleus='1H',data_nature='experimental')
        partition=identity('CC')[2];self.assertNotEqual(partition,'test')
        c=dict(record_id='a',source='synthetic',split=partition,canonical_smiles='CC',
               source_record=raw,source_record_sha256=content_sha(raw))
        r=dict(record_id='a',split=partition,source_record_sha256=content_sha(raw),full_source_record_reviewed=True,
            action='mask_atom_keep_spectrum',reason_zh='合成证据',classification='synthetic',
            repair_uniquely_supported=False,repair_proposal=None)
        second=copy.deepcopy(r);second['reason_zh']='独立会话合成证据'
        return [c],[dict(session='one',reviews=[r]),dict(session='two',reviews=[second])],{
            'a':dict(record_id='a',source='synthetic',split=partition,status='synthetic')}

    def test_complete_review_retains_uncertainty_and_does_not_enable_weak_targets(self):
        c,b,i=self.fixture();before=copy.deepcopy((c,b,i));result=resolve(c,b,i,{})
        self.assertEqual((c,b,i),before);self.assertTrue(result[0]['review_complete'])
        self.assertFalse(result[0]['weak_supervision_enabled']);self.assertFalse(result[0]['domain_shift_confirmed'])

    def test_source_coverage_partition_and_review_guards(self):
        for mode in ('missing','same_session','same_payload','hash','source_body','test','partial','hard_repair','stale_split'):
            c,b,i=self.fixture()
            if mode=='missing':b=b[:1]
            elif mode=='same_session':b[1]['session']='one'
            elif mode=='same_payload':b[1]['reviews']=copy.deepcopy(b[0]['reviews'])
            elif mode=='hash':b[0]['reviews'][0]['source_record_sha256']='b'*64
            elif mode=='source_body':c[0]['source_record']['smiles']='CCC'
            elif mode=='test':c[0]['split']='test'
            elif mode=='partial':b[0]['reviews'][0]['full_source_record_reviewed']=False
            elif mode=='stale_split':i['a']['split']='test'
            else:b[0]['reviews'][0]['repair_uniquely_supported']=True
            with self.assertRaises(ValueError):resolve(c,b,i,{})

    def test_review_proposals_remain_documentation_and_never_activate_targets(self):
        c,b,i=self.fixture()
        proposal=dict(type='conditional_weak_target_only',raw_hard_repair=False,shared_ppm=1.2)
        b[0]['reviews'][0]['repair_proposal']=proposal
        result=resolve(c,b,i,{})[0]
        self.assertEqual(result['reviewer_proposals_not_applied'][0],proposal)
        self.assertFalse(result['hard_atom_supervision']);self.assertFalse(result['weak_supervision_enabled'])

    def test_disagreement_requires_explicit_reasoned_adjudication(self):
        c,b,i=self.fixture();b[1]['reviews'][0]['action']='hold_experimental_domain_pending_provenance'
        with self.assertRaises(ValueError):resolve(c,b,i,{})
        result=resolve(c,b,i,{'a':dict(action='mask_atom_keep_spectrum',reason_zh='独立复核的明确理由')})
        self.assertFalse(result[0]['reviewer_action_agreement']);self.assertTrue(result[0]['experimental_spectral_supervision'])

    def test_policy_overlay_preserves_other_bytes_and_cannot_activate_hard_targets(self):
        c,b,i=self.fixture();b[0]['reviews'][0]['action']=b[1]['reviews'][0]['action']='hold_experimental_domain_pending_provenance'
        result=resolve(c,b,i,{})
        untouched=b'{"record_id":"other","split":"val","source":"synthetic"}\r\n'
        prior=dict(record_id='a',split='train',source='synthetic',keep_spectral_supervision=True,
                   keep_atom_parent_supervision=False,q=[1,2,3])
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'prior';out=Path(d)/'out';p.write_bytes(untouched+(json.dumps(prior)+'\n').encode())
            report=overlay_policy(p,out,result);self.assertEqual(report['reviewed_records'],1)
            self.assertTrue(out.read_bytes().startswith(untouched));updated=json.loads(out.read_bytes().splitlines()[1])
            self.assertFalse(updated['keep_spectral_supervision']);self.assertFalse(updated['keep_atom_parent_supervision'])
            self.assertEqual(updated['q'],prior['q'])


if __name__=='__main__':unittest.main()
