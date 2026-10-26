"""功能概述：非作者检验未来12test审阅器的读取边界与执行门槛顺序。
仅人工bytes与metadata；计算口径继续复用已审数学源码，不新增科学oracle。
三个verify流程测试只替换已审外层metadata组件，保留本reader的门槛顺序与Copies。
"""
import argparse
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import review_cap_test_results as r
from test_review_cap_test_results import StoreFixture

class IndependentReadBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='cap12_independent_')
        self.addCleanup(self.tmp.cleanup);self.f=StoreFixture(self.tmp.name)
    def test_canonical_tensor_relabelled_json_refused_before_first_byte(self):
        key=self.f.tensor;alias=self.f.root/'alias.json';alias.write_bytes(b'artificial tensor')
        h=r.digest(alias.read_bytes());self.f.files[key]=str(alias);self.f.hashes[key]=h;self.f.inventory[key]=h
        self.f.tensor=self.f.remote+'/other_hashonly.pt';self.f.h='b'*64;self.f.inventory[self.f.tensor]=self.f.h;self.f.save()
        store=self.f.store();old=Path.read_bytes;opened=[]
        def guard(p):
            if p==alias:opened.append(str(p));raise AssertionError('tensor bytes opened')
            return old(p)
        with patch.object(Path,'read_bytes',guard),self.assertRaisesRegex(ValueError,'tensor'):
            store.raw(key,h)
        self.assertEqual(opened,[])
    def test_legacy_ranking_is_payload_and_not_metadata_before_unlock(self):
        for basename in ('ranks.jsonl','rankings.jsonl'):
            key=self.f.add('legacy/'+basename,b'{"rank":1}\n');self.f.inventory[key]=self.f.hashes[key];self.f.save()
            store=self.f.store();old=Path.read_bytes;opened=[]
            def guard(p):
                if p.name==basename:opened.append(str(p));raise AssertionError('unregistered payload opened')
                return old(p)
            with patch.object(Path,'read_bytes',guard),self.assertRaisesRegex(ValueError,'prediction|payload|registered'):
                store.raw(key,self.f.hashes[key])
            self.assertEqual(opened,[])
    def test_main_request_replaced_by_tensor_symlink_rejected_before_final_read(self):
        request=self.f.root/'review_request.json';request.write_text('{}')
        target=self.f.root/'features.pt';target.write_bytes(b'artificial tensor never read')
        out=self.f.root/'new_review'
        store=SimpleNamespace(inventory={},maps={},reads={})
        def verify(*args):
            request.unlink();request.symlink_to(target)
            return dict(status='PASS')
        old=Path.read_bytes;seen=[]
        def guard(p):
            if p==request and p.is_symlink():
                seen.append(str(p));raise AssertionError('request symlink tensor read at finish')
            return old(p)
        import sys
        with patch.object(sys,'argv',['reader','--manifest','unused.json','--request',str(request),'--out',str(out)]),\
             patch.object(r,'load_helpers',return_value=(None,None)),patch.object(r,'Copies',return_value=store),\
             patch.object(r,'verify',side_effect=verify),patch.object(Path,'read_bytes',guard),\
             self.assertRaisesRegex(ValueError,'symlink_redirect'):
            r.main()
        self.assertEqual(seen,[])
    def parts(self,inner_exit=0):
        root=Path(self.f.remote)
        own=[root/'resource_ready_analysis/summarize_sealed_cap_results.py',root/'sealed_cap_matrix.py',
             root/'sealed_cap_pairmath.py',root/'native_cost_evaluation/cap_test_evaluation_common.py']
        for p in own:
            k=self.f.add(str(p.relative_to(root)),b'# synthetic source');self.f.inventory[k]=self.f.hashes[k]
        args=dict(source=str(root/'source'),protocol=str(root/'protocol.json'),bindings=str(root/'bindings.json'),
          frozen_selection=str(root/'frozen.json'),freeze_review=str(root/'review.json'),queue=str(root/'queue.json'),
          run=str(root/'runs/test_success'),analysis_review=str(root/'analysis_review.json'),statistics_output=str(root/'statistics'))
        argv=['python','-B',str(own[0])]
        for k,v in args.items():
            if k=='statistics_output':continue
            argv+=['--'+k.replace('_','-'),v]
        argv+=['--out',args['statistics_output']]
        record=dict(actual_inner_exit=inner_exit,actual_external_exit=0,actual_ssh_exit=0,argv=argv)
        key=self.f.add('statistics_execution.json',json.dumps(record).encode());self.f.inventory[key]=self.f.hashes[key];self.f.save()
        request=dict(schema='cap10_actual12_test_result_review_request_20261025',root=str(root),args=args,
          statistics_execution=dict(path=key,sha256=self.f.hashes[key]))
        store=self.f.store()
        def gate(args,paths,require_cell=False):
            for p in paths:store.values[str(p)]=self.f.hashes[str(p)]
            return store,{}, {},dict(input_sha256={},source_sha256={},preselected='baseline'),None
        stats=SimpleNamespace(c=SimpleNamespace(Pins=object,sealed_gate=gate,reviewed=lambda *a:None),
          load_metadata=lambda *a:dict(scope='analysis_implementation'),
          execution=lambda *a:(dict(input_sha256={},source_sha256={}),{str(cell):{} for cell,_,_ in self.f.refs()},self.f.refs()),
          matrix=SimpleNamespace(CONFIGS=[cell[0] for cell,_,_ in self.f.refs()]),
          verify_leaf_values=lambda *a:(_ for _ in ()).throw(RuntimeError('payload_phase_reached')))
        independent=SimpleNamespace(actual_stage=lambda *a:({},dict(strict_double_exit0=True)))
        return store,request,stats,independent
    def verify(self,store,request,stats,independent):
        return r.verify(store,request,stats,independent)
    def test_bool_statistics_exit_rejected_before_any_prediction(self):
        s,q,stats,i=self.parts(inner_exit=False)
        with self.assertRaisesRegex(ValueError,'strict_int_EXIT0'):self.verify(s,q,stats,i)
        self.assertFalse(s.metadata_complete)
        self.assertFalse(any(event[0]=='prediction' for event in s.events))
    def test_independent_last_cell_execution_rejects_before_unlock(self):
        s,q,stats,i=self.parts()
        i.actual_stage=lambda *a:(_ for _ in ()).throw(ValueError('last_cell_not_double0'))
        with self.assertRaisesRegex(ValueError,'last_cell_not_double0'):self.verify(s,q,stats,i)
        self.assertFalse(s.metadata_complete);self.assertFalse(any(x[0]=='prediction' for x in s.events))
    def test_all_metadata_pass_then_first_registered_payload_is_read(self):
        s,q,stats,i=self.parts()
        with self.assertRaisesRegex(RuntimeError,'payload_phase_reached'):self.verify(s,q,stats,i)
        self.assertTrue(s.metadata_complete)
        events=[x[0] for x in s.events]
        self.assertLess(events.index('complete12_gate'),events.index('prediction'))
        self.assertEqual(sum(x=='prediction' for x in events),1)

if __name__=='__main__':unittest.main()
