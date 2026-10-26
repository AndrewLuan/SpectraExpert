"""功能概述：独立复核冻结构建器的seed缓存绑定、forward身份和具名allowlist。
复用作者临时fixture，分别注入seed互换、forward SHA变更、验证producer漂移；
正常构建核对config→seed→allowed映射与审核SHA，拦截测试缓存/预测读取；
旧封存registry再覆盖跨seed bank、原冻结授权及最终review对旧队列的绑定。
仅生成临时合成文件，不运行模型，不读取真实数据或访问外部API。
"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_build_pool_frozen import Fixture, write
import build_pool_frozen as m


class IndependentFreezeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.f=Fixture(self.temp.name)

    def test_original_seed_swap_attack_now_rejected(self):
        entry=self.f.plan['evaluations'][0]
        other=next(e for e in self.f.plan['evaluations']
                   if e['config']==entry['config'] and e['seed']!=entry['seed'])
        entry['cache'],entry['cache_sha256']=other['cache'],other['cache_sha256']
        write(self.f.plan_path,self.f.plan);self.f.approve()
        with self.assertRaisesRegex(ValueError,'test_cache_seed_or_inherited_path_mismatch'):
            self.f.build()

    def test_same_path_changed_forward_model_rejected(self):
        entry=self.f.plan['evaluations'][0]
        p=Path(entry['cache']).parent/'summary.json'
        producer=json.loads(p.read_text());producer['checkpoint_sha256']='f'*64
        write(p,producer);self.f.approve()
        with self.assertRaisesRegex(ValueError,'test_cache_identity_or_scope_mismatch'):
            self.f.build()

    def test_validation_producer_drift_not_absorbed_by_plan_review(self):
        config=m.validation.CONFIGS[0]
        protocol=self.f.root/'formal'/config.rsplit('_',1)[0]/'s193'/config.rsplit('_',1)[1]/'original_val/protocol.json'
        p=Path(json.loads(protocol.read_text())['args']['cache']).parent/'summary.json'
        producer=json.loads(p.read_text());producer['checkpoint_sha256']='f'*64
        write(p,producer);self.f.approve()
        with self.assertRaisesRegex(ValueError,'validation_cache_producer_not_selection_bound'):
            self.f.build()

    def test_named_ranker_mapping_and_review_digest(self):
        _,frozen,queue=self.f.build()
        self.assertEqual(set(frozen['ranker_configurations']),set(frozen['configurations']))
        mapped=[]
        for config,seeds in frozen['ranker_configurations'].items():
            self.assertEqual(set(seeds),{'193','211'})
            for seed,allowed in seeds.items():
                self.assertEqual(allowed['seed'],int(seed))
                self.assertEqual(allowed['recall'],config.rsplit('_',1)[1])
                mapped.append(allowed)
        self.assertEqual(len(mapped),18)
        self.assertEqual(mapped,frozen['allowed_ranker_configs'])
        self.assertEqual(frozen['freeze_review'],dict(path=str(self.f.review_path),sha256=m.sha(self.f.review_path)))
        self.assertEqual(frozen['validation_summary_path'],str(self.f.summary_path))
        self.assertEqual(frozen['test_plan_path'],str(self.f.plan_path))
        self.assertTrue(all(len(j['inputs_sha256'])==16 for j in queue['jobs']))

    def test_test_cache_and_predictions_never_opened(self):
        original=Path.open
        def guard(path,*args,**kwargs):
            if path.name in ('features.pt','predictions.jsonl'):
                raise AssertionError('test_value_read')
            return original(path,*args,**kwargs)
        with patch.object(Path,'open',guard):
            result,_,_=self.f.build()
        self.assertEqual(result['status'],'BUILT_NOT_EXECUTED')
        self.assertIs(result['test_cache_values_read'],False)
        self.assertIs(result['test_predictions_read'],False)


    def test_legacy_extract_bank_cannot_cross_seed(self):
        argv=self.f.legacy_jobs[0]['argv']
        argv[argv.index('--bank')+1]=str(self.f.old/'banks/inherited_s211/bank.pt')
        self.f.update_legacy_queue();self.f.approve()
        with self.assertRaisesRegex(ValueError,'legacy_extract_scientific_registration_mismatch'):
            self.f.build()

    def test_legacy_extract_must_be_allowed_by_original_freeze(self):
        self.f.legacy_frozen['allowed_extract_configs']=[]
        write(self.f.legacy_selection_path,self.f.legacy_frozen);self.f.approve()
        with self.assertRaisesRegex(ValueError,'legacy_extract_not_frozen'):
            self.f.build()

    def test_final_review_binds_legacy_queue(self):
        approval=json.loads(self.f.review_path.read_text())
        del approval['reviewed_sha256'][str(self.f.legacy_queue_path)]
        write(self.f.review_path,approval)
        with self.assertRaisesRegex(ValueError,'review_does_not_bind_current_inputs'):
            self.f.build()


if __name__=='__main__':
    unittest.main()
