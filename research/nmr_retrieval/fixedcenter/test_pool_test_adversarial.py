"""功能概述：独立构造完整18任务、1371查询封存结果，以核对汇总端到端契约。
build_full 创建两seed、九配置的合成模型/cache/protocol/predictions、冻结审核与driver
真实格式双EXIT证据；既有producer辅助函数只用于生成合成行和标准字段，不运行模型。
Tests 验证run不调用choose、保留既定preselected、无晋升；并复现配置标签替换和冻结
文件替换拒绝，以及审核身份和固定identity门禁。全部文件临时创建，不读实际测试数据。
"""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import summarize_pool_test as m
import test_summarize_pool_test as t


def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n')


def build_full(base,independent=True):
    base=Path(base).resolve();template=base/'template'
    _,_,rows,args=t.fixture(template)
    manifest=json.loads((Path(args['cache']).parent/'summary.json').read_text())
    metric=json.loads((Path(args['out'])/'summary.json').read_text())['metrics']
    identity={k:manifest[k] for k in ('query_sha256','identity_content_sha256','variant_mask_content_sha256')}
    envelope=base/'envelope';envelope.mkdir()
    frozen_path,queue_path,run=t.sealed_fixture(envelope)
    frozen=json.loads(frozen_path.read_text());queue=json.loads(queue_path.read_text())
    review_path=Path(frozen['freeze_review']['path'])
    reviewed={str(Path(m.__file__).resolve()):m.validation.sha(m.__file__),
              str(Path(m.validation.__file__).resolve()):m.validation.sha(m.validation.__file__)}
    dump(review_path,dict(status='PASS',independent=independent,reviewer='independent synthetic reviewer',
         scope='freeze_plan_and_implementation',reviewed_sha256=reviewed))
    science=dict(reviewed);science[str(review_path)]=m.validation.sha(review_path)
    for job in queue['jobs']:
        folder=Path(job['out']);config=folder.parent.parent.name;seed=int(folder.parent.name[1:])
        allow=frozen['ranker_configurations'][config][str(seed)];pool=allow['recall']
        for role in ('no_aug_checkpoint','aug_checkpoint'):
            ck=Path(allow[role]);ck.parent.mkdir(parents=True,exist_ok=True);ck.write_text(config+str(seed)+role)
            dump(ck.parent/'summary.json',dict(synthetic=True));(ck.parent/'EXIT_CODE').write_text('0')
            for p in (ck,ck.parent/'summary.json',ck.parent/'EXIT_CODE'):science[str(p)]=m.validation.sha(p)
        cache=envelope/'caches'/f's{seed}'/pool/'test/features.pt'
        cache.parent.mkdir(parents=True,exist_ok=True);cache.write_text('synthetic '+str(seed)+pool)
        cache_manifest=dict(manifest,recall=pool,feature_cache_sha256=m.validation.sha(cache))
        dump(cache.parent/'summary.json',cache_manifest);(cache.parent/'EXIT_CODE').write_text('0')
        allow['cache_sha256']=m.validation.sha(cache)
        for p in (cache,cache.parent/'summary.json',cache.parent/'EXIT_CODE'):science[str(p)]=m.validation.sha(p)
        job['_args']=dict(stage='evaluate',cache=str(cache),no_aug_checkpoint=allow['no_aug_checkpoint'],
            aug_checkpoint=allow['aug_checkpoint'],seed=seed,out=str(folder),frozen_selection=str(frozen_path))
    driver=envelope/'driver.py';driver.write_text('# synthetic driver source')
    science[str(driver)]=m.validation.sha(driver)
    frozen.update(fixed_test_identity=identity,reviewed_inputs_sha256=reviewed,
        evaluation_inputs_sha256=dict(science),
        freeze_review=dict(path=str(review_path),sha256=m.validation.sha(review_path)))
    frozen['allowed_ranker_configs']=[copy.deepcopy(v) for seeds in frozen['ranker_configurations'].values() for v in seeds.values()]
    dump(frozen_path,frozen);source_pins=dict(science);source_pins[str(frozen_path)]=m.validation.sha(frozen_path)
    for job in queue['jobs']:
        args=job.pop('_args');folder=Path(job['out'])
        direct=[args['cache'],str(Path(args['cache']).parent/'summary.json'),str(Path(args['cache']).parent/'EXIT_CODE'),
            args['no_aug_checkpoint'],args['aug_checkpoint'],str(frozen_path)]
        inputs={p:source_pins[p] for p in direct}
        for role in ('no_aug_checkpoint','aug_checkpoint'):
            for name in ('summary.json','EXIT_CODE'):
                p=str(Path(args[role]).parent/name);inputs[p]=source_pins[p]
        job['inputs_sha256']=inputs
        job['argv']=['python','-m','nmr_transfer.h_pool_ranker']+sum((['--'+k.replace('_','-'),str(v)] for k,v in args.items()),[])
        dump(folder/'protocol.json',dict(args=args,inputs_sha256=inputs))
        dump(folder/'summary.json',dict(input_sha256=inputs,checkpoints={args[k]:inputs[args[k]]
            for k in ('no_aug_checkpoint','aug_checkpoint')},metrics=metric,cohort='test',
            recall=folder.parent.parent.name.rsplit('_',1)[1],candidate_order_sha256=m.validation.CANDIDATE_SHA,
            query_sha256=identity['query_sha256'],reranking_seconds_mean=.02))
        (folder/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    queue['source_inputs_sha256']=source_pins;dump(queue_path,queue)
    dump(frozen_path.parent/'build_receipt.json',dict(status='BUILT_NOT_EXECUTED',promotion=False,
        output_sha256={str(p):m.validation.sha(p) for p in (frozen_path,queue_path)}))
    launch=json.loads((run/'launch_receipt.json').read_text())
    launch.update(queue_sha256={str(queue_path):m.validation.sha(queue_path)},bundle_sha256=source_pins,
                  source_sha256={str(driver):science[str(driver)]})
    dump(run/'launch_receipt.json',launch)
    pinned=dict(source_pins,**launch['queue_sha256']);start=json.loads((run/'driver_start.json').read_text())
    start['inputs_sha256']=pinned;dump(run/'driver_start.json',start)
    completion=json.loads((run/'completion.json').read_text())
    for job in queue['jobs']:
        record=completion['jobs'][job['id']]
        record.update(argv=job['argv'],requested_inputs_sha256=job['inputs_sha256'],queue_source_sha256=pinned)
        dump(run/(job['id']+'.receipt.json'),record)
    dump(run/'completion.json',completion)
    return frozen_path,queue_path,run


class FullRunTests(unittest.TestCase):
    def test_complete_end_to_end_does_not_select_or_promote(self):
        with tempfile.TemporaryDirectory() as tmp:
            f,q,r=build_full(Path(tmp));out=Path(tmp)/'result';out.mkdir()
            with patch.object(m.validation,'choose',side_effect=AssertionError('sealed results may not select')):
                result=m.run(SimpleNamespace(frozen=f,queue=q,run=r),out)
            self.assertEqual(result['preselected'],m.validation.CONFIGS[0])
            self.assertFalse(result['selection_performed']);self.assertFalse(result['promotion'])
            self.assertEqual(result['independent_acceptance_status'],'unavailable')
            self.assertEqual(result['metric_rows'],486)
            self.assertTrue((out/'metrics.tsv').is_file())

    def test_review_independence_is_required_with_self_consistent_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            f,q,r=build_full(Path(tmp),independent=False)
            with self.assertRaisesRegex(ValueError,'independent_review_required'):
                m.verify_run(f,q,r,{},{})


class IdentityTests(unittest.TestCase):
    def test_real_relabel_reproduction_now_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp).resolve();job,frozen,rows,args=t.fixture(base);old=Path(job['out'])
            config='mixed_listwise_single'
            new=old.parents[1].with_name(config)/old.parent.name/old.name
            old.parents[1].rename(new.parents[1]);job['out']=str(new);args['out']=str(new)
            job['argv'][job['argv'].index('--out')+1]=str(new)
            p=new/'protocol.json';v=json.loads(p.read_text());v['args']['out']=str(new);dump(p,v)
            # The wrong label has a distinct frozen mapping; original pair stays globally allowlisted.
            frozen['ranker_configurations'][config]={'193':dict(frozen['allowed_ranker_configs'][0],no_aug_checkpoint='/wrong/for-this-label.pt')}
            with self.assertRaisesRegex(ValueError,'not_allowlisted_for_config'):
                m.read_test(job,frozen,{},{},base/'frozen.json')

    def test_valid_hex_fixed_identity_is_exact_three_fields(self):
        correct={k:'a'*64 for k in ('query_sha256','identity_content_sha256','variant_mask_content_sha256')}
        self.assertTrue(m.fixed_identity(correct))
        for bad in ({},dict(correct,extra='a'*64),dict(correct,query_sha256='g'*64)):
            self.assertFalse(m.fixed_identity(bad))


if __name__=='__main__':unittest.main()
