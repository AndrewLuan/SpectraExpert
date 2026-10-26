"""功能概述：独立验证候选池反事实工具的实际run入口与旧producer契约。

RunFixture在临时目录提供四份合成train缓存及四个冻结模型的producer文件。只mock缓存/
checkpoint反序列化与64结构选择；validate_caches、check_science、validate_pair、实际
ResearchRanker评分、六probe、query/aggregate输出与输入始末SHA均走真实实现。
RunEntryTests核对旧producer仅有best epoch/alpha/training_cache_sha256时能完成，并拒绝
producer元数据漂移。没有真实训练、真实缓存或test读取；源码使用现存唯一checkout。
"""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import pool_normalization_counterfactual as c

for root in (Path('/Users/andrewluan/Desktop/Projects/IRfingerprints/SpectrExpert'),
             Path('/home/user/IRfingerprints/SpectrExpert')):
    if (root/'nmr_transfer/h_research_ranker.py').is_file():
        SOURCE = root; sys.path.insert(0, str(root)); break
else:
    raise FileNotFoundError('scientific_checkout_source_missing')

import nmr_transfer.h_research_ranker as ranker
import nmr_transfer.h_pool_ranker as pool

torch.set_num_threads(1)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value if isinstance(value, str) else json.dumps(value))


def row(which):
    ids = torch.arange(512) if which == 'single' else torch.cat((torch.arange(256), torch.arange(512, 768)))
    positions = (ids == 600).nonzero().flatten(); present = bool(len(positions))
    if not present:
        ids = torch.cat((ids, torch.tensor([600])))
    t = ids.float()
    x = torch.stack([torch.sin(t/(j+1)) for j in range(33)], 1); x[:, 0] = t*.001
    return dict(query_order=0, source='synthetic', record_id='one', smiles='CC', group='g_train',
        variant='observed', mask_content_sha256='synthetic_mask', candidate_indices=ids,
        evaluation_pool_size=512, features=x, coarse_scores=t*.001,
        predicted_ppm=t[:, None]%7*.1+torch.tensor([[.5, 1., 2., 3.]]),
        tags=torch.zeros((len(ids), 4), dtype=torch.long), H=torch.full((len(ids),), 10.),
        K=torch.full((len(ids),), 10.), kind=0,
        observation_input=dict(semantics='quantile_support', xs=[1., 2.], weights=[.5, .5],
            windows=[], partial_peaks=False, reported_count=None),
        query_observation=dict(windows=[], partial_peaks=False),
        target_in_coarse_pool=present, target_position=int(positions[0]) if present else 512,
        outside_pool_rank=None if present else 1001, negative_exclusion_mask=(ids == 600))


class RunFixture:
    def __init__(self, folder):
        self.base = Path(folder).resolve(); self.root = self.base/'current'; self.old = self.base/'nmr_global_retrieval_20261022'
        self.contract = self.base/'contract.md'; put(self.contract, 'synthetic contract')
        self.caches = {}; self.models = {}; self.producers = []
        for seed in c.SEEDS:
            for recall in ('single', 'dual'):
                path = self.old/f'features/inherited_s{seed}/{recall}/train/features.pt'
                put(path, 'synthetic_cache_'+str(seed)+recall)
                manifest = dict(schema='nmr_research_20261022', cohort='train', recall=recall,
                    candidate_order_sha256=c.CANDIDATE_SHA, candidate_count=114982, pool_size=512,
                    checkpoint_sha256=f'{seed:064x}', query_sha256='a'*64,
                    identity_content_sha256='b'*64, variant_mask_content_sha256='c'*64,
                    negative_excluded_structure_sha256='d'*64, training_subset='train',
                    source_inputs_sha256={str(self.contract): c.sha(self.contract)})
                self.caches[str(path)] = dict(manifest=manifest, rows=[row(recall)])
                put(path.parent/'summary.json', dict(manifest, feature_cache_sha256=c.sha(path)))
                put(path.parent/'EXIT_CODE', '0\n')
            single = self.old/f'features/inherited_s{seed}/single/train/features.pt'
            for aug in (False, True):
                role = 'aug' if aug else 'no_aug'; path = self.old/f'rankers/factor_R0O1F0/s{seed}/{role}/best.pt'
                put(path, 'synthetic_checkpoint_'+str(seed)+role)
                torch.manual_seed(seed+int(aug)); model = ranker.ResearchRanker(True, [.2]*9)
                with torch.no_grad(): model.network[-1].weight.normal_(0, .1)
                scale = torch.linspace(.5, 2., 33)
                saved = dict(schema='nmr_research_ranker_20261022', observation=True,
                    args=dict(seed=seed, augment=aug, epochs=64, samples=8192, batch_size=32, anchor=.1),
                    epoch=8, alpha=.5 if aug else .25, train_sha256=c.sha(single),
                    cache_protocol=copy.deepcopy(self.caches[str(single)]['manifest']),
                    source_inputs_sha256={str(self.contract): c.sha(self.contract)},
                    ranker_source_sha256={str(SOURCE/'nmr_transfer/h_research_ranker.py'): c.sha(SOURCE/'nmr_transfer/h_research_ranker.py')})
                self.models[str(path)] = (model, scale, saved)
                producer = path.parent/'summary.json'; self.producers.append(producer)
                put(producer, dict(best=dict(epoch=8, alpha=saved['alpha']), training_cache_sha256=c.sha(single)))
                put(path.parent/'EXIT_CODE', '0\n')

    def execute(self, output):
        out = self.base/output; out.mkdir()
        args = SimpleNamespace(source=str(SOURCE), root=str(self.root), contract=str(self.contract), device='cpu', out=str(out))
        key = c.identity(next(iter(self.caches.values()))['rows'][0])
        def read_cache(path, known_sha):
            if c.sha(path) != known_sha:
                raise ValueError('synthetic_cache_pin_changed')
            producer = json.loads((Path(path).parent/'summary.json').read_text())
            if producer['feature_cache_sha256'] != known_sha:
                raise ValueError('synthetic_cache_producer_changed')
            return self.caches[str(path)], known_sha
        manifest = dict(schema=c.SCHEMA, group_count=1, entries=[dict(key=list(key))], groups=['g_train'], skips=[])
        with patch.object(ranker, 'read_feature_cache', side_effect=read_cache) as reader, \
                patch.object(pool, 'load_checkpoint', side_effect=lambda path, device:self.models[str(path)]) as loader, \
                patch.object(pool, 'validate_caches', wraps=pool.validate_caches) as vc, \
                patch.object(pool, 'check_science', wraps=pool.check_science) as cs, \
                patch.object(pool, 'validate_pair', wraps=pool.validate_pair) as vp, \
                patch.object(c, 'selection_manifest', return_value=manifest):
            result = c.run(args, out)
            calls = dict(cache_reads=reader.call_count, checkpoint_reads=loader.call_count,
                         validate_caches=vc.call_count, check_science=cs.call_count, validate_pair=vp.call_count)
        return result, out, calls


class RunEntryTests(unittest.TestCase):
    def test_actual_run_compatible_with_legacy_producers(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = RunFixture(folder); result, out, calls = fixture.execute('out')
            self.assertEqual(result['status'], 'PASS'); self.assertEqual(result['queries'], 2)
            self.assertEqual(calls['cache_reads'], 4); self.assertEqual(calls['checkpoint_reads'], 4)
            self.assertEqual(calls['validate_caches'], 2); self.assertEqual(calls['validate_pair'], 2)
            self.assertGreaterEqual(calls['check_science'], 8)
            self.assertEqual(result['initial_model_state_sha256'], result['final_model_state_sha256'])
            rows = [json.loads(line) for line in (out/'query.jsonl').read_text().splitlines()]
            self.assertTrue(all(r['ranks']['S_native']['rank'] == 1001 for r in rows))
            self.assertTrue(all(r['common_candidates'] == 256 and set(r['ranks']) == set(c.PROBES) for r in rows))
            self.assertTrue(all(r['checks']['original_rows_unchanged'] for r in rows))
            self.assertIn('common_score_differences', result['summary']['overall'])
            self.assertTrue(result['inputs_unchanged'])
            self.assertTrue((out/'protocol.json').is_file() and (out/'selection.json').is_file() and (out/'summary.json').is_file())

    def test_actual_run_rejects_legacy_producer_metadata_drift(self):
        for field in ('epoch', 'alpha', 'training_cache_sha256'):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as folder:
                fixture = RunFixture(folder); path = fixture.producers[0]
                producer = json.loads(path.read_text())
                if field == 'training_cache_sha256': producer[field] = 'f'*64
                else: producer['best'][field] += 1
                put(path, producer)
                with self.assertRaisesRegex(ValueError, 'checkpoint_producer'):
                    fixture.execute('out')


if __name__ == '__main__':
    unittest.main()
