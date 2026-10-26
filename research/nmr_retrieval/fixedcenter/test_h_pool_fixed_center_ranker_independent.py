"""功能概述：独立审计固定中心追加臂的候选资格、解析权重及评分不变性。
SyntheticRows构造1024个候选身份的可控资格表，以及自然512/追加正例两种缓存形状；
全部结构、谱张量和标签均为合成内容。IndependentFixedCenterTests按已知候选计数
核mask/internal/heldout排除，用不等group行数的常数特征给出解析query-row均值，
再核旧10tensor继承、共同候选分数、真实optimizer一步后的O/scale/36常量冻结。
只调用新模块纯函数与合成tensor，不运行真实fit，不读取任何远端缓存或test。
"""
import copy
from pathlib import Path
import sys
import unittest

import torch
import h_pool_fixed_center_ranker as f

torch.set_num_threads(1)


def group(internal, start=0):
    return next('group'+str(i) for i in range(start, start+1000)
                if f.internal_group('group'+str(i)) is internal)


class SyntheticRows:
    def __init__(self):
        self.query_group = group(False)
        self.other_query_group = group(False, 100)
        self.eligible_group = group(False, 200)
        self.internal_group = group(True)
        self.heldout_group = 'historical_heldout'
        groups = [self.eligible_group]*1024
        groups[0] = groups[512] = self.heldout_group
        groups[1] = groups[513] = self.internal_group
        groups[600] = self.query_group; groups[700] = self.other_query_group
        self.qualification = dict(groups=groups, heldout={self.heldout_group},
            internal=torch.tensor([f.internal_group(g) for g in groups]), metadata={})

    def row(self, recall, target=600, order=0, constant=None):
        ids = torch.arange(512) if recall == 'single' else torch.cat((torch.arange(256), torch.arange(512, 768)))
        pos = (ids == target).nonzero().flatten(); present = bool(len(pos))
        if not present: ids = torch.cat((ids, torch.tensor([target])))
        t = ids.float(); x = torch.stack([torch.sin(t/(j+2)) for j in range(33)], 1)
        if constant is not None: x.fill_(constant)
        qgroup = self.qualification['groups'][target]
        return dict(query_order=order, source='synthetic', record_id=str(order), smiles='C', group=qgroup,
            variant='observed', evaluation_pool_size=512, candidate_indices=ids, features=x,
            coarse_scores=t*.001, predicted_ppm=t[:, None]%7*.1+torch.tensor([[.5, 1., 2., 3.]]),
            tags=torch.zeros((len(ids), 4), dtype=torch.long), H=torch.full((len(ids),), 10.),
            K=torch.full((len(ids),), 10.), observation_input=dict(semantics='quantile_support',
                xs=[1., 2.], weights=[.5, .5], windows=[], partial_peaks=False, reported_count=None),
            query_observation=dict(windows=[], partial_peaks=False), kind=0,
            target_position=int(pos[0]) if present else 512, target_in_coarse_pool=present,
            negative_exclusion_mask=torch.tensor([self.qualification['groups'][int(i)] == qgroup
                or self.qualification['groups'][int(i)] in self.qualification['heldout'] for i in ids]))


class IndependentFixedCenterTests(unittest.TestCase):
    def test_exact_natural_candidate_qualification_ignores_target_fields(self):
        rows = SyntheticRows(); s = rows.row('single'); d = rows.row('dual')
        single = f.eligible_positions(s, rows.qualification)
        dual = f.eligible_positions(d, rows.qualification)
        self.assertEqual(len(single), 510); self.assertEqual(len(dual), 507)
        self.assertNotIn(512, single.tolist())
        for candidate_row, positions in ((s, single), (d, dual)):
            groups = [rows.qualification['groups'][int(i)] for i in candidate_row['candidate_indices'][positions]]
            self.assertTrue(all(g != rows.heldout_group and not f.internal_group(g) and g != candidate_row['group'] for g in groups))
        changed = copy.deepcopy(s); changed['target_position'] = 0; changed['target_in_coarse_pool'] = True
        self.assertTrue(torch.equal(single, f.eligible_positions(changed, rows.qualification)))
        blocked = copy.deepcopy(rows.qualification); blocked['internal'].fill_(True)
        with self.assertRaisesRegex(ValueError, 'empty_eligible'):
            f.eligible_positions(s, blocked)

    def test_query_row_equal_mean_not_group_equal_mean(self):
        rows = SyntheticRows(); single = []; dual = {}
        # 同group两行的中心为2和8，另一group一行20；query-row均值10，group均值12.5。
        for order, (target, value) in enumerate(((600, 2.), (600, 8.), (700, 20.))):
            s = rows.row('single', target, order, value)
            d = rows.row('dual', target, order, value+2.)
            single.append(s); dual[f.base.row_key(d)] = d
        model = f.ResearchRanker(True, [.2]*9)
        x, o, audit = f.estimate_centers(model, single, dual, rows.qualification, torch.device('cpu'))
        # 两池各.5，所以每行分别3、9、21；三行等权精确为11。
        self.assertTrue(torch.equal(x, torch.full((33,), 11.)))
        self.assertEqual(o.shape, (3,)); self.assertEqual(x.dtype, torch.float32)
        self.assertEqual(audit['query_rows'], 3); self.assertEqual(audit['fit_groups'], 2)
        self.assertTrue(all(a['query_row_weight'] == 1/3 for a in audit['row_audit']))

    def test_inherited_state_candidate_local_scores_and_frozen_optimizer_state(self):
        rows = SyntheticRows(); old = f.ResearchRanker(True, [.2]*9)
        torch.manual_seed(901)
        with torch.no_grad(): old.network[-1].weight.normal_(0, .1)
        model = f.inherit_model(old, dict(initial_sigmas=[.2]*9), torch.zeros(33), torch.zeros(3), torch.device('cpu'))
        self.assertEqual(len(old.state_dict()), 10)
        for name, tensor in old.state_dict().items(): self.assertTrue(torch.equal(tensor, model.state_dict()[name]))
        scale = torch.linspace(.5, 2., 33); before_scale = scale.clone()
        s, d = rows.row('single'), rows.row('dual')
        a = model.row_delta(s, scale, torch.device('cpu'))[:256]
        b = model.row_delta(d, scale, torch.device('cpu'))[:256]
        self.assertTrue(torch.allclose(a, b, atol=1e-7, rtol=1e-6))
        before = {k:v.clone() for k,v in model.state_dict().items() if k.startswith('observation_model.') or k in f.CENTER_BUFFERS}
        opt = torch.optim.AdamW(model.network.parameters(), lr=.001, weight_decay=.001)
        opt.zero_grad(); loss = model.row_delta(s, scale, torch.device('cpu')).square().mean(); loss.backward(); opt.step()
        for name, value in before.items(): self.assertTrue(torch.equal(value, model.state_dict()[name]))
        self.assertTrue(torch.equal(before_scale, scale))


if __name__ == '__main__':
    unittest.main()
