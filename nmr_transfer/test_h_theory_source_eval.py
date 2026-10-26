"""功能概述：合成集成检查理论val分层诊断不会把部分匹配解释成完整谱误差。
同一图分别标记完整/未知覆盖，额外预测峰只惩罚完整分布；固定校准目标变化时
模型仍只前向一次，防止不同数值前向混入标签口径对照；空理论val、来源缺失及校准
身份错配均必须拒绝，避免以成功退出码接受零记录结果。无需源数据、GPU或服务。
"""
import unittest
import numpy as np
import torch
from .h_theory_source_eval import evaluate,source_validation_rows
from . import test_h_replay as replay_fixtures


class SourceMetricTests(unittest.TestCase):
    def test_original_validation_selection_rejects_empty_and_missing_sources(self):
        r=dict(nucleus='1H',domain='theory',split='train',source='A');v=dict(r,split='val');excluded=dict(v,domain='experimental');self.assertEqual(source_validation_rows([r,v,excluded]),[v])
        for wrong in [[r],[],[r,v,dict(r,source='B')]]:
            with self.assertRaises(ValueError):source_validation_rows(wrong)
    def test_empty_or_identity_mismatched_evaluation_is_rejected(self):
        r,g,*_=replay_fixtures.TestReplay().fixtures();r=dict(r,domain='theory',split='val',record_id='a');device=torch.device('cpu')
        for raw,aligned in [([],[]),([r],[]),([r],[dict(r,record_id='b')])]:
            with self.assertRaises(ValueError):evaluate(None,raw,aligned,{r['smiles']:g},device)
    def test_missing_coverage_and_same_prediction(self):
        r,g,*_=replay_fixtures.TestReplay().fixtures();rows=[dict(r,domain='theory',split='val',source='synthetic',group='synthetic_connectivity',record_id=str(i),q=np.ones(128,np.float32),observation=dict(kind=kind,windows=[])) for i,kind in enumerate(['unknown','complete_atoms'])];aligned=[dict(r,q=r['q']+1) for r in rows]
        class Frozen(torch.nn.Module):
            def __init__(self):super().__init__();self.calls=0
            def forward(self,b):self.calls+=1;return torch.tensor([.1]*64+[.9]*64).reshape(1,-1).expand(b.num_graphs,-1)
        model=Frozen();summary,records=evaluate(model,rows,aligned,{r['smiles']:g},torch.device('cpu'))
        self.assertEqual(model.calls,1);self.assertAlmostEqual(records[0]['errors']['raw'],0,places=6);self.assertAlmostEqual(records[1]['errors']['raw'],4,places=5);self.assertAlmostEqual(records[0]['errors']['calibrated'],1,places=5);self.assertEqual(summary['source_coverage:synthetic|unknown']['records'],1)


if __name__=='__main__':unittest.main()
