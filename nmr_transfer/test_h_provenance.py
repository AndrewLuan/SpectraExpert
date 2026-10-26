"""同次发射类别与更新骨干缓存的同步合成检查，不读取真实数据或GPU任务。
验证predict保存与每个固定分位点对应的类别，并恢复端到端模型的缓存开关；
防止后续宽度消融从另一次数值略有变化的前向重建类别。依赖h_condinfer的
候选推断接口，无持久化、真实检索或外部服务副作用。
另核验声明苯外恒等的局部残差可以逐位复用既有分位点及类别；未声明恒等契约
的普通模型拒绝复用。真实权重/配置/候选次序的所有权核验由main负责。
"""
import unittest
import torch
from .h_condinfer import predict
from .h_condmodel import ConditionSpecGNN
from .h_solvent_local import SolventLocalSpecGNN
from .h_conditions import encode_conditions
from .h_labels import make_key
from . import test_h_replay as replay_fixtures


class TestProvenance(unittest.TestCase):
    def test_local_residual_reuses_unchanged_condition_exactly(self):
        r,g,_,_,_=replay_fixtures.TestReplay().fixtures();meta=encode_conditions(dict(solvent='Water'));r=dict(r,cond=meta['old_cond']);metadata={make_key(r):meta};bank=dict(smiles=[r['smiles']],physics=torch.tensor(r['physics']).reshape(1,-1),val=[r],test=[],paired=[])
        m=SolventLocalSpecGNN('1H',32,2,'local','solvent',0).eval()
        p,fixed,tags=predict(m,bank,{r['smiles']:g},metadata,'local',torch.device('cpu'),lambda *a,**k:None)
        reuse=dict(predictions=p,fixed_predictions=fixed,candidate_quantile_classes=tags)
        with torch.no_grad():m.solvent_local_head.bias.fill_(.3)
        actual,ff,tt=predict(m,bank,{r['smiles']:g},metadata,'local',torch.device('cpu'),lambda *a,**k:None,reuse)
        for k in p:
            torch.testing.assert_close(actual[k],p[k],rtol=0,atol=0);self.assertTrue(torch.equal(tt[k],tags[k]));torch.testing.assert_close(ff[k],fixed[k],rtol=0,atol=0)
        with self.assertRaisesRegex(ValueError,'invariance_contract'):
            predict(ConditionSpecGNN('1H',32,2,'local'),bank,{r['smiles']:g},metadata,'local',torch.device('cpu'),lambda *a,**k:None,reuse)

    def test_prediction_saves_same_pass_tags_and_restores_cache_strategy(self):
        r,g,_,_,_=replay_fixtures.TestReplay().fixtures();bank=dict(smiles=[r['smiles']],physics=torch.tensor(r['physics']).reshape(1,-1),val=[r],test=[],paired=[]);m=ConditionSpecGNN('1H',32,2,'local').eval();m.use_frozen_embedding=False
        p,fixed,tags=predict(m,bank,{r['smiles']:g},{},'local',torch.device('cpu'),lambda *args,**kwargs:None)
        self.assertFalse(m.use_frozen_embedding);self.assertEqual(set(p),set(tags));self.assertEqual(len(p),1)
        for key,q in p.items():self.assertEqual(q.shape,(1,128));self.assertEqual(tags[key].shape,q.shape);self.assertTrue(bool((tags[key]<9).all()))


if __name__=='__main__':unittest.main()
