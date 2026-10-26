"""功能概述：检查域标记属于查询输入，并与数据划分、候选自有谱和头容量分开。
测试Spectra的理论/实验标记及候选批次统一实验域；非零残差时观测域头能区分
相同结构/溶剂的两域，shared对照使用相同参数而屏蔽域输入。缺失、非法标记必须
拒绝，factory恢复域头布局。只用合成图/模型，不读取真实谱或外部服务。
"""
import tempfile,unittest
from pathlib import Path
import torch
from . import test_h_preserved_group_model as preserved_fixtures
from . import test_h_replay as replay_fixtures
from .h_condinfer import candidate_batches
from .h_conditions import ConditionsSpectra
from .emission_model import make_model
from .prepare import sha


class DomainConditioningTests(unittest.TestCase):
    batch=preserved_fixtures.PreservedGroupTests.batch
    def model(self,folder,mode='observed'):
        path=Path(folder)/'registry.pt';torch.save(preserved_fixtures.PreservedGroupTests().registry(),path);cfg=dict(model='preserved_group_atom',nucleus='1H',width=16,layers=1,condition_mode='local',response_mode='solvent',context_depth=0,message_mode='late',equivalence_registry_path=str(path),equivalence_registry_sha256=sha(path),equivalence_mode='local',domain_conditioning=mode);return make_model(cfg).eval()
    def test_modality_is_record_type_and_candidates_share_query_domain(self):
        r,g,*_=replay_fixtures.TestReplay().fixtures();rs=[dict(r,domain='experimental',split='train'),dict(r,domain='theory',split='test')];dataset=ConditionsSpectra(rs,{r['smiles']:g},'1H',{})
        self.assertFalse(bool(dataset[0].modality_is_theory));self.assertTrue(bool(dataset[1].modality_is_theory));bank=dict(smiles=[r['smiles']],physics=torch.tensor(r['physics']).reshape(1,8),candidate_own_domain=['theory']);b=candidate_batches(bank,{r['smiles']:g},torch.device('cpu'))[0];self.assertFalse(bool(b.modality_is_theory.any()))
    def test_observed_domain_can_shift_theory_while_shared_uses_same_capacity(self):
        with tempfile.TemporaryDirectory() as folder:
            observed=self.model(folder);shared=self.model(folder,'shared');shared.load_state_dict(observed.state_dict());self.assertEqual(sum(p.numel() for p in shared.parameters()),sum(p.numel() for p in observed.parameters()));b=self.batch(['CCO','CCO']);b.modality_is_theory=torch.tensor([False,True]);baseline=observed(b)
            with torch.no_grad():
                for p in observed.equivalence_head.parameters():p.zero_()
                observed.equivalence_head[0].weight[0,-1]=1.;observed.equivalence_head[-1].weight[0,0]=1.
            shared.load_state_dict(observed.state_dict());out=observed(b);control=shared(b);self.assertTrue(torch.equal(out[0],baseline[0]));self.assertGreater(float((out[1]-out[0]).detach().min()),.01);self.assertTrue(torch.equal(control[0],control[1]));self.assertTrue(torch.equal(control[0],baseline[0]))
    def test_missing_and_invalid_domain_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            model=self.model(folder);b=self.batch(['CCO','CCO'])
            with self.assertRaises(ValueError):model(b)
            b.modality_is_theory=torch.tensor([0.,2.])
            with self.assertRaises(ValueError):model(b)
            b.modality_is_theory=torch.tensor([True])
            with self.assertRaises(ValueError):model(b)


if __name__=='__main__':torch.set_num_threads(1);unittest.main()
