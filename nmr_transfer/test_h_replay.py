"""理论重放与酰胺联合监督的同步合成测试，不加载真实谱或远程资源。
验证理论校准仅改变目标、联合母节点批次偏移正确、单甲基观测仍保持部分监督。
依赖h_replaytrain和PyG图契约；不启动训练任务或改写数据。
"""
import unittest
import numpy as np
import torch
from .data import graph,quantiles,physical_features
from .h_conditions import local_environment
from .h_labels import make_key
from .h_amide_audit import amide_groups
from .h_replaytrain import ReplaySpectra,grouped_objective
from .train import collate
from .emission_model import weighted_quantiles


class TestReplay(unittest.TestCase):
    def fixtures(self,partial=False):
        s='CN(C)C=O';g=graph(s);g.local_environment=torch.tensor(local_environment(s));n,parents=amide_groups(g)[0]
        r=dict(smiles=s,source='synthetic',record_id='a',nucleus='1H',domain='experimental',cond=np.zeros(4,np.float32),physics=physical_features(s),q=quantiles([2.,3.,8.]),observation=dict(kind='unknown',windows=[]))
        groups={parents[0]:[2.]}
        if not partial:groups[parents[1]]=[3.]
        annotations={make_key(r):dict(groups=groups)}
        return r,g,n,parents,annotations

    def test_theory_calibration_uses_existing_target_without_mutating_source(self):
        r,g,_,_,labels=self.fixtures();r['domain']='theory';r['domain_q']=quantiles([4.,5.]);before=r['q'].copy()
        raw=ReplaySpectra([r],{r['smiles']:g},'1H',{},annotations=labels)[0];fixed=ReplaySpectra([r],{r['smiles']:g},'1H',{},annotations=labels,calibrated=True)[0]
        torch.testing.assert_close(fixed.q,torch.tensor(r['domain_q']/10).reshape(1,-1));torch.testing.assert_close(fixed.supervised_q,fixed.q);np.testing.assert_equal(r['q'],before);self.assertFalse(torch.equal(raw.q,fixed.q))

    def test_group_owners_are_local_until_batch_and_union_preserves_mass(self):
        r,g,n,parents,labels=self.fixtures();ds=ReplaySpectra([r],{r['smiles']:g},'1H',{},annotations=labels,grouped=True);b=collate([ds[0],ds[0]]);global_owners=b.group_owner+b.ptr[b.batch]
        for j in range(2):
            off=int(b.ptr[j]);self.assertTrue(bool((global_owners[torch.tensor(parents)+off]==n+off).all()));self.assertEqual(int(b.group_mask[n+off].sum()),6);self.assertEqual(int(b.group_expected[n+off]),6)
        values=torch.tensor([.2,.3,.2,.3],requires_grad=True);weights=torch.tensor([3.,3.,3.,3.]);owners=torch.tensor([parents[0],parents[1],parents[0]+len(g.x),parents[1]+len(g.x)]);groups=torch.tensor([0,0,1,1]);details=dict(values=values,weights=weights,owners=owners,groups=groups,quantiles=weighted_quantiles(values,weights,groups,2))
        parts=grouped_objective(details,b);torch.testing.assert_close(parts['site'],torch.tensor(0.),atol=1e-7,rtol=0);parts['loss'].backward();self.assertTrue(torch.isfinite(values.grad).all())

    def test_single_observed_methyl_does_not_fill_other_peak(self):
        r,g,n,parents,labels=self.fixtures(partial=True);b=collate([ReplaySpectra([r],{r['smiles']:g},'1H',{},annotations=labels,grouped=True)[0]])
        self.assertEqual(int(b.group_mask[n].sum()),3);self.assertEqual(int(b.group_expected[n]),6)
        values=torch.tensor([.2,.9]);weights=torch.tensor([3.,3.]);owners=torch.tensor(parents);groups=torch.zeros(2,dtype=torch.long);d=dict(values=values,weights=weights,owners=owners,groups=groups,quantiles=weighted_quantiles(values,weights,groups,1))
        torch.testing.assert_close(grouped_objective(d,b)['site'],torch.tensor(0.),atol=1e-7,rtol=0)


if __name__=='__main__':unittest.main()
