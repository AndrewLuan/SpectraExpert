"""功能概述：验证审核标签训练接入的身份隔离、配对曝光及真实梯度边界。
合成源原子数组检查更正前置条件、mask/谱多重集保持；两图可微损失检查暂缓谱
严格零梯度而保留样本仍训练。默认权重逐项等同旧目标，test/理论同ID拒绝，
共同val使用固定可信子集；测试不读取研究数据、模型权重、远程文件或API。
"""
import copy
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from rdkit import Chem

from .data import identity
from .h_hydrogen_order import annotation
from .h_labels import make_key
from .h_replaytrain import grouped_objective
from .h_reviewed_supervision import apply_review, build_patches, common_validation, reviewed_objective


class ReviewedSupervisionTests(unittest.TestCase):
    def fixture(self):
        smi = next(s for s in ('CCO', 'CCCO', 'CCCCO', 'CCCOC') if identity(s)[2] == 'train')
        mol = Chem.AddHs(Chem.MolFromSmiles(smi))
        vals = [float(a.GetNeighbors()[0].GetIdx()+1) if a.GetAtomicNum()==1 else None for a in mol.GetAtoms()]
        old = dict(record_id='a', source_ref='synthetic', smiles=smi, data_nature='experimental', nucleus='1H',
                   atom_targets=vals, atom_target_mask=[v is not None for v in vals],
                   atom_order=dict(source_to_canonical=list(range(len(vals))),
                                   atomic_numbers=[a.GetAtomicNum() for a in mol.GetAtoms()]))
        old['spectrum'] = [dict(atom_index=i, shift=v) for i,v in enumerate(vals) if v is not None]
        i = next(i for i,v in enumerate(vals) if v is not None)
        j = next(j for j,v in enumerate(vals) if v is not None and v != vals[i])
        new=copy.deepcopy(old);new['atom_targets'][i],new['atom_targets'][j]=vals[j],vals[i]
        new['spectrum']=[dict(atom_index=k,shift=v) for k,v in enumerate(new['atom_targets']) if v is not None]
        patch=dict(record_id='a',split='train',source_indices=[i,j],before=[vals[i],vals[j]],after=[vals[j],vals[i]])
        new['assignment_correction']=patch
        smi,ann=annotation(old,'coordinate_order');key=('experimental','synthetic','a',smi)
        row=dict(domain=key[0],source=key[1],record_id='a',smiles=smi,split='train',q=np.arange(128,dtype=np.float32))
        labels={key:ann};patches=build_patches([patch],{'a':old},{'a':new},labels)
        artifact=dict(patches=patches,decisions={key:dict(split='train',experimental_spectral_supervision=False)})
        return old,new,patch,labels,row,artifact

    def test_corrections_and_masks_preserve_original_spectrum_and_exposure(self):
        old,new,patch,labels,row,artifact=self.fixture();key=make_key(row);before=copy.deepcopy((old,new,labels))
        for mode in ('legacy','corrections','atom','domain'):
            bank=copy.deepcopy(labels);rs,audit=apply_review(bank,[row],artifact,mode)
            self.assertEqual(make_key(rs[0]),key);np.testing.assert_array_equal(rs[0]['q'],row['q'])
            self.assertEqual(rs[0]['review_spectral_weight'],0. if mode=='domain' else 1.)
            self.assertEqual(audit['patched_records'],0 if mode=='legacy' else 1)
            if mode in ('atom','domain'):self.assertEqual(bank[key]['groups'],{})
            elif mode=='legacy':self.assertEqual(bank[key]['groups'],labels[key]['groups'])
            else:self.assertEqual(bank[key]['groups'],artifact['patches'][key]['annotation']['groups'])
        self.assertEqual((old,new,labels),before)

    def test_rejects_stale_labels_changed_mask_or_spectrum_and_wrong_training_scope(self):
        old,new,patch,labels,row,artifact=self.fixture()
        for what in ('labels','mask','values'):
            bank=copy.deepcopy(labels);candidate=copy.deepcopy(new)
            if what=='labels':bank[make_key(row)]['groups']={}
            elif what=='mask':candidate['atom_target_mask'][-1]=False
            else:candidate['atom_targets'][-1]=999.
            with self.assertRaises(ValueError):build_patches([patch],{'a':old},{'a':candidate},bank)
        for domain,split in [('theory','train'),('experimental','test'),('experimental','val')]:
            with self.assertRaises(ValueError):apply_review(copy.deepcopy(labels),[dict(row,domain=domain,split=split)],artifact,'domain')
        with self.assertRaises(ValueError):apply_review(copy.deepcopy(labels),[],artifact,'domain')

    def test_common_validation_held_subset_is_identical_for_all_arms(self):
        _,_,_,_,row,_=self.fixture();row=dict(row,split='val');other=dict(row,record_id='b')
        artifact=dict(decisions={make_key(row):dict(split='val',experimental_spectral_supervision=False)})
        rs,audit=common_validation([row,other],artifact);self.assertEqual(rs,[other]);self.assertEqual(audit['held'],1)
        with self.assertRaises(ValueError):common_validation([dict(row,split='test')],artifact)

    def loss_fixture(self,weights):
        v=torch.tensor([.1,.6],requires_grad=True);q=v[:,None].expand(-1,128)
        z=torch.zeros(6,6);mask=torch.zeros(6,6,dtype=torch.bool)
        b=SimpleNamespace(num_graphs=2,x=torch.zeros(6,99),batch=torch.tensor([0,0,0,1,1,1]),
            ptr=torch.tensor([0,3,6]),q=torch.tensor([.2,.3])[:,None].expand(-1,128),
            windows=torch.zeros(2,8,2),window_mask=torch.zeros(2,8,dtype=torch.bool),coverage_kind=torch.tensor([2,2]),
            site_targets=z,site_mask=mask,site_expected=torch.zeros(6,dtype=torch.long),
            group_targets=z,group_mask=mask,group_expected=torch.zeros(6,dtype=torch.long),
            group_owner=torch.tensor([0,1,2,0,1,2]),group_site_weight=torch.ones(6),
            shape_supervision=torch.ones(2,dtype=torch.bool),supervised_q=torch.tensor([.2,.3])[:,None].expand(-1,128),
            review_spectral_weight=torch.tensor(weights))
        d=dict(values=v,weights=torch.ones(2),owners=torch.tensor([0,3]),groups=torch.tensor([0,1]),quantiles=q)
        return v,b,d

    def test_full_weight_matches_legacy_and_held_spectrum_has_exact_zero_gradient(self):
        v,b,d=self.loss_fixture([1.,1.]);original=grouped_objective(d,b);new=reviewed_objective(d,b)
        for k in original:torch.testing.assert_close(original[k],new[k],atol=0,rtol=0)
        v,b,d=self.loss_fixture([1.,0.]);parts=reviewed_objective(d,b);parts['loss'].backward()
        self.assertNotEqual(v.grad[0].item(),0.);self.assertEqual(v.grad[1].item(),0.)
        v,b,d=self.loss_fixture([0.,0.]);loss=reviewed_objective(d,b)['loss'];loss.backward()
        self.assertEqual(loss.item(),0.);torch.testing.assert_close(v.grad,torch.zeros_like(v))
        b.group_mask[3,0]=True
        with self.assertRaises(ValueError):reviewed_objective(d,b)


if __name__=='__main__':unittest.main()
