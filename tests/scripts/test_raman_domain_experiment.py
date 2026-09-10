"""功能概述：验证域迁移实验的缺失覆盖隔离、多正样本损失及训练谱差异拟合。
主要职责：掩码测试确保任意缺失区填充值不影响表示；损失测试校验重复身份不互为负样本；
仿射测试使用已知偏移谱验证训练配对拟合方向；增强测试检查输出掩码与有限值。
代码库关系：直接调用 raman_domain_experiment，不读取真实训练/测试数据或连接服务器。
输入输出与副作用：小型 CPU tensor 和合成谱，无磁盘数据、API 请求或正式训练。
"""
import numpy as np
import torch
import torch.nn.functional as F

from scripts.raman_domain_experiment import (AXIS, MaskedSpectrumEncoder, multi_positive_loss,
                                           coverage_mask, fit_domain, augment_domain)


def test_masked_values_cannot_change_embedding():
    torch.manual_seed(0)
    model=MaskedSpectrumEncoder(signal_length=65,patch_size=8,stride=4,embed_dim=16,depth=1,num_heads=2,projection_dim=8).eval()
    x=torch.randn(2,65);mask=torch.ones_like(x,dtype=torch.bool);mask[:,:20]=False
    changed=x.clone();changed[~mask]=float('nan')
    with torch.no_grad():
        a=model(x,mask);b=model(changed,mask)
    assert torch.allclose(a,b) and torch.isfinite(b).all()
    assert torch.allclose(a.norm(dim=1),torch.ones(2),atol=1e-5)


def test_multi_positive_does_not_repel_duplicate_identity():
    z=F.normalize(torch.randn(3,8),dim=1)
    assert abs(float(multi_positive_loss(z,z,torch.zeros(3,dtype=torch.long))))<1e-6
    logits=z@z.T/.1
    assert torch.allclose(multi_positive_loss(z,z,torch.arange(3)),F.cross_entropy(logits,torch.arange(3)),atol=1e-6)


def test_training_pair_shift_and_masked_augmentation():
    t=sum(a*np.exp(-.5*((AXIS-center)/6)**2) for a,center in ((1,800),(.7,1300),(.5,2100)))
    e=np.interp(AXIS-20,AXIS,t,left=0,right=0)
    mask=np.ones(len(t),dtype=bool)
    params,detail=fit_domain(t[None],mask[None],e[None],mask[None])
    assert detail[0]['correlation_aligned']>.99
    assert detail[0]['shift']==20
    assert not params['frequency_fit_enabled']  # 不用一个配对样本拟合总体增强范围。
    m=coverage_mask([375,2500])
    y,valid=augment_domain(t,m,params,np.random.default_rng(0),m[None])
    assert np.isfinite(y).all() and valid.any() and (y[~valid]==0).all()
