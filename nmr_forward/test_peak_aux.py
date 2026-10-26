# 功能概述：检验峰组辅助损失的排序匹配、积分资格、缺失 J 掩码与梯度边界。
# 主要职责：PeakAuxTests 检查排序不变性、整数积分复制、未知多重性掩码、J大小、
# dd/tt阶段置换、dt/td顺序、矛盾J数量及覆盖计数；验证损失与梯度排除无效标签。
# 联合指标要求位移与类别同时正确，验证条件化J误差和覆盖分母不改变独立损失。
# 代码库关系：依赖 peak_aux、multiplets、PyTorch、unittest；支持包调用与 discovery。
# 输入输出与副作用：确定性 CPU 小张量，无文件修改、网络或真实模型训练。
import unittest
import torch
try:
    from .peak_aux import peak_aux_loss
    from .multiplets import MULTIPLICITY_CLASSES
except ImportError:
    from peak_aux import peak_aux_loss
    from multiplets import MULTIPLICITY_CLASSES


class PeakAuxTests(unittest.TestCase):
    def setup_case(self):
        out = dict(shifts=torch.tensor([[[4.,1.,8.,99.]]], requires_grad=True),
                   proton_mask=torch.tensor([[[True,True,True,False]]]),
                   mult_logits=torch.zeros(1,1,4,len(MULTIPLICITY_CLASSES),requires_grad=True),
                   couplings=torch.zeros(1,1,4,3,requires_grad=True))
        rows = [dict(protons=3,integral=3,annotations=[
            dict(shift_mid=8.,integration_h=1,multiplicity='m',coupling_hz=[]),
            dict(shift_mid=1.,integration_h=2,multiplicity='dd',coupling_hz=[8.,2.])])]
        return out, dict(rows=rows)

    def test_sort_match_integral_replication_and_gradients(self):
        out,batch=self.setup_case()
        result=peak_aux_loss(out,batch)
        self.assertEqual(result['counts']['eligible_records'],1)
        self.assertEqual(result['counts']['mult_slots'],3)
        self.assertEqual(result['counts']['j_values'],4)
        self.assertAlmostEqual(result['j_loss'].item(),4.5)
        result['loss'].backward()
        jgrad=out['couplings'].grad
        self.assertTrue((jgrad[0,0,:2,:2]<0).all())
        self.assertEqual(jgrad[0,0,:,2].abs().sum().item(),0)
        self.assertEqual(jgrad[0,0,2:].abs().sum().item(),0)
        m= MULTIPLICITY_CLASSES.index('m')
        dd=MULTIPLICITY_CLASSES.index('dd')
        self.assertLess(out['mult_logits'].grad[0,0,2,m].item(),0)
        self.assertLess(out['mult_logits'].grad[0,0,0,dd].item(),0)
        self.assertEqual(out['mult_logits'].grad[0,0,3].abs().sum().item(),0)

    def test_slot_permutation_invariance(self):
        out,batch=self.setup_case()
        out['couplings']=torch.arange(12.).reshape(1,1,4,3).requires_grad_()
        expected=peak_aux_loss(out,batch)
        order=torch.tensor([2,0,3,1])
        permuted={k:v[:,:,order] for k,v in out.items()}
        actual=peak_aux_loss(permuted,batch)
        torch.testing.assert_close(expected['loss'],actual['loss'])

    def test_missing_and_partial_j_preserve_positions(self):
        out,batch=self.setup_case()
        batch['rows'][0]['annotations'][1]['coupling_hz']=[None,2.]
        result=peak_aux_loss(out,batch)
        self.assertEqual(result['counts']['j_values'],2)
        self.assertEqual(result['counts']['invalid_j_values'],1)
        result['loss'].backward()
        self.assertEqual(out['couplings'].grad[...,0].abs().sum().item(),0)
        self.assertGreater(out['couplings'].grad[...,1].abs().sum().item(),0)

    def test_overflow_and_m_do_not_supervise_j(self):
        out,batch=self.setup_case()
        batch['rows'][0]['annotations'][1]['coupling_hz']=[1,2,3,4]
        batch['rows'][0]['annotations'][0]['coupling_hz']=[8]
        result=peak_aux_loss(out,batch)
        self.assertEqual(result['counts']['j_values'],0)
        self.assertEqual(result['counts']['j_overflow_groups'],1)
        self.assertEqual(result['counts']['mult_slots'],3)

    def test_invalid_integrals_and_slot_counts(self):
        for key,value in [('integral',2),('protons',4)]:
            out,batch=self.setup_case();batch['rows'][0][key]=value
            result=peak_aux_loss(out,batch)
            self.assertEqual(result['counts']['eligible_records'],0)
            self.assertEqual(result['loss'].item(),0)
            result['loss'].backward()
        out,batch=self.setup_case()
        batch['rows'][0]['annotations'][1]['integration_h']=1.5
        self.assertEqual(peak_aux_loss(out,batch)['counts']['skipped_integral'],1)
        out,batch=self.setup_case();out['proton_mask'][0,0,2]=False
        self.assertEqual(peak_aux_loss(out,batch)['counts']['skipped_slot_count'],1)

    def test_batched_gather_and_detached_metrics(self):
        out,batch=self.setup_case()
        out={k:v.detach().repeat(2,*([1]*(v.ndim-1))) for k,v in out.items()}
        for k in ('shifts','mult_logits','couplings'):
            out[k].requires_grad_()
        batch['rows']=batch['rows']*2
        result=peak_aux_loss(out,batch)
        self.assertEqual(result['counts']['eligible_records'],2)
        self.assertEqual(result['mult_slots'],6)
        self.assertEqual(result['j_values'],8)
        self.assertEqual(result['multiplicity_correct'].item(),0)
        self.assertEqual(result['j_abs_error_sum'].item(),40)
        self.assertFalse(result['j_abs_error_sum'].requires_grad)
        self.assertFalse(result['multiplicity_correct'].requires_grad)
        result['loss'].backward()
        torch.testing.assert_close(out['couplings'].grad[0],out['couplings'].grad[1])

    def test_nonfinite_record_is_skipped_without_nan_zero(self):
        out,batch=self.setup_case()
        out['shifts']=out['shifts'].detach()
        out['shifts'][0,0,0]=float('nan')
        out['shifts'].requires_grad_()
        result=peak_aux_loss(out,batch)
        self.assertEqual(result['counts']['skipped_nonfinite_shifts'],1)
        self.assertEqual(result['loss'].item(),0)
        result['loss'].backward()

    def test_optional_heads_and_weight(self):
        out,batch=self.setup_case()
        result=peak_aux_loss(out,batch,j_weight=.3)
        torch.testing.assert_close(result['loss'],result['multiplicity_loss']+.3*result['j_loss'])
        del out['mult_logits'];del out['couplings']
        result=peak_aux_loss(out,batch)
        self.assertEqual(result['loss'].item(),0)
        result['loss'].backward()
        self.assertEqual(result['counts']['j_values'],0)
        self.assertEqual(result['counts']['mult_slots'],0)

    def test_unknown_multiplicity_excluded_from_loss_and_accuracy(self):
        for unknown in (None, '', 'not parsed', 'other'):
            out,batch=self.setup_case()
            batch['rows'][0]['annotations'][1]['multiplicity']=unknown
            result=peak_aux_loss(out,batch)
            self.assertEqual(result['counts']['mult_slots'],1)
            self.assertEqual(result['counts']['unknown_mult_slots'],2)
            self.assertEqual(result['counts']['known_mult_groups'],1)
            self.assertEqual(result['counts']['unknown_mult_groups'],1)
            self.assertEqual(result['counts']['j_values'],0)
            result['loss'].backward()
            self.assertEqual(out['mult_logits'].grad[0,0,:2].abs().sum().item(),0)
            self.assertGreater(out['mult_logits'].grad[0,0,2].abs().sum().item(),0)

    def test_all_unknown_has_zero_supervision_and_finite_backward(self):
        out,batch=self.setup_case()
        for annotation in batch['rows'][0]['annotations']:
            annotation['multiplicity']=None
        result=peak_aux_loss(out,batch)
        self.assertEqual(result['loss'].item(),0)
        self.assertEqual(result['mult_slots'],0)
        self.assertEqual(result['multiplicity_correct'].item(),0)
        result['loss'].backward()
        self.assertEqual(out['mult_logits'].grad.abs().sum().item(),0)

    def test_j_magnitude_and_symmetric_stages(self):
        for mult in ('dd','tt'):
            for js in ([8.,2.],[2.,8.],[-2.,-8.]):
                out,batch=self.setup_case()
                a=batch['rows'][0]['annotations'][1]
                a.update(multiplicity=mult,coupling_hz=js)
                out['couplings']=torch.tensor([[[[8.,2.,9.],[8.,2.,9.],[0.,0.,0.],[0.,0.,0.]]]],requires_grad=True)
                result=peak_aux_loss(out,batch)
                self.assertEqual(result['j_loss'].item(),0)
                self.assertEqual(result['j_abs_error_sum'].item(),0)
                self.assertEqual(result['j_values'],4)
                self.assertEqual(result['counts']['j_supervised_groups'],1)

    def test_asymmetric_stages_remain_ordered(self):
        for mult in ('dt','td','dq','qd'):
            out,batch=self.setup_case()
            batch['rows'][0]['annotations'][1].update(multiplicity=mult,coupling_hz=[2.,8.])
            out['couplings']=torch.tensor([[[[8.,2.,9.],[8.,2.,9.],[0.,0.,0.],[0.,0.,0.]]]])
            result=peak_aux_loss(out,batch)
            self.assertEqual(result['j_abs_error_sum'].item(),24.)

    def test_partial_symmetric_j_matches_without_imputing(self):
        out,batch=self.setup_case()
        batch['rows'][0]['annotations'][1]['coupling_hz']=[None,8.]
        out['couplings']=torch.tensor([[[[8.,2.,9.],[8.,2.,9.],[0.,0.,0.],[0.,0.,0.]]]],requires_grad=True)
        result=peak_aux_loss(out,batch)
        self.assertEqual(result['j_values'],2)
        self.assertEqual(result['j_abs_error_sum'].item(),0.)
        self.assertEqual(result['counts']['invalid_j_values'],1)

    def test_j_count_contradictions_exclude_entire_group(self):
        for mult,js in [('s',[8.]),('d',[8.,2.]),('dd',[8.]),('tt',[8.,2.,1.])]:
            out,batch=self.setup_case()
            batch['rows'][0]['annotations'][1].update(multiplicity=mult,coupling_hz=js)
            result=peak_aux_loss(out,batch)
            self.assertEqual(result['j_values'],0)
            self.assertEqual(result['counts']['j_count_mismatch_groups'],1)
            self.assertEqual(result['counts']['j_supervised_groups'],0)
            self.assertEqual(result['mult_slots'],3)
            result['loss'].backward()
            self.assertEqual(out['couplings'].grad.abs().sum().item(),0)

    def test_missing_and_invalid_j_coverage(self):
        for js,missing,invalid in [(None,1,0),([],1,0),('8 Hz',0,1)]:
            out,batch=self.setup_case()
            batch['rows'][0]['annotations'][1]['coupling_hz']=js
            result=peak_aux_loss(out,batch)
            self.assertEqual(result['j_values'],0)
            self.assertEqual(result['counts']['j_supported_groups'],1)
            self.assertEqual(result['counts']['j_missing_groups'],missing)
            self.assertEqual(result['counts']['j_invalid_format_groups'],invalid)

    def test_joint_metrics_exclude_wrong_shift_or_class_keep_total_j(self):
        dd=MULTIPLICITY_CLASSES.index('dd')
        m=MULTIPLICITY_CLASSES.index('m')
        for wrong_shift,wrong_class,expected_joint_slots,expected_joint_j in (
                (False,False,3,4),(True,False,2,2),(False,True,2,2),(True,True,1,0)):
            out,batch=self.setup_case()
            # 排序匹配两个1ppm的dd和一个8ppm的m；前两槽J各贡献(8+2)Hz误差。
            out['shifts']=torch.tensor([[[1.05,1.01,8.,99.]]],requires_grad=True)
            out['mult_logits']=torch.zeros_like(out['mult_logits']).detach()
            out['mult_logits'][0,0,:2,dd]=5.
            out['mult_logits'][0,0,2,m]=5.
            if wrong_shift:
                out['shifts']=torch.tensor([[[1.3,1.01,8.,99.]]],requires_grad=True)
            if wrong_class:
                out['mult_logits'][0,0,1,dd]=0.
                out['mult_logits'][0,0,1,m]=5.
            out['mult_logits'].requires_grad_()
            result=peak_aux_loss(out,batch)
            self.assertEqual(result['joint_multiplicity_correct_at_0_1ppm'].item(),expected_joint_slots)
            self.assertEqual(result['joint_J_values'].item(),expected_joint_j)
            self.assertEqual(result['joint_J_abs_error_sum'].item(),expected_joint_j*5.)
            self.assertEqual(result['mult_slots'],3)
            self.assertEqual(result['j_values'],4)
            self.assertEqual(result['j_abs_error_sum'].item(),20.)
            self.assertAlmostEqual(result['j_loss'].item(),4.5)
            self.assertFalse(result['joint_J_abs_error_sum'].requires_grad)

    def test_joint_j_requires_multiplicity_head(self):
        out,batch=self.setup_case()
        del out['mult_logits']
        result=peak_aux_loss(out,batch)
        self.assertEqual(result['j_values'],4)
        self.assertEqual(result['joint_J_values'].item(),0)
        self.assertEqual(result['joint_J_abs_error_sum'].item(),0)
        self.assertEqual(result['joint_multiplicity_correct_at_0_1ppm'].item(),0)


if __name__ == '__main__':
    unittest.main()
