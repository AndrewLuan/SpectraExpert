"""功能概述：资源恢复准入checker的非作者纯metadata边界核验。
不读实际结果/预测/张量；确认完整旧checker失败必须传播，不能生成新准入，
并拒科学CLI扩展、JJ SSH bool与实现review的双alias冲突。
"""
import copy
from pathlib import Path
import unittest
from unittest.mock import patch
import review_resource_ready_evaluation_admission as c
from test_review_resource_ready_evaluation_admission import ResourceFixture

class IndependentAdmissionTests(unittest.TestCase):
    def test_prior_full_checker_failure_is_not_bypassed(self):
        f=ResourceFixture()
        with patch.object(c,'BINDING_SHA',f.old.bp['sha256']),patch.object(c.base,'verify',side_effect=ValueError('prior_full_guard_rejected')) as prior:
            with self.assertRaisesRegex(ValueError,'prior_full_guard_rejected'):c.verify(f.b,f.request)
        prior.assert_called_once()
        request=prior.call_args.args[1]
        self.assertEqual(request['schema'],'cap10_native_cost_evaluation_admission_request_20261025')
        for phase in ('validation','mechanism'):
            self.assertEqual(request['queues'][phase]['path'],str(f.root/(phase+'_native_cost_queue.json')))
            self.assertEqual(request['runs'][phase],str(f.root/'runs'/(phase+'_native_cost')))

    def test_scientific_CLI_extra_option_not_permitted(self):
        f=ResourceFixture();f.b.docs[f.qrefs['validation']['path']]['jobs'][0]['command']+=['--new-grid','10']
        with self.assertRaisesRegex(ValueError,'only_output_access_changed_no_scientific_change'):f.queue()

    def test_false_SSH_exit_not_integer_zero(self):
        f=ResourceFixture();f.b.docs[f.request['reader_JJ_execution']['path']]['actual_ssh_exit']=False
        with self.assertRaisesRegex(ValueError,'strict_actual_exit0'):c.verify_JJ(f.b,f.request,'reader',f.readers,43)

    def test_operator_dual_alias_conflicting_SHA_rejected(self):
        f=ResourceFixture();pins=dict(f.operators)
        key=next(iter(pins));local=str(Path(f.protocol['local_root'])/Path(key).relative_to(f.root));pins[local]='b'*64
        with self.assertRaisesRegex(ValueError,'operator_current_SHA'):
            c.reviewed_alias(dict(status='PASS',independent=True,reviewed_sha256=pins),f.operators,f.root,f.protocol,'operator')
