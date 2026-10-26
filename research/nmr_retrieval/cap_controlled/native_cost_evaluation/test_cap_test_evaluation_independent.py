"""功能概述：非作者独立复现 sealed adapter 的完整实际 query 身份守卫。

只用作者人工六行/512候选 fixture；不打开真实 test、预测或研究 tensor。
重放完整入口，要求 loaded rows 的六字段身份、顺序和 mask 内容由程序重算，
不能只靠 manifest 的三个自报 SHA。另检查最终独审在首个 cache hash 之前拒绝。
"""
import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cap_test_evaluation_common as c
import cap_test_evaluate_ranker as e
from test_cap_test_evaluate_ranker import SealedFixture


class IdentityGuardTests(unittest.TestCase):
    def test_loaded_group_change_must_not_keep_declared_identity_SHA(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=SealedFixture(tmp)
            f.data['rows'][0]['group']='changed_identity_only_no_model_input'
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError,'identity|content'):
                    f.run()

    def test_loaded_mask_change_must_not_keep_declared_mask_SHA(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=SealedFixture(tmp);f.data['rows'][0]['mask_content_sha256']='e'*64
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError,'mask|content'):
                    f.run()

    def test_loaded_query_order_reversal_must_not_keep_declared_identity_SHA(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=SealedFixture(tmp);f.data['rows']=list(reversed(f.data['rows']))
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError,'identity|content'):
                    f.run()

    def test_pruned_unrelated_final_review_fails_before_cache_first_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=SealedFixture(tmp)
            review=json.loads(f.review.read_text());review['reviewed_sha256']={str(f.protocol):c.sha(f.protocol,True)}
            c.write(f.review,review);original=c.sha;opened=[]
            def guarded(path,*args,**kwargs):
                if Path(path).suffix in ('.pt','.pth','.ckpt'):
                    opened.append(str(path));raise AssertionError('payload hash before final review')
                return original(path,*args,**kwargs)
            with patch.object(c,'sha',side_effect=guarded):
                with self.assertRaisesRegex(ValueError,'specific_test_freeze_review_complete_exact_SHA'):
                    f.gate()
            self.assertEqual(opened,[])


if __name__=='__main__':unittest.main()
