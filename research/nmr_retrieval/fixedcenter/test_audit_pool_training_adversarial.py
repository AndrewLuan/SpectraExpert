"""功能概述：独立验证训练审计的真实退出证据和模型容量门禁。
ReceiptTests 使用作者的完整16任务临时fixture，只对exit类型和UTC作单点破坏；
CapacityTests 用合成tensor检查末轮/最佳模型的shape和dtype保持原网络结构；
ScheduleTests 从pin训练源码AST提取纯training_schedule，以两seed与两aug模式
交叉验证独立曝光重放。所有测试仅处理合成身份和tensor，不读取实际谱、不训练。
"""
import ast
import collections
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np
import torch
import test_audit_pool_training as t
import audit_pool_training as a


class ReceiptTests(unittest.TestCase):
    def test_boolean_exit_rejected(self):
        def mutate(root,jobs):
            path=root/'runs/formal'/(jobs[0]['id']+'.receipt.json')
            receipt=json.loads(path.read_text());receipt['internal_exit']=False
            path.write_text(json.dumps(receipt))
        with self.assertRaises(ValueError):t.FilesystemTests().run_tree(mutate)

    def test_invalid_utc_rejected(self):
        def mutate(root,jobs):
            path=root/'runs/formal'/(jobs[0]['id']+'.receipt.json')
            receipt=json.loads(path.read_text());receipt['finished_utc']='synthetic'
            path.write_text(json.dumps(receipt))
        with self.assertRaises(ValueError):t.FilesystemTests().run_tree(mutate)


class CapacityTests(unittest.TestCase):
    def test_network_shape_preserved(self):
        values=t.fixture()
        values[3]['last']['state_dict']['network.weight']=torch.ones((2,1))
        with self.assertRaises(ValueError):a.audit_model(*values)

    def test_network_dtype_preserved(self):
        values=t.fixture()
        values[3]['best']['state_dict']['network.weight']=torch.ones((1,2),dtype=torch.float64)
        with self.assertRaises(ValueError):a.audit_model(*values)


class ScheduleTests(unittest.TestCase):
    def test_actual_schedule_matches_independent_replay(self):
        source=next(p for p in (Path(__file__).resolve().parents[2]/'nmr_transfer/h_pool_ranker.py',
            Path('/Users/andrewluan/Desktop/Projects/IRfingerprints/SpectrExpert/nmr_transfer/h_pool_ranker.py')) if p.is_file())
        tree=ast.parse(source.read_text())
        func=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='training_schedule')
        scope={'np':np};exec(compile(ast.Module(body=[func],type_ignores=[]),str(source),'exec'),scope)
        rows=t.ExposureTests().rows()
        for seed in (193,211):
            for augment in (False,True):
                with self.subTest(seed=seed,augment=augment):
                    groups=collections.defaultdict(list)
                    for row in rows:
                        if not augment and row['variant']!='observed':continue
                        h=hashlib.sha256(('ranker_internal_317'+str(row['group'])).encode()).hexdigest()
                        if int(h[:12],16)%5:groups[row['group']].append(row)
                    names=sorted(groups);identity=hashlib.sha256();mixed=hashlib.sha256()
                    for epoch in range(1,9):
                        for row,pool in scope['training_schedule'](names,groups,seed,epoch,10):
                            identity.update(json.dumps(tuple(row[k] for k in a.KEY_FIELDS),separators=(',',':')).encode()+b'\n')
                            mixed.update(bytes([pool]))
                    replay,_,_=a.recompute_exposure(rows,seed,augment,epochs=8,samples=10)
                    self.assertEqual(replay[8]['exposure'],identity.hexdigest())
                    self.assertEqual(replay[8]['mixed'],mixed.hexdigest())


if __name__=='__main__':unittest.main()
