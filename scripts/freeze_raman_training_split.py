#!/usr/bin/env python3
"""功能概述：从冻结 Raman 10k 候选池划出 500 个理论验证组，保护全部实验测试身份。
主要职责：build_split 按 seed 哈希排序非测试组并生成互斥 train/validation ID；main 校验
原采样 manifest 和数据哈希，以独占写入方式新增 training_split.json，不修改光谱或采样。
代码库关系：消费 sample_raman_closed_library 的 sample_manifest/rows，输出真实的
raman_closed_library_train_v1 schema；训练器需显式适配该 schema，不伪装 scaffold split。
输入输出与副作用：只读已冻结文件，生成一个新清单；数量、唯一性、包含关系或哈希错误即失败。
"""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def build_split(rows, protected, seed=0, count=500):
    groups = {r['group_hash']:r['mol_id'] for r in rows}
    ids = {r['mol_id'] for r in rows}
    protected = set(protected)
    if len(groups) != len(rows) or len(ids) != len(rows) or not protected <= groups.keys():
        raise ValueError('non-unique IDs/groups or missing protected group')
    pool = sorted(groups.keys()-protected, key=lambda g:(hashlib.sha256(f'{seed}:validation-groups:{g}'.encode()).hexdigest(),g))
    if not 0 < count < len(pool):
        raise ValueError('invalid validation count')
    validation_groups = set(pool[:count])
    training = [r['mol_id'] for r in rows if r['group_hash'] not in validation_groups]
    validation = [r['mol_id'] for r in rows if r['group_hash'] in validation_groups]
    assert len(training)+len(validation)==len(rows)
    assert not set(training)&set(validation)
    assert {groups[g] for g in protected} <= set(training)
    return dict(schema_version='raman_closed_library_train_v1',seed=seed,
                train=training,validation=validation,test=[],
                validation_groups=sorted(validation_groups),protected_test_groups=sorted(protected),
                candidate_count=len(rows),train_count=len(training),validation_count=len(validation),
                policy='seeded group hash; sample validation only from non-experimental-test groups')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',type=Path,required=True)
    args=p.parse_args()
    root=args.data_root
    manifest_path=root/'sample_manifest.json'
    expected='9262856ad71db4802ca103261e91507f23969afab83cadf5f4ef7c64899a2ecb'
    if digest(manifest_path)!=expected:
        raise ValueError('wrong frozen sample manifest')
    manifest=json.loads(manifest_path.read_text())
    folder=root/'theory_train_candidate'
    for name,wanted in manifest['outputs_sha256']['theory_train_candidate'].items():
        if digest(folder/name)!=wanted:
            raise ValueError(f'source hash mismatch: {name}')
    rows=json.loads((folder/'rows.json').read_text())
    if len(rows)!=10000 or len(manifest['test_groups'])!=200:
        raise ValueError('expected frozen 10000/200 dataset')
    result=build_split(rows,manifest['test_groups'])
    result.update(sample_manifest_sha256=expected,source_rows_sha256=digest(folder/'rows.json'),
                  source_sha256={n:digest(folder/n) for n in ('spectra.parquet','smiles.parquet','wavenumber.parquet')},
                  data_dir=str(folder),script_sha256=digest(Path(__file__)))
    path=root/'training_split.json'
    with path.open('x') as f:
        f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(path=str(path),sha256=digest(path),train=9500,validation=500,protected_test_groups=200)))


if __name__=='__main__':
    main()
