"""功能概述：校验 Raman 闭库训练划分与冻结采样身份、理论源文件的绑定。
主要职责：validate_closed_library_split 验证 sample/rows/source SHA-256、逐行 ID/group
映射、分区、计数和受保护实验身份；按原 seed 哈希规则核对验证组，返回后由训练器继续
检查 canonical identity 隔离并计算 checkpoint 身份哈希。_digest 分块读取文件。
代码库关系：消费 freeze_raman_training_split 的 raman_closed_library_train_v1；
不导入采样脚本、不读取实验谱。train_structure_aware 在建图或创建输出前调用。
输入输出与副作用：只读 data_dir 及其父目录 sample_manifest.json/rows.json，错误立即失败；
不修改冻结清单，不把用户约定的已见分子实验测试当作内部 validation。
"""
import hashlib
import json
from pathlib import Path


def _digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            value.update(block)
    return value.hexdigest()


def validate_closed_library_split(directory, manifest, source_hashes, smiles):
    directory = Path(directory)
    sample_path = directory.parent / 'sample_manifest.json'
    rows_path = directory / 'rows.json'
    if _digest(sample_path) != manifest.get('sample_manifest_sha256'):
        raise ValueError('closed-library sample manifest hash mismatch')
    if _digest(rows_path) != manifest.get('source_rows_sha256'):
        raise ValueError('closed-library source_rows_sha256 mismatch')
    sample = json.loads(sample_path.read_text())
    if sample.get('schema_version') != 'raman_closed_library_sample_v1':
        raise ValueError('closed-library sample schema mismatch')
    if manifest.get('source_sha256') != source_hashes:
        raise ValueError('closed-library source_sha256 mismatch')
    expected = sample.get('outputs_sha256', {}).get('theory_train_candidate', {})
    if any(expected.get(name) != value for name, value in source_hashes.items()) or expected.get('rows.json') != _digest(rows_path):
        raise ValueError('closed-library sample source hashes mismatch')
    rows = json.loads(rows_path.read_text())
    ids = [r['mol_id'] for r in rows]
    groups = [r['group_hash'] for r in rows]
    if any(type(i) is not int for i in ids) or len(set(ids)) != len(ids) or len(set(groups)) != len(groups):
        raise ValueError('closed-library requires unique IDs/groups')
    if any(not isinstance(g, str) or not g for g in groups):
        raise ValueError('closed-library invalid group identity')
    if groups != sample.get('candidate_groups') or len(rows) != sample.get('candidate_count'):
        raise ValueError('closed-library candidate identity/order mismatch')
    if set(ids) != set(smiles['mol_id']) or 'group_hash' not in smiles:
        raise ValueError('closed-library rows/SMILES ID mismatch')
    by_id = smiles.set_index('mol_id')
    for row in rows:
        other = by_id.loc[row['mol_id']]
        if other['group_hash'] != row['group_hash'] or other['smiles'] != row['smiles']:
            raise ValueError('closed-library rows/SMILES identity mismatch')
        if '/theory/raman/' not in row.get('source_path', '') or other['source_path'] != row['source_path']:
            raise ValueError('closed-library requires theoretical Raman sources')
    assigned = []
    for split in ('train', 'validation', 'test'):
        values = manifest.get(split)
        if not isinstance(values, list) or any(type(i) is not int for i in values):
            raise ValueError('closed-library split IDs must be integers')
        assigned.extend(values)
    if len(assigned) != len(set(assigned)) or set(assigned) != set(ids) or manifest['test']:
        raise ValueError('closed-library split must partition theory into train/validation')
    for key, count in [('candidate_count', len(rows)), ('train_count', len(manifest['train'])),
                       ('validation_count', len(manifest['validation']))]:
        if manifest.get(key) != count:
            raise ValueError('closed-library count mismatch: ' + key)
    protected = sample.get('test_groups')
    if not isinstance(protected, list) or len(protected) != len(set(protected)) or len(protected) != sample.get('test_count'):
        raise ValueError('closed-library invalid protected groups')
    if sorted(protected) != manifest.get('protected_test_groups'):
        raise ValueError('closed-library protected group declaration mismatch')
    group_by_id = dict(zip(ids, groups))
    train_groups = {group_by_id[i] for i in manifest['train']}
    val_groups = {group_by_id[i] for i in manifest['validation']}
    if not set(protected) <= train_groups or train_groups & val_groups:
        raise ValueError('closed-library protected groups must remain in train')
    seed = manifest.get('seed')
    if type(seed) is not int or seed != sample.get('seed'):
        raise ValueError('closed-library seed mismatch')
    ordered = sorted(set(groups) - set(protected), key=lambda g: (
        hashlib.sha256(f'{seed}:validation-groups:{g}'.encode()).hexdigest(), g))
    expected_groups = set(ordered[:manifest['validation_count']])
    if not val_groups or val_groups != expected_groups or sorted(val_groups) != manifest.get('validation_groups'):
        raise ValueError('closed-library validation groups differ from frozen policy')
