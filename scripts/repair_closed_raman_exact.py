"""功能概述：修复闭库抽样中的完整结构不匹配，保留原版本并生成精确覆盖的新版本。
主要职责：canonical 保留同位素和立体信息；select_changes 用固定哈希选择修复谱和替代组；
verify_sources 将选中缓存逐条核对源分片；repair 校验父数据、写新版数据/划分/审计和 kNN 索引。
代码库关系：复用 sample_raman_closed_library 的谱质量与写入函数、freeze_raman_training_split
的划分规则及正式训练器校验器。缓存只用于选择，原理论/实验分片是最终记录事实来源。
输入输出与副作用：读取显式数据和缓存路径，不按模型分数采样；输出必须不存在，全部验证通过
才发布 staging 目录。候选槽位和测试顺序不变，无法精确补齐或来源不一致即失败，不覆盖旧产物。
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd
from rdkit import Chem

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sample_raman_closed_library import file_hash, ordered, usable, write_dataset
from freeze_raman_training_split import build_split
from spectraexpert.data.closed_library_split import validate_closed_library_split


def canonical(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError('invalid source SMILES')
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def record_key(row, seed):
    value = f"{seed}:exact-record:{row['source_split']}:{row['source_line']}"
    return hashlib.sha256(value.encode()).hexdigest(), row['source_path'], row['source_line']


def select_changes(candidates, queries, cache, seed=0):
    """返回完整新列表和变更记录；不修改调用方列表，不读取或使用检索分数。"""
    candidates, queries = list(candidates), list(queries)
    positions = {r['mol_id']: i for i, r in enumerate(candidates)}
    theory, experiment = {}, {}
    for role, dest in [('candidate', theory), ('query', experiment)]:
        for row in cache[role]:
            if usable(row['spectrum'], row['coverage']):
                dest.setdefault((row['group_hash'], canonical(row['smiles'])), []).append(row)
    changes, missing = [], []
    for qi, query in enumerate(queries):
        ci = positions[query['mol_id']]
        old = candidates[ci]
        if canonical(old['smiles']) == canonical(query['smiles']):
            continue
        choices = theory.get((query['group_hash'], canonical(query['smiles'])), [])
        if choices:
            new = dict(min(choices, key=lambda r: record_key(r, seed)), mol_id=old['mol_id'])
            candidates[ci] = new
            changes.append(dict(kind='theory_exact_repair', query_index=qi, mol_id=old['mol_id'],
                                old_smiles=old['smiles'], new_smiles=new['smiles']))
        else:
            missing.append((qi, ci))
    # 禁止从保留的候选组选择替代目标，也不重新使用刚移出的组。
    occupied = {r['group_hash'] for r in candidates}
    pairs = set(theory) & set(experiment)
    by_group = {}
    for group, smiles in pairs:
        if group not in occupied:
            by_group.setdefault(group, []).append(smiles)
    pool = ordered(by_group, seed, 'exact-replacement-groups')
    if len(pool) < len(missing):
        raise ValueError('insufficient independent exact replacement groups')
    for (qi, ci), group in zip(missing, pool):
        smiles = ordered(by_group[group], seed, 'exact-replacement-smiles')[0]
        old_query = queries[qi]
        mid = old_query['mol_id']
        candidates[ci] = dict(min(theory[group, smiles], key=lambda r: record_key(r, seed)), mol_id=mid)
        queries[qi] = dict(min(experiment[group, smiles], key=lambda r: record_key(r, seed)), mol_id=mid)
        changes.append(dict(kind='test_replacement', query_index=qi, mol_id=mid,
                            old_group=old_query['group_hash'], new_group=group,
                            old_smiles=old_query['smiles'], new_smiles=queries[qi]['smiles']))
    canon = [canonical(r['smiles']) for r in candidates]
    if len(set(canon)) != len(candidates):
        raise ValueError('duplicate complete candidate structures')
    for query in queries:
        candidate = candidates[positions[query['mol_id']]]
        if canonical(candidate['smiles']) != canonical(query['smiles']) or candidate['group_hash'] != query['group_hash']:
            raise ValueError('exact query/candidate identity mismatch')
    return candidates, queries, sorted(changes, key=lambda c: c['query_index'])


def verify_sources(records):
    paths = {}
    for row in records:
        paths.setdefault(row['source_path'], {}).setdefault(row['source_line'], []).append(row)
    audit = []
    for name, wanted in sorted(paths.items()):
        seen, sha = set(), hashlib.sha256()
        with gzip.open(name, 'rb') as handle:
            for line_no, raw in enumerate(handle, 1):
                sha.update(raw)
                if line_no not in wanted:
                    continue
                actual = json.loads(raw)
                for row in wanted[line_no]:
                    for key in ('smiles', 'group_hash', 'source', 'condition', 'coverage'):
                        if actual.get(key) != row.get(key):
                            raise ValueError(f'source metadata mismatch: {name}:{line_no}:{key}')
                    if not np.array_equal(np.asarray(actual['spectrum'], dtype=np.float64), row['spectrum']):
                        raise ValueError(f'source spectrum mismatch: {name}:{line_no}')
                seen.add(line_no)
        if seen != set(wanted):
            raise ValueError('source line missing')
        audit.append(dict(path=name, decompressed_sha256=sha.hexdigest(), checked_lines=sorted(seen)))
        print(json.dumps({'verified_source': name, 'selected_lines': len(seen)}), flush=True)
    return audit


def read_records(path):
    result = []
    with gzip.open(path, 'rt') as handle:
        for line in handle:
            row = json.loads(line)
            row['spectrum'] = np.asarray(row['spectrum'], dtype=np.float64)
            result.append(row)
    return result


def repair(parent, pilot, source_root, output):
    parent, pilot, source_root, output = [p.resolve() for p in (parent, pilot, source_root, output)]
    if output.exists():
        raise FileExistsError(output)
    sample = json.loads((parent/'sample_manifest.json').read_text())
    for role, files in sample['outputs_sha256'].items():
        for name, expected in files.items():
            if file_hash(parent/role/name) != expected:
                raise ValueError(f'parent hash mismatch: {role}/{name}')
    if file_hash(pilot/'records.json') != sample['pilot_records_sha256']:
        raise ValueError('unbound pilot records')
    if file_hash(pilot/'protocol.json') != sample['pilot_protocol_sha256']:
        raise ValueError('unbound pilot protocol')
    candidates = read_records(parent/'theory_train_candidate/records.jsonl.gz')
    queries = read_records(parent/'experimental_test/records.jsonl.gz')
    if len(candidates) != 10000 or len(queries) != 200:
        raise ValueError('requires original 10000/200 data')
    cache = json.loads((pilot/'records.json').read_text())
    with np.load(pilot/'overlap_arrays.npz', allow_pickle=False) as arrays:
        for role, domain in [('candidate', 'theory'), ('query', 'exp')]:
            matrix = arrays[role]
            for i, row in enumerate(cache[role]):
                row['spectrum'] = matrix[i]
                row['source_path'] = str(source_root/'combined'/domain/'raman'/f"{row['split']}.jsonl.gz")
                row['source_split'] = row.pop('split')
                row['source_line'] = row.pop('line')
    new_candidates, new_queries, changes = select_changes(candidates, queries, cache, sample['seed'])
    changed_ids = {r['mol_id'] for r in changes}
    replaced = {r['query_index'] for r in changes if r['kind'] == 'test_replacement'}
    if len(changes) != 21 or len(replaced) != 8:
        raise ValueError('parent repair differs from audited 13 repairs / 8 replacements')
    selected = [r for r in new_candidates if r['mol_id'] in changed_ids] + [new_queries[i] for i in sorted(replaced)]
    audit = verify_sources(selected)
    staging = Path(tempfile.mkdtemp(prefix=output.name+'.staging-', dir=output.parent))
    try:
        hashes = {}
        for role, rows in [('theory_train_candidate', new_candidates), ('experimental_test', new_queries)]:
            hashes[role] = write_dataset(staging/role, rows, [r['mol_id'] for r in rows])
        sample.update(candidate_groups=[r['group_hash'] for r in new_candidates],
                      test_groups=[r['group_hash'] for r in new_queries], outputs_sha256=hashes,
                      identity='complete RDKit canonical isomeric SMILES; isotopes retained',
                      sampling='parent slots retained; exact record repair; seed hash ordered new exact-paired groups',
                      dataset_revision='exact_v2', parent_data_root=str(parent),
                      parent_manifest_sha256=file_hash(parent/'sample_manifest.json'),
                      repair_script_sha256=file_hash(Path(__file__)))
        report = dict(parent_manifest_sha256=sample['parent_manifest_sha256'], changes=changes,
                      source_audit=audit, cache_arrays_sha256=file_hash(pilot/'overlap_arrays.npz'),
                      exact_coverage=200, unique_canonical_candidates=10000,
                      preserved_query_indices=[i for i in range(200) if i not in replaced],
                      replaced_query_indices=sorted(replaced), selection_uses_retrieval_scores=False)
        (staging/'repair_report.json').write_text(json.dumps(report, indent=2)+'\n')
        sample['repair_report_sha256'] = file_hash(staging/'repair_report.json')
        (staging/'sample_manifest.json').write_text(json.dumps(sample, indent=2)+'\n')
        split = build_split(new_candidates, sample['test_groups'], sample['seed'], 500)
        split.update(sample_manifest_sha256=file_hash(staging/'sample_manifest.json'),
                     source_rows_sha256=hashes['theory_train_candidate']['rows.json'],
                     source_sha256={n:hashes['theory_train_candidate'][n] for n in ('spectra.parquet','smiles.parquet','wavenumber.parquet')},
                     data_dir=str(output/'theory_train_candidate'), script_sha256=file_hash(Path(__file__)))
        (staging/'training_split.json').write_text(json.dumps(split, indent=2)+'\n')
        validate_closed_library_split(staging/'theory_train_candidate', split, split['source_sha256'],
                                      pd.read_parquet(staging/'theory_train_candidate/smiles.parquet'))
        # 独立校验实际 Parquet 与源谱插值；保留查询逐点相等。
        axis = np.arange(500., 4001.)
        for role, rows in [('theory_train_candidate', new_candidates), ('experimental_test', new_queries)]:
            data = pd.read_parquet(staging/role/'spectra.parquet').set_index('mol_id')
            for row in rows:
                expected = np.interp(axis, np.linspace(0,4000,3000), row['spectrum']).astype(np.float32)
                if not np.array_equal(data.loc[row['mol_id']].to_numpy(dtype=np.float32), expected):
                    raise ValueError('written spectral values differ')
        old_q = pd.read_parquet(parent/'experimental_test/spectra.parquet').set_index('mol_id')
        new_q = pd.read_parquet(staging/'experimental_test/spectra.parquet').set_index('mol_id')
        for i in report['preserved_query_indices']:
            mid = queries[i]['mol_id']
            if not np.array_equal(old_q.loc[mid], new_q.loc[mid]) or queries[i]['smiles'] != new_queries[i]['smiles']:
                raise ValueError('preserved query changed')
        matrix = pd.read_parquet(staging/'theory_train_candidate/spectra.parquet').drop(columns='mol_id').to_numpy(dtype=np.float32)
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
        np.savez_compressed(staging/'knn_candidate_index.npz', embeddings=matrix,
                            mol_ids=np.array([r['mol_id'] for r in new_candidates]),
                            canonical_smiles=np.array([canonical(r['smiles']) for r in new_candidates]),
                            sample_manifest_sha256=np.array(file_hash(staging/'sample_manifest.json')))
        (staging/'index_status.json').write_text(json.dumps(dict(knn='rebuilt raw intensity cosine index',
            knn_sha256=file_hash(staging/'knn_candidate_index.npz'),
            model='requires a checkpoint trained on this revised dataset; old model index not reused'), indent=2)+'\n')
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging)
        raise
    print(json.dumps(dict(output=str(output), exact_coverage=200, theory_repairs=13, test_replacements=8,
                         manifest_sha256=file_hash(output/'sample_manifest.json'))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('parent', 'pilot', 'source-root', 'output'):
        parser.add_argument('--'+flag, type=Path, required=True)
    args = parser.parse_args()
    repair(args.parent, args.pilot, args.source_root, args.output)
