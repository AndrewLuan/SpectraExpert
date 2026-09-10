#!/usr/bin/env python3
"""功能概述：冻结实验 Raman 测试组与包含全部测试身份的理论训练/检索候选集。
主要职责：ordered 使用带种子的哈希排序；usable 固定谱质量条件；sample 从已核验的
重合组缓存抽测试组、按完整理论索引随机顺序抽候选组；write_dataset 写同源 JSONL 和
训练 Dataset 可读取的 Parquet 三件套；main 管理参数与仅新建输出目录。
代码库关系：复用 pilot_raman_overlap_knn 的 3000 点轴、只读索引和分组口径，输出
3501 点 Parquet 与现有 StructureAwareSpectralDataset 的数据字段一致。理论集同时作为
训练与 kNN 候选；实验谱独立保存，不产生或伪装为旧训练器的 scaffold split。
输入输出与副作用：读取明确的理论 Raman 分片和此前 pilot 自建缓存，不读取其他模态，
不修改源数据。抽样单位为索引最大分子片段/忽略立体化学的分子组，每组保留一条谱。
先创建 staging、通过包含关系和数量校验才发布；失败仅清理本次 staging。采样不训练。
"""
import argparse
import gzip
import hashlib
import json
import shutil
import sqlite3
import tempfile
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from pilot_raman_overlap_knn import AXIS, HASH, INDEX, REGION, ROOT, load_overlap


def ordered(values, seed, namespace):
    return sorted(values, key=lambda value: (hashlib.sha256(
        f'{seed}:{namespace}:{value}'.encode()).hexdigest(), value))


def usable(values, coverage):
    values = np.asarray(values, dtype=np.float32)
    if values.shape != (3000,) or not np.isfinite(values).all():
        return False
    if not isinstance(coverage, list) or len(coverage) != 2 or not 0 <= coverage[0] < coverage[1] <= 3000:
        return False
    mask = REGION & (np.arange(3000) >= coverage[0]) & (np.arange(3000) < coverage[1])
    return bool(mask.sum() >= np.ceil(REGION.sum()/2) and np.ptp(values[mask]) > 1e-10)


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_dataset(directory, records, ids):
    directory.mkdir()
    target = np.arange(500., 4001.)
    schema = pa.schema([('mol_id', pa.int64())] + [(f'w_{int(x)}', pa.float32()) for x in target])
    metadata = []
    with pq.ParquetWriter(directory/'spectra.parquet', schema, compression='zstd') as writer, \
            gzip.open(directory/'records.jsonl.gz', 'wt') as raw:
        for start in range(0, len(records), 256):
            batch = records[start:start+256]
            identifiers = ids[start:start+256]
            matrix = np.stack([np.interp(target, AXIS, r['spectrum']) for r in batch]).astype(np.float32)
            arrays = [pa.array(identifiers, type=pa.int64())] + [pa.array(matrix[:, i]) for i in range(3501)]
            writer.write_table(pa.Table.from_arrays(arrays, schema=schema))
            for identifier, row in zip(identifiers, batch):
                meta = {k: v for k, v in row.items() if k != 'spectrum'} | {'mol_id': identifier}
                metadata.append(meta)
                raw.write(json.dumps(meta | {'spectrum': np.asarray(row['spectrum']).tolist()}, allow_nan=False)+'\n')
    pq.write_table(pa.Table.from_pylist(metadata), directory/'smiles.parquet')
    pq.write_table(pa.table({'wavenumber': target.astype(np.float32)}), directory/'wavenumber.parquet')
    (directory/'rows.json').write_text(json.dumps(metadata, indent=2))
    return {name: file_hash(directory/name) for name in ('spectra.parquet','smiles.parquet','wavenumber.parquet','records.jsonl.gz','rows.json')}


def sample(root, index, pilot, output, candidate_count, test_count, seed):
    if not 0 < test_count <= candidate_count:
        raise ValueError('require 0 < test count <= candidate count')
    if output.exists():
        raise FileExistsError(output)
    overlap = load_overlap(index)
    pilot_protocol = json.loads((pilot/'protocol.json').read_text())
    if pilot_protocol['overlap'] != overlap:
        raise ValueError('pilot overlap differs from live index')
    rows = json.loads((pilot/'records.json').read_text())
    with np.load(pilot/'overlap_arrays.npz') as z:
        q, c = z['query'], z['candidate']
    exp, theory = {}, {}
    for kind, matrix, dest in [('query',q,exp),('candidate',c,theory)]:
        for i, row in enumerate(rows[kind]):
            if not usable(matrix[i], row['coverage']):
                continue
            record = dict(row, spectrum=matrix[i], source_path=str(root/'combined'/('exp' if kind=='query' else 'theory')/'raman'/f"{row['split']}.jsonl.gz"))
            dest.setdefault(row['group_hash'], []).append(record)
    eligible = set(exp) & set(theory) & set(overlap)
    if len(eligible) < test_count:
        raise ValueError('not enough eligible overlap groups')
    tests = ordered(eligible, seed, 'test-groups')[:test_count]
    with sqlite3.connect(f'file:{index}?mode=ro', uri=True) as con:
        all_theory = {r[0].lower() for r in con.execute('SELECT hex(group_hash) FROM molecule_groups WHERE theory_raman_count>0')}
    rest = ordered(all_theory-set(tests), seed, 'candidate-groups')
    needed = candidate_count-test_count
    pool = set(rest[:max(needed*2, needed+1000)]) | set(tests)
    best, priorities, audit = {}, {}, []
    for split in ('train','valid','test'):
        path = root/'combined'/'theory'/'raman'/f'{split}.jsonl.gz'
        digest = hashlib.sha256()
        counts = {'scanned':0, 'pool_records':0, 'unusable_records':0}
        with gzip.open(path, 'rb') as f:
            for line_no, line in enumerate(f, 1):
                digest.update(line)
                counts['scanned'] += 1
                match = HASH.search(line)
                if match is None:
                    raise ValueError(f'missing group_hash: {path}:{line_no}')
                group = match[1].decode()
                if group not in pool:
                    continue
                row = json.loads(line)
                counts['pool_records'] += 1
                values = np.asarray(row['spectrum'], dtype=np.float32)
                if not usable(values, row.get('coverage')):
                    counts['unusable_records'] += 1
                    continue
                priority = hashlib.sha256(f'{seed}:theory-record:{split}:{line_no}'.encode()).hexdigest()
                if group not in best or priority < priorities[group]:
                    best[group] = {k:row.get(k) for k in ('smiles','group_hash','source','condition','coverage')} | {
                        'spectrum':values, 'source_path':str(path), 'source_split':split, 'source_line':line_no}
                    priorities[group] = priority
        audit.append({'path':str(path), 'decompressed_sha256':digest.hexdigest(), 'counts':counts})
        print(json.dumps(audit[-1]), flush=True)
    if set(tests)-set(best):
        raise ValueError('required test identities lack usable theory spectra')
    extras = [g for g in rest if g in best][:needed]
    if len(extras) != needed:
        raise ValueError('oversampling pool exhausted; increase pool, never silently reduce size')
    candidates = ordered(set(tests) | set(extras), seed, 'candidate-order')
    if len(candidates) != candidate_count or not set(tests) <= set(candidates):
        raise AssertionError('candidate/test inclusion failure')
    identifiers = {g:i for i,g in enumerate(candidates)}
    queries = []
    for group in tests:
        choices = exp[group]
        chosen = min(choices, key=lambda r:hashlib.sha256(f"{seed}:exp-record:{r['split']}:{r['line']}".encode()).hexdigest())
        queries.append({k:chosen.get(k) for k in ('smiles','group_hash','source','condition','coverage','spectrum','source_path')} |
                       {'source_split':chosen['split'], 'source_line':chosen['line']})
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output.name+'.staging-', dir=output.parent))
    try:
        train_hashes = write_dataset(staging/'theory_train_candidate', [best[g] for g in candidates], list(range(candidate_count)))
        test_hashes = write_dataset(staging/'experimental_test', queries, [identifiers[g] for g in tests])
        report = {'schema_version':'raman_closed_library_sample_v1', 'seed':seed,
                  'candidate_count':candidate_count, 'test_count':test_count,
                  'indexed_theory_groups':len(all_theory), 'indexed_overlap_groups':len(overlap),
                  'eligible_overlap_groups':len(eligible), 'excluded_overlap_groups':sorted(set(overlap)-eligible),
                  'test_groups':tests, 'candidate_groups':candidates,
                  'test_is_subset_of_training_groups':True,
                  'training_and_knn_candidates':'theory_train_candidate', 'test_queries':'experimental_test',
                  'sampling':'seeded SHA256 group ordering, required tests + first usable random theory groups; one seeded record per group',
                  'quality':'finite 3000 points, coverage of >=50% of 500..4000 region, ptp>1e-10',
                  'identity':'index parent-connectivity groups, largest fragment, stereochemistry ignored',
                  'task':'seen-molecule theory-to-experiment closed-library retrieval; no experimental training inputs',
                  'parquet_axis':{'start':500,'stop':4000,'points':3501,'interpolation':'linear'},
                  'jsonl_axis':{'start':0,'stop':4000,'points':3000},
                  'coverage':'original 3000-point half-open interval; interpolation does not create measured coverage',
                  'training_internal_validation':'not assigned; do not use experimental test for checkpoint selection',
                  'inputs':audit, 'pilot_protocol_sha256':file_hash(pilot/'protocol.json'),
                  'pilot_records_sha256':file_hash(pilot/'records.json'),
                  'script_sha256':file_hash(Path(__file__)),
                  'outputs_sha256':{'theory_train_candidate':train_hashes,'experimental_test':test_hashes}}
        (staging/'sample_manifest.json').write_text(json.dumps(report, indent=2))
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging)
        raise
    print(json.dumps({k:v for k,v in report.items() if k not in ('candidate_groups','test_groups','inputs','outputs_sha256')}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--index', type=Path, default=INDEX)
    p.add_argument('--pilot-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--candidate-count', type=int, default=10000)
    p.add_argument('--test-count', type=int, default=200)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    started = time.perf_counter()
    sample(args.root,args.index,args.pilot_dir,args.output_dir,args.candidate_count,args.test_count,args.seed)
    print(json.dumps({'completed_seconds':time.perf_counter()-started}), flush=True)


if __name__ == '__main__':
    main()
