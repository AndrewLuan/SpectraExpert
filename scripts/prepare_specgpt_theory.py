"""功能概述：把 SpecGPT continuous_v1 理论 Raman 分片转换为现有训练器的 Parquet 三件套。
主要职责：prepare 从处理清单恢复真实波数轴，分批插值写谱，规范化 SMILES，检查 frozen
group/identity 跨 split 泄漏，过滤非法或常数谱；main 提供显式路径和小样本限制。
代码库关系：复用 structure_dataset 的 identity/hash 契约；train_structure_aware 消费输出
split.json，使用 specgpt_frozen_v1 标记外部分组划分，绝不冒充本项目 scaffold split。
输入输出与副作用：只读取清单内 theory/Raman 文件；保留重复分子的不同谱，赋予唯一行 ID。
输出先写同级临时目录，全部校验成功后发布；拒绝覆盖已有目录，失败清理本次临时产物。
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from spectraexpert.data.structure_dataset import canonicalize_smiles, canonical_smiles_sha256


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare(manifest_path, output_dir, *, max_records_per_file=None, batch_size=4096):
    manifest_path, output_dir = Path(manifest_path), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if batch_size < 1 or (max_records_per_file is not None and max_records_per_file < 1):
        raise ValueError('batch size and sample limit must be positive')
    source = json.loads(manifest_path.read_text())
    if source.get('processing_version') != 'continuous_v1' or source.get('domain') != 'theory':
        raise ValueError('requires continuous_v1 theory manifest')
    config = source['config']
    axis_config = config['modalities']['Raman']
    if axis_config['axis_unit'] != 'cm-1':
        raise ValueError('Raman axis must use cm-1')
    axis = np.linspace(axis_config['axis_min'], axis_config['axis_max'], config['grid_size'])
    target = np.arange(500., 4001.)
    if not np.isfinite(axis).all() or len(axis) < 2 or not np.all(np.diff(axis) > 0) or axis[0] > 500 or axis[-1] < 4000:
        raise ValueError('source axis must cover 500..4000 cm-1')
    entries = source['files']
    if not entries or any(e.get('domain') != 'theory' or e.get('modality') != 'Raman' for e in entries):
        raise ValueError('only theory Raman entries are allowed')
    paths = [Path(e['outputs'][s]) for e in entries for s in ('train', 'valid', 'test')]
    if len(set(p.resolve() for p in paths)) != len(paths):
        raise ValueError('duplicate source paths across manifest')
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output_dir.name + '.staging-', dir=output_dir.parent))
    rows, spectra, spectrum_ids = [], [], []
    assignments = {s: [] for s in ('train', 'validation', 'test')}
    identities, groups = {}, {}
    stats, inputs = Counter(), []
    schema = pa.schema([('mol_id', pa.int64())] + [(f'w_{int(w)}', pa.float32()) for w in target])
    writer = None
    try:
        writer = pq.ParquetWriter(staging / 'spectra.parquet', schema, compression='zstd')

        def flush():
            if not spectra:
                return
            matrix = np.stack(spectra)
            arrays = [pa.array(spectrum_ids, type=pa.int64())]
            arrays.extend(pa.array(matrix[:, i], type=pa.float32()) for i in range(len(target)))
            writer.write_table(pa.Table.from_arrays(arrays, schema=schema))
            spectra.clear()
            spectrum_ids.clear()

        for entry in entries:
            for split_name, split in [('train', 'train'), ('valid', 'validation'), ('test', 'test')]:
                path = Path(entry['outputs'][split_name])
                counts = Counter()
                with gzip.open(path, 'rt') as handle:
                    lines = handle if max_records_per_file is None else itertools.islice(handle, max_records_per_file)
                    for line_number, line in enumerate(lines, 1):
                        counts['read'] += 1
                        record = json.loads(line)
                        if record.get('modality') != 'raman' or record.get('domain', 'theory') != 'theory':
                            raise ValueError(f'non-theory Raman record: {path}:{line_number}')
                        group = record.get('group_hash')
                        if not isinstance(group, str) or not group:
                            raise ValueError(f'missing group_hash: {path}:{line_number}')
                        if groups.setdefault(group, split) != split:
                            raise ValueError('frozen group leakage across splits')
                        try:
                            canonical = canonicalize_smiles(record.get('smiles'))
                        except ValueError:
                            counts['invalid_smiles'] += 1
                            continue
                        if identities.setdefault(canonical, split) != split:
                            raise ValueError('canonical identity leakage across splits')
                        try:
                            values = np.asarray(record['spectrum'], dtype=np.float64)
                        except (KeyError, TypeError, ValueError):
                            counts['invalid_spectrum'] += 1
                            continue
                        if values.shape != axis.shape or not np.isfinite(values).all():
                            counts['invalid_spectrum'] += 1
                            continue
                        interpolated = np.interp(target, axis, values).astype(np.float32)
                        if np.ptp(interpolated) <= 1e-10:
                            counts['constant_spectrum'] += 1
                            continue
                        identifier = len(rows)
                        rows.append(dict(mol_id=identifier, smiles_canonical=canonical, group_hash=group,
                                         source_path=str(path), source_line=line_number, split=split,
                                         source=record.get('source', ''), coverage_json=json.dumps(record.get('coverage'))))
                        assignments[split].append(identifier)
                        spectra.append(interpolated)
                        spectrum_ids.append(identifier)
                        counts['accepted'] += 1
                        if len(spectra) >= batch_size:
                            flush()
                stats.update(counts)
                inputs.append(dict(path=str(path), split=split, counts=dict(counts),
                                   sha256=sha256(path) if max_records_per_file is None else None))
        flush()
        writer.close()
        writer = None
        if any(not values for values in assignments.values()):
            raise ValueError('every frozen split must retain usable spectra')
        pq.write_table(pa.Table.from_pylist(rows), staging / 'smiles.parquet')
        pq.write_table(pa.table({'wavenumber': target.astype(np.float32)}), staging / 'wavenumber.parquet')
        manifest = dict(schema_version='specgpt_frozen_v1', **assignments,
                        source_smiles_sha256=canonical_smiles_sha256([r['mol_id'] for r in rows], [r['smiles_canonical'] for r in rows]),
                        source_sha256={n: sha256(staging / n) for n in ('spectra.parquet', 'smiles.parquet', 'wavenumber.parquet')})
        metadata = dict(domain='theory', modality='Raman', source_manifest=str(manifest_path),
                        source_manifest_sha256=sha256(manifest_path), source_config=config,
                        split_fingerprint_sha256=source.get('split_fingerprint_sha256'),
                        target_axis=dict(start=500, stop=4000, points=3501), interpolation='linear',
                        max_records_per_file=max_records_per_file, inputs=inputs, counts=dict(stats))
        (staging / 'split.json').write_text(json.dumps(manifest, indent=2) + '\n')
        (staging / 'preparation_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        staging.rename(output_dir)
        return metadata
    except BaseException:
        if writer is not None:
            writer.close()
        shutil.rmtree(staging)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path('/data/SpecGPT/dataset/processed_stage_2_v1/continuous/manifests/processed_manifest_theory_raman.json'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--max-records-per-file', type=int)
    args = parser.parse_args()
    print(json.dumps(prepare(args.manifest, args.output_dir, max_records_per_file=args.max_records_per_file), indent=2))


if __name__ == '__main__':
    main()
