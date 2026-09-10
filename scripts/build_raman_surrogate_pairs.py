#!/usr/bin/env python3
"""功能概述：构建用于校准 Raman 理论→实验代理模型的独立精确配对数据集。
主要职责：从索引取得同时含理论与实验 Raman 的分子组，流式读取 continuous_v1 曲线，
用保留立体与同位素的 canonical SMILES 再做严格身份交集；按分子组生成训练、验证和受保护
最终测试角色；保存全部合格理论/实验重复测量、覆盖掩码、逐分子关系表与完整来源审计。
代码库关系：消费 SpecGPT stage-2 连续谱及修正版 10k/200 manifest，不拟合代理参数，也不
读取检索得分。输出供物理约束有效域模拟器按同一 molecule_id 动态采样理论—实验组合。
输入输出与副作用：只读六个 combined Raman gzip、SQLite 索引和受保护测试 manifest；输出
目录必须不存在，经 staging 原子发布。默认路径相对于仓库根目录，可用 CLI
参数覆盖；当前上游只有 robust-minmax 归一化连续曲线，故产物明确标记
Raman activity、原始计数、激光、温度、仪器和预处理细节不可监督。
"""

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sqlite3
import tempfile

import numpy as np
from rdkit import Chem


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / 'data/external/specgpt/processed_stage_2_v1/continuous'
DEFAULT_INDEX = PROJECT_ROOT / 'data/external/specgpt/data_index/specgpt_dataset_index.sqlite'
DEFAULT_PROTECTED = (
    PROJECT_ROOT / 'outputs/raman_closed_10k_test200_seed0_exact_v2/sample_manifest.json'
)
GROUP_PATTERN = re.compile(rb'"group_hash"\s*:\s*"([0-9a-f]{32})"')
GRID_SIZE = 3000
AXIS = np.linspace(0.0, 4000.0, GRID_SIZE, dtype=np.float32)
EVAL_REGION = AXIS >= 500.0


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError('invalid SMILES')
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def deterministic_order(values, seed, purpose):
    return sorted(values, key=lambda value: hashlib.sha256(
        f'{seed}:{purpose}:{value}'.encode()).hexdigest())


def load_overlap(index_path):
    with sqlite3.connect(f'file:{index_path}?mode=ro', uri=True) as connection:
        rows = connection.execute(
            'SELECT lower(hex(group_hash)), exp_raman_count, theory_raman_count '
            'FROM molecule_groups WHERE exp_raman_count > 0 AND theory_raman_count > 0'
        ).fetchall()
    return {group: {'experimental': int(exp), 'theory': int(theory)}
            for group, exp, theory in rows}


def quality(spectrum, coverage):
    if spectrum.shape != (GRID_SIZE,) or not np.isfinite(spectrum).all():
        return 'invalid_spectrum'
    if (not isinstance(coverage, list) or len(coverage) != 2
            or not all(isinstance(value, int) for value in coverage)
            or not 0 <= coverage[0] < coverage[1] <= GRID_SIZE):
        return 'invalid_coverage'
    mask = EVAL_REGION & (np.arange(GRID_SIZE) >= coverage[0]) & (
        np.arange(GRID_SIZE) < coverage[1])
    if mask.sum() < math.ceil(EVAL_REGION.sum() * 0.5):
        return 'insufficient_coverage'
    if float(np.ptp(spectrum[mask])) <= 1e-10:
        return 'constant_spectrum'
    return None


def extract(root, domain, overlap):
    records, spectra, masks, audit = [], [], [], []
    for split in ('train', 'valid', 'test'):
        path = root / 'combined' / domain / 'raman' / f'{split}.jsonl.gz'
        digest, counts = hashlib.sha256(), Counter()
        with gzip.open(path, 'rb') as handle:
            for line_number, raw in enumerate(handle, 1):
                digest.update(raw)
                counts['scanned'] += 1
                match = GROUP_PATTERN.search(raw)
                if match is None:
                    raise ValueError(f'missing group_hash: {path}:{line_number}')
                group_hash = match.group(1).decode()
                if group_hash not in overlap:
                    continue
                counts['indexed_overlap'] += 1
                row = json.loads(raw)
                spectrum = np.asarray(row.get('spectrum'), dtype=np.float32)
                reason = quality(spectrum, row.get('coverage'))
                if reason:
                    counts[reason] += 1
                    continue
                try:
                    canonical_smiles = canonical(row['smiles'])
                except (KeyError, ValueError):
                    counts['invalid_smiles'] += 1
                    continue
                coverage = [int(value) for value in row['coverage']]
                record_id = hashlib.sha256(
                    f'{domain}:{split}:{line_number}:{path}'.encode()).hexdigest()[:24]
                metadata = {
                    'record_id': record_id,
                    'domain': domain,
                    'source_split': split,
                    'source_line': line_number,
                    'source_path': str(path),
                    'group_hash': group_hash,
                    'smiles': row['smiles'],
                    'canonical_smiles': canonical_smiles,
                    'source': row.get('source', ''),
                    'condition': row.get('condition', ''),
                    'intensity_scale': row.get('intensity_scale'),
                    'coverage': coverage,
                }
                records.append(metadata)
                spectra.append(spectrum)
                mask = np.zeros(GRID_SIZE, dtype=bool)
                mask[coverage[0]:coverage[1]] = True
                masks.append(mask)
                counts['quality_retained'] += 1
        entry = {
            'path': str(path),
            'bytes': path.stat().st_size,
            'decompressed_sha256': digest.hexdigest(),
            'counts': dict(counts),
        }
        audit.append(entry)
        print(json.dumps({'extracted': entry}), flush=True)
    return records, np.asarray(spectra, dtype=np.float32), np.asarray(masks, dtype=bool), audit


def write_jsonl_gzip(path, rows):
    with gzip.open(path, 'xt', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def manifest_semantics(source_manifests):
    configs = [document['config'] for document in source_manifests.values()]
    for config in configs:
        if config['grid_size'] != GRID_SIZE:
            raise ValueError('unexpected grid size')
        modality = config['modalities']['Raman']
        if modality['axis_min'] != 0 or modality['axis_max'] != 4000:
            raise ValueError('unexpected Raman axis')
    return {
        'dataset_kind': 'physics-constrained effective-domain surrogate calibration',
        'representation': 'continuous robust-minmax normalized Raman curve',
        'axis': {'unit': 'cm^-1', 'minimum': 0.0, 'maximum': 4000.0,
                 'points': GRID_SIZE, 'spacing': 'linear'},
        'available_supervision': [
            'normalized spectral shape', 'measured coverage mask', 'database source',
            'sample condition string when supplied by upstream',
        ],
        'unavailable_or_unverified': {
            'theory_vertical_axis_is_raman_activity': False,
            'normal_mode_frequencies_and_activities': False,
            'raman_tensors_or_depolarization_ratios': False,
            'laser_wavelength': False,
            'sample_temperature': False,
            'instrument_identity_and_response': False,
            'raw_photon_or_detector_counts': False,
            'exposure_gain_dark_current_read_noise': False,
            'theoretical_broadening_kernel': False,
            'complete_preprocessing_provenance': False,
        },
        'interpretation_limit': (
            'Do not interpret curve-level fitted parameters as uniquely recovered physical '
            'measurement conditions. Use them only to calibrate transferable effective priors.'
        ),
    }


def build(root, index_path, protected_manifest, output, seed, validation_fraction):
    root, index_path, protected_manifest, output = map(
        lambda path: Path(path).resolve(), (root, index_path, protected_manifest, output))
    if output.exists():
        raise FileExistsError(output)
    if not 0 < validation_fraction < 1:
        raise ValueError('validation fraction must be between zero and one')
    source_manifests = {
        domain: json.loads((root / 'manifests' /
                            f'processed_manifest_{domain}_raman.json').read_text())
        for domain in ('exp', 'theory')
    }
    semantics = manifest_semantics(source_manifests)
    protected_document = json.loads(protected_manifest.read_text())
    protected_groups = set(protected_document['test_groups'])
    if len(protected_groups) != 200:
        raise ValueError('protected manifest must contain 200 distinct test groups')
    overlap = load_overlap(index_path)
    extracted = {}
    for domain in ('theory', 'exp'):
        extracted[domain] = extract(root, domain, overlap)
    theory_records, theory_spectra, theory_masks, theory_audit = extracted['theory']
    exp_records, exp_spectra, exp_masks, exp_audit = extracted['exp']

    by_domain = {}
    for domain, records in [('theory', theory_records), ('exp', exp_records)]:
        grouped = defaultdict(list)
        for index, record in enumerate(records):
            grouped[(record['group_hash'], record['canonical_smiles'])].append(index)
        by_domain[domain] = grouped
    exact_keys = set(by_domain['theory']) & set(by_domain['exp'])
    if not exact_keys:
        raise ValueError('no exact theory-experiment identities found')

    # 同一忽略立体的分子组必须整体进入同一角色，避免近等价结构跨 train/validation 泄漏。
    exact_groups = {group for group, _ in exact_keys}
    ordinary_groups = exact_groups - protected_groups
    ordered_groups = deterministic_order(ordinary_groups, seed, 'surrogate-validation-groups')
    validation_count = max(1, math.ceil(len(ordered_groups) * validation_fraction))
    validation_groups = set(ordered_groups[:validation_count])

    selected_indices = {'theory': [], 'exp': []}
    molecules = []
    identity_ids = {}
    for group_hash, canonical_smiles in sorted(exact_keys):
        identity_hash = hashlib.sha256(canonical_smiles.encode()).hexdigest()
        molecule_id = 'raman_' + identity_hash[:20]
        if molecule_id in identity_ids and identity_ids[molecule_id] != canonical_smiles:
            raise ValueError('molecule id collision')
        identity_ids[molecule_id] = canonical_smiles
        role = ('protected_final_test' if group_hash in protected_groups else
                'surrogate_validation' if group_hash in validation_groups else
                'surrogate_train')
        local = {}
        for domain in ('theory', 'exp'):
            old_indices = by_domain[domain][(group_hash, canonical_smiles)]
            start = len(selected_indices[domain])
            selected_indices[domain].extend(old_indices)
            local[domain] = list(range(start, start + len(old_indices)))
        molecules.append({
            'molecule_id': molecule_id,
            'identity_sha256': identity_hash,
            'group_hash': group_hash,
            'canonical_smiles': canonical_smiles,
            'role': role,
            'allowed_for_parameter_fit': role == 'surrogate_train',
            'allowed_for_model_selection': role == 'surrogate_validation',
            'theory_indices': local['theory'],
            'experimental_indices': local['exp'],
            'theory_count': len(local['theory']),
            'experimental_count': len(local['exp']),
            'available_pair_combinations': len(local['theory']) * len(local['exp']),
        })

    final_records, final_spectra, final_masks = {}, {}, {}
    molecule_lookup = {(row['group_hash'], row['canonical_smiles']): row for row in molecules}
    for domain, records, spectra, masks in [
        ('theory', theory_records, theory_spectra, theory_masks),
        ('exp', exp_records, exp_spectra, exp_masks),
    ]:
        chosen = selected_indices[domain]
        final_spectra[domain] = spectra[chosen]
        final_masks[domain] = masks[chosen]
        final_records[domain] = []
        for new_index, old_index in enumerate(chosen):
            record = dict(records[old_index])
            molecule = molecule_lookup[(record['group_hash'], record['canonical_smiles'])]
            record.update(spectrum_index=new_index, molecule_id=molecule['molecule_id'],
                          role=molecule['role'])
            final_records[domain].append(record)

    roles = Counter(row['role'] for row in molecules)
    molecule_groups = defaultdict(list)
    for row in molecules:
        molecule_groups[row['group_hash']].append(row['molecule_id'])
    if any(len({molecule_lookup[(row['group_hash'], row['canonical_smiles'])]['role']
                    for row in molecules if row['group_hash'] == group}) != 1
           for group in molecule_groups):
        raise AssertionError('group role leakage')
    if any(row['theory_count'] == 0 or row['experimental_count'] == 0 for row in molecules):
        raise AssertionError('unpaired molecule emitted')

    overlap_without_exact = sorted(set(overlap) - exact_groups)
    staging = Path(tempfile.mkdtemp(prefix=output.name + '.staging-', dir=output.parent))
    try:
        np.save(staging / 'wavenumber_cm-1.npy', AXIS)
        for domain in ('theory', 'exp'):
            np.save(staging / f'{domain}_spectra.npy', final_spectra[domain])
            np.save(staging / f'{domain}_coverage_masks.npy', final_masks[domain])
            write_jsonl_gzip(staging / f'{domain}_records.jsonl.gz', final_records[domain])
        write_jsonl_gzip(staging / 'molecules.jsonl.gz', molecules)
        split_document = {
            'schema': 'raman_surrogate_molecule_split_v1',
            'seed': seed,
            'validation_fraction_requested': validation_fraction,
            'policy': 'whole group_hash isolation; deterministic SHA-256 order',
            'roles': {role: sorted(row['molecule_id'] for row in molecules if row['role'] == role)
                      for role in ('surrogate_train', 'surrogate_validation', 'protected_final_test')},
            'groups': {role: sorted({row['group_hash'] for row in molecules if row['role'] == role})
                       for role in ('surrogate_train', 'surrogate_validation', 'protected_final_test')},
        }
        (staging / 'split.json').write_text(json.dumps(split_document, indent=2) + '\n')
        exclusions = {
            'indexed_overlap_groups_without_exact_canonical_identity': overlap_without_exact,
            'count': len(overlap_without_exact),
            'reason': ('the index identity ignores stereochemistry/isotopes; these groups have no '
                       'complete canonical SMILES shared by both domains after quality checks'),
        }
        (staging / 'exclusions.json').write_text(json.dumps(exclusions, indent=2) + '\n')
        output_files = sorted(path.name for path in staging.iterdir())
        hashes = {name: file_hash(staging / name) for name in output_files}
        manifest = {
            'schema': 'raman_surrogate_exact_pairs_v1',
            'seed': seed,
            'source_root': str(root),
            'source_index': str(index_path),
            'source_index_sha256': file_hash(index_path),
            'source_manifests': source_manifests,
            'source_audit': theory_audit + exp_audit,
            'protected_manifest': str(protected_manifest),
            'protected_manifest_sha256': file_hash(protected_manifest),
            'identity': 'RDKit canonical isomeric SMILES with isotopes retained',
            'pairing': ('exact complete canonical identity within indexed connectivity group; all '
                        'quality-passing repeats retained; combinations sampled dynamically'),
            'quality': ('finite 3000-point curve; valid coverage; >=50% coverage of 500..4000 cm^-1; '
                        'non-constant on covered evaluation region'),
            'counts': {
                'indexed_overlap_groups': len(overlap),
                'exact_identity_groups': len(exact_groups),
                'exact_molecules': len(molecules),
                'theory_spectra': len(final_records['theory']),
                'experimental_spectra': len(final_records['exp']),
                'available_pair_combinations': sum(row['available_pair_combinations'] for row in molecules),
                'roles': dict(roles),
                'overlap_groups_without_exact_identity': len(overlap_without_exact),
            },
            'semantics': semantics,
            'array_alignment': ('records spectrum_index indexes the corresponding domain spectra and '
                                'coverage mask arrays; molecule index lists use the same indices'),
            'leakage_policy': ('protected_final_test records are distributed for audit/evaluation only; '
                               'allowed_for_parameter_fit and model_selection are false'),
            'script_sha256': file_hash(Path(__file__)),
            'files_sha256': hashes,
        }
        (staging / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        # 发布前从磁盘回读关键不变量和哈希，防止索引与数组错位。
        for name, expected in hashes.items():
            if file_hash(staging / name) != expected:
                raise ValueError(f'output hash changed: {name}')
        if np.load(staging / 'theory_spectra.npy', mmap_mode='r').shape != (len(final_records['theory']), GRID_SIZE):
            raise ValueError('theory output shape mismatch')
        if np.load(staging / 'exp_spectra.npy', mmap_mode='r').shape != (len(final_records['exp']), GRID_SIZE):
            raise ValueError('experimental output shape mismatch')
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({'completed': str(output), 'counts': manifest['counts'],
                      'manifest_sha256': file_hash(output / 'manifest.json')}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--index', type=Path, default=DEFAULT_INDEX)
    parser.add_argument('--protected-manifest', type=Path, default=DEFAULT_PROTECTED)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--validation-fraction', type=float, default=0.2)
    args = parser.parse_args()
    build(args.root, args.index, args.protected_manifest, args.output,
          args.seed, args.validation_fraction)


if __name__ == '__main__':
    main()
