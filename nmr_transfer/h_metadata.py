"""功能概述：按冻结记录身份恢复积分谱条件，同时保留旧条件用于因果消融。
metadata_view只在recovered_local策略投影recovered_*字段，旧bank保持原vector和
已知掩码；raw_conditions保留原字段证据，把明确缺失字符串保持为未知，不推测pH。
recover逐行扫描原14GB JSONL，完整计算SHA，只解析冻结ID对应的记录；经过原
normalize_record结构恢复及canonical核验后附加元数据，不改谱位置、积分、分区。
main输出新metadata与原cache的只读路径链接，manifest记录输入hash、匹配覆盖和
train溶剂/酰胺支持；原文件hash或身份冲突时失败，不发布完成侧表。HMDB及原子
谱的既有条件保持原样；无外部API，原始标签和候选库不作为条件估计目标。
balance_local_support在已有source/solvent质量内平方根平衡NMe2酰胺有无，
只读取候选二维结构与训练条件；局部采样改变作为独立消融，不扩充训练样本。
reencode_recovery只从已核验侧表保留的原始条件证据重新编码明确别名；
--reencode复用已完成的全文件SHA/ID核验，输出独立侧表，不重新扫描或覆盖旧结果。
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import re
import time

import torch
from rdkit import Chem
from .h_conditions import condition_from_old, encode_conditions
from .prepare import sha

RECOVERED_MODE = 'recovered_local'
MISSING = {'', 'nan', 'none', 'null', 'n/a', 'na', 'unknown', 'not reported',
           'not_known', 'not known', 'not_available', 'not available', 'not specified'}


def metadata_view(metadata, mode):
    if mode != RECOVERED_MODE:
        return {key: meta for key, meta in metadata.items() if not meta.get('recovery_only', False)}
    return {key: dict(meta, **{name[len('recovered_'):]: value
                             for name, value in meta.items() if name.startswith('recovered_')})
            for key, meta in metadata.items()
            if not meta.get('recovery_only', False) or
            (meta.get('recovery_split') == 'test' and '.' not in key[3])}


def raw_conditions(record):
    conditions = record.get('experimental_conditions') or record.get('conditions') or {}
    source = record.get('source_metadata') or {}
    candidates = [('solvent', record.get('solvent')),
                  ('experimental_conditions.solvent', conditions.get('solvent')),
                  ('source_metadata.NMR_solvent', source.get('NMR_solvent'))]
    field, value = next(((k, str(v).strip()) for k, v in candidates
                         if str(v or '').strip().lower() not in MISSING), ('missing', ''))
    clean_conditions = {k: v for k, v in conditions.items() if k != 'solvent'}
    return encode_conditions(dict(solvent=value, experimental_conditions=clean_conditions)), field


def balance_local_support(rows, metadata, weights):
    from .h_labels import make_key
    pattern = Chem.MolFromSmarts('[NX3]([CH3])([CH3])[CX3](=[OX1])')
    structures = {s: Chem.MolFromSmiles(s).HasSubstructMatch(pattern) for s in {r['smiles'] for r in rows}}
    groups = collections.defaultdict(list)
    for i, row in enumerate(rows):
        groups[row['source'], metadata[make_key(row)]['family']].append(i)
    result = weights.clone()
    for indices in groups.values():
        counts = collections.Counter(structures[rows[i]['smiles']] for i in indices)
        if len(counts) < 2:
            continue
        mass = weights[indices].sum()
        for i in indices:
            result[i] = weights[i] * counts[structures[rows[i]['smiles']]] ** -.5
        result[indices] *= mass / result[indices].sum()
    return result


def reencode_recovery(metadata):
    transitions = collections.Counter(); changed = 0
    for key, meta in metadata.items():
        if 'solvent_recovery' not in meta:
            continue
        evidence = meta['solvent_recovery']
        raw = dict(solvent=evidence['raw_solvent'], experimental_conditions=evidence['experimental_conditions'],
                   source_metadata=dict(NMR_solvent=evidence['NMR_solvent']))
        new, field = raw_conditions(raw)
        if meta['recovered_family'] != new['family']:
            changed += 1
        if meta.get('recovery_split') == 'train' and not meta.get('recovery_only'):
            transitions[key[1], meta['recovered_family'], new['family']] += 1
        for name, value in new.items():
            if name != 'old_cond':
                meta['recovered_' + name] = value
        meta['solvent_recovery'] = dict(evidence, resolved_source_field=field)
    return dict(changed_family_records=changed, actual_training_transitions=[
        dict(source=s, old=o, resolved=n, records=c) for (s, o, n), c in sorted(transitions.items())])


def recover(raw_path, frozen, metadata, expected_sha, status):
    from nmr_forward.source_io import normalize_record
    wanted = {}
    for row in frozen:
        rid = str(row['record_id'])
        if rid in wanted:
            raise ValueError('duplicate_frozen_id:' + rid)
        wanted[rid] = row
    index = collections.defaultdict(list)
    for key in metadata:
        if key[0] == 'experimental':
            index[key[1], key[2]].append(key)
    pattern = re.compile(rb'"record_id"\s*:\s*"([^"\\]+)"')
    wanted_bytes = {s.encode(): s for s in wanted}
    found = {}; digest = hashlib.sha256(); byte_count = 0
    with open(raw_path, 'rb') as stream:
        for line_number, line in enumerate(stream, 1):
            digest.update(line); byte_count += len(line)
            match = pattern.search(line)
            rid = wanted_bytes.get(match[1]) if match else None
            if rid is not None:
                raw = json.loads(line)
                if raw.get('record_id') != rid:
                    raise ValueError('nested_record_id_match:' + rid)
                normalized, reason = normalize_record(raw)
                if reason:
                    raise ValueError('previously_accepted_record_invalid:' + rid + ':' + reason)
                row = wanted[rid]
                mol = Chem.RemoveHs(Chem.MolFromSmiles(normalized['smiles']))
                smi = Chem.MolToSmiles(mol, isomericSmiles=True)
                group = hashlib.sha256(Chem.MolToSmiles(mol, isomericSmiles=False).encode()).hexdigest()
                partition = int(group[:8], 16) % 100
                split = 'train' if partition < 80 else 'val' if partition < 90 else 'test'
                if smi != row['smiles'] or split != row['split'] or raw['source'] != row['source']:
                    raise ValueError('frozen_identity_mismatch:' + rid)
                meta, field = raw_conditions(raw)
                # transfer.prepare把legacy canonical再交给identity一次；对特殊桥环立体
                # 表示必须复用这条确切路径，不以字符串差异覆盖冻结结构。
                transfer_smi = Chem.MolToSmiles(Chem.RemoveHs(Chem.MolFromSmiles(row['smiles'])), isomericSmiles=True)
                keys = index.get((row['source'], rid)) or [('experimental', row['source'], rid, transfer_smi)]
                if any(k[3] != transfer_smi for k in keys):
                    raise ValueError('prepared_identity_mismatch:' + rid)
                provenance = dict(line=line_number, source_field=field, raw_solvent=raw.get('solvent'),
                                  experimental_conditions=raw.get('experimental_conditions'),
                                  NMR_solvent=(raw.get('source_metadata') or {}).get('NMR_solvent'))
                if rid in found:
                    raise ValueError('duplicate_raw_id:' + rid)
                found[rid] = (keys, meta, provenance)
            if line_number % 250000 == 0:
                status('streaming', lines=line_number, bytes=byte_count, matched=len(found), wanted=len(wanted))
    actual_sha = digest.hexdigest()
    if actual_sha != expected_sha:
        raise ValueError('raw_input_sha256_mismatch:' + actual_sha)
    if len(found) != len(wanted):
        raise ValueError('frozen_ids_unresolved:' + str(len(wanted) - len(found)))
    for rid, (keys, meta, provenance) in found.items():
        for key in keys:
            old = metadata.get(key, condition_from_old(wanted[rid]['cond']))
            recovered = {('recovered_' + k): v for k, v in meta.items() if k != 'old_cond'}
            metadata[key] = dict(old, **recovered, solvent_recovery=provenance,
                                 recovery_only=key not in metadata,
                                 recovery_split=wanted[rid]['split'])
    counts = collections.Counter(); amides = []; amide = Chem.MolFromSmarts('[NX3]([CH3])([CH3])[CX3](=[OX1])')
    for row in frozen:
        if row['split'] != 'train':
            continue
        keys, meta, _ = found[str(row['record_id'])]
        old = metadata[keys[0]]
        counts[(row['source'], old['family'], meta['family'])] += 1
        if 'benzene' in meta['family'] and Chem.MolFromSmiles(row['smiles']).HasSubstructMatch(amide):
            amides.append(dict(source=row['source'], record_id=row['record_id'], smiles=row['smiles'],
                               solvent=meta['solvent'], xs=row['xs'], integrals=row['weights']))
    return dict(raw_sha256=actual_sha, matched=len(found), frozen=len(wanted), bytes=byte_count,
                train_family_transitions=[dict(source=s, old=o, recovered=n, records=c)
                                          for (s, o, n), c in sorted(counts.items())],
                train_benzene_NMe2_amides=amides,
                scope='exact frozen record IDs; complete source hash; no labels, rows or splits changed')


def main():
    p = argparse.ArgumentParser()
    for name in ['raw', 'frozen', 'audit', 'reencode', 'prepared', 'out']:
        p.add_argument('--' + name, required=name in ['prepared', 'out'])
    a = p.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=False); start = time.time()
    def status(stage, **kw):
        data = dict(stage=stage, seconds=time.time() - start, **kw)
        (out / 'status.json').write_text(json.dumps(data)); print(json.dumps(data), flush=True)
    try:
        torch.set_num_threads(4); status('loading_frozen')
        if a.reencode:
            parent = Path(a.reencode)
            if (parent / 'EXIT_CODE').read_text().strip() != '0':
                raise ValueError('recovery_incomplete')
            parent_manifest = json.loads((parent / 'manifest.json').read_text())
            if sha(parent / 'metadata.pt') != parent_manifest['metadata_sha256']:
                raise ValueError('recovered_metadata_sha256_mismatch')
            metadata = torch.load(parent / 'metadata.pt', map_location='cpu', weights_only=False)
            original_audit = json.loads((parent / 'metadata_audit.json').read_text())
            evidence = original_audit['solvent_recovery']
            if parent_manifest['raw_recovery_sha256'] != evidence['raw_sha256']:
                raise ValueError('recovery_provenance_sha256_mismatch')
            original_audit['solvent_alias_resolution'] = reencode_recovery(metadata)
        else:
            if not all([a.raw, a.frozen, a.audit]):
                raise ValueError('raw_frozen_audit_required')
            frozen = torch.load(a.frozen, map_location='cpu', weights_only=False, mmap=True)
            metadata = torch.load(Path(a.prepared) / 'metadata.pt', map_location='cpu', weights_only=False)
            audit = json.loads(Path(a.audit).read_text())
            evidence = recover(a.raw, frozen, metadata, audit['input_sha256'], status)
            original_audit = json.loads((Path(a.prepared) / 'metadata_audit.json').read_text())
        torch.save(metadata, out / 'metadata.pt')
        (out / 'cache.pt').symlink_to((Path(a.prepared) / 'cache.pt').resolve())
        manifest = json.loads((Path(a.prepared) / 'manifest.json').read_text())
        manifest.update(metadata_sha256=sha(out / 'metadata.pt'), raw_recovery_sha256=evidence['raw_sha256'],
                        metadata_policy=RECOVERED_MODE, legacy_metadata_sha256=sha(Path(a.prepared) / 'metadata.pt'),
                        recovery_source_sha256=sha(__file__), reencoded_parent=str(a.reencode) if a.reencode else None)
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
        (out / 'metadata_audit.json').write_text(json.dumps(dict(original_audit, solvent_recovery=evidence), indent=2))
        status('complete', matched=evidence['matched']); (out / 'EXIT_CODE').write_text('0\n')
    except Exception as exc:
        status('failed', error=repr(exc)); (out / 'EXIT_CODE').write_text('1\n'); raise


if __name__ == '__main__':
    main()
