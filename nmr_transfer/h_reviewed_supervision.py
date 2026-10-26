"""功能概述：将已冻结的归属审核接入训练，保持源谱、缺失mask和抽样身份不变。

build_patches只把既有更正summary列出的train记录映射到canonical母节点；先核对
原annotations与源记录，保留原位移多重集、mask及其他字段。load_artifact核验独占
衍生物哈希和train/val边界。apply_review区分legacy/corrections/atom/domain四种
干预：两项既有更正、逐记录硬归属屏蔽、实验域谱损失归零；理论同ID保持隔离。
common_validation给所有配对臂相同的可靠谱级val，排除审核暂缓的val记录，不改
冻结检索val/test。reviewed_objective按原batch分母对暂缓记录的谱项归零；不重采样，
不填缺失标签，不开启条件弱目标。全保留批次直接调用原grouped_objective。
main核对原源文件、更正源文件和审核产物SHA，仅解析两条已更正train源记录，
输出小型artifact.pt及manifest/EXIT_CODE。由h_count_curriculum显式引用，原缓存、
候选、测试、旧annotations与已有模型文件只读；不拟合位移，不访问外部服务。
"""
import argparse
import collections
import copy
import json
import math
from pathlib import Path

import torch

from .data import identity
from .h_hydrogen_order import annotation
from .h_labels import make_key
from .h_replaytrain import grouped_objective
from .observation import observation_loss
from .prepare import sha

MODES = ('legacy', 'corrections', 'atom', 'domain')


def build_patches(corrections, originals, corrected, labels):
    patches = {}
    for item in corrections:
        rid = item['record_id']; old = originals[rid]; new = corrected[rid]
        smi, _, split = identity(old['smiles'])
        if (split != 'train' or item['split'] != 'train' or old['data_nature'] != 'experimental'
                or old['nucleus'] != '1H' or new['record_id'] != rid):
            raise ValueError('review_patch_requires_experimental_train_H')
        allowed = {'atom_targets', 'spectrum', 'assignment_correction'}
        if (any(old.get(k) != new.get(k) for k in set(old)|set(new) if k not in allowed)
                or [old['atom_targets'][i] for i in item['source_indices']] != item['before']
                or [new['atom_targets'][i] for i in item['source_indices']] != item['after']):
            raise ValueError('review_patch_changed_nonlabel_fields')
        values = lambda r: sorted(float(v) for v, m in zip(r['atom_targets'], r['atom_target_mask']) if m)
        if values(old) != values(new):raise ValueError('review_patch_changed_spectrum_multiset')
        expected_spectrum = [dict(atom_index=i, shift=v) for i, (v, m) in
                             enumerate(zip(new['atom_targets'], new['atom_target_mask'])) if m]
        if new['spectrum'] != expected_spectrum:raise ValueError('review_patch_spectrum_array_mismatch')
        key = ('experimental', old['source_ref'], rid, smi)
        _, before = annotation(old, 'coordinate_order'); _, after = annotation(new, 'coordinate_order')
        if key not in labels or labels[key]['groups'] != before['groups']:
            raise ValueError('review_patch_original_annotation_mismatch')
        changed = [i for i, (a, b) in enumerate(zip(old['atom_targets'], new['atom_targets'])) if a != b]
        if not changed or not set(changed).issubset(item['source_indices']):
            raise ValueError('review_patch_change_scope_mismatch')
        patches[key] = dict(before_groups=before['groups'], annotation=dict(labels[key], **after),
                            changed_source_indices=changed, evidence=item)
    return patches


def load_artifact(folder):
    folder = Path(folder); meta = json.loads((folder/'manifest.json').read_text())
    if (folder/'EXIT_CODE').read_text().strip() != '0' or sha(folder/'artifact.pt') != meta['artifact_sha256']:
        raise ValueError('review_artifact_incomplete_or_tampered')
    artifact = torch.load(folder/'artifact.pt', map_location='cpu', weights_only=False)
    for key, r in artifact['decisions'].items():
        if key[0] != 'experimental' or identity(key[3])[2] != r['split'] or r['split'] not in ('train', 'val'):
            raise ValueError('review_artifact_scope_mismatch')
        if r['hard_atom_supervision'] or r['weak_supervision_enabled']:
            raise ValueError('review_artifact_activates_unsupported_targets')
    for key in artifact['patches']:
        if key[0] != 'experimental' or identity(key[3])[2] != 'train':
            raise ValueError('review_patch_scope_mismatch')
    return artifact, meta


def apply_review(labels, rows, artifact, mode):
    if mode not in MODES:raise ValueError('invalid_review_mode')
    if any(r['domain'] != 'experimental' or r['split'] != 'train' for r in rows):
        raise ValueError('review_application_requires_experimental_train')
    keys = {make_key(r) for r in rows}; patched = []
    if mode != 'legacy':
        for key, patch in artifact['patches'].items():
            if key not in keys or labels[key]['groups'] != patch['before_groups']:
                raise ValueError('review_patch_not_matched_to_training_annotation')
            labels[key] = copy.deepcopy(patch['annotation']); patched.append(key[2])
    expected = {k for k, v in artifact['decisions'].items() if v['split'] == 'train'}
    if not expected.issubset(keys):raise ValueError('review_training_coverage_gap')
    result = []; masked = []; held = []
    for row in rows:
        key = make_key(row); decision = artifact['decisions'].get(key); keep = True
        if decision and mode in ('atom', 'domain'):
            old = labels.get(key, {})
            labels[key] = dict(old, groups={}, trusted_site_H=0, site_assignment_quality='independent_full_review')
            masked.append(key[2])
            if mode == 'domain' and not decision['experimental_spectral_supervision']:
                keep = False; held.append(key[2])
        result.append(dict(row, review_spectral_weight=float(keep)))
    return result, dict(mode=mode, patched_records=len(patched), patched_ids=patched,
                        reviewed_training_records=len(expected), masked_atom_records=len(masked),
                        held_spectral_records=len(held), held_ids=held,
                        source_q_or_missing_mask_modified=False, weak_targets_enabled=False)


def common_validation(rows, artifact):
    if any(r['domain'] != 'experimental' or r['split'] != 'val' for r in rows):
        raise ValueError('common_review_validation_requires_experimental_val')
    held = {k for k, v in artifact['decisions'].items()
            if v['split'] == 'val' and not v['experimental_spectral_supervision']}
    if not held.issubset({make_key(r) for r in rows}):raise ValueError('review_validation_coverage_gap')
    return [r for r in rows if make_key(r) not in held], dict(original=len(rows), held=len(held),
        kept=len(rows)-len(held), selection='same forward validation for every review arm; frozen retrieval queries unchanged')


def reviewed_objective(details, batch):
    parts = grouped_objective(details, batch)
    w = batch.review_spectral_weight.reshape(-1).to(details['quantiles'].dtype)
    if w.numel() != batch.num_graphs or not torch.all((w == 0) | (w == 1)):
        raise ValueError('invalid_review_spectral_weights')
    if bool((w == 1).all()):return parts
    if batch.group_mask[w[batch.batch] == 0].any():
        raise ValueError('held_spectrum_still_has_atom_supervision')
    pred = details['quantiles']; presence = (observation_loss(pred, batch)*w).mean()
    shape_mask = batch.shape_supervision.reshape(-1) > 0
    shape = (((pred-batch.supervised_q).abs().mean(1)*w)[shape_mask].mean()
             if shape_mask.any() else pred.sum()*0)
    residual = (batch.q[:, :, None]-pred[:, None, :])/.025
    per_nll = (-torch.logsumexp(-.5*residual.square(), dim=-1)+math.log(pred.shape[1])).mean(1)
    nll = (per_nll*w).mean()
    return dict(parts, loss=presence+.75*parts['site']+.5*shape+.005*nll,
                presence=presence, shape=shape, nll=nll)


def main():
    p = argparse.ArgumentParser()
    for name in ('review_root', 'labels', 'original_source', 'out'):
        p.add_argument('--'+name.replace('_', '-'), required=True)
    a = p.parse_args(); root = Path(a.review_root); out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    try:
        corrected_root = root/'reviewed_corrections'; resolution = root/'remaining_resolution'
        report = json.loads((corrected_root/'summary.json').read_text())
        final = json.loads((resolution/'completion_manifest.json').read_text())
        corrected_path = corrected_root/'atom_assigned.corrected.jsonl'
        paths = [Path(a.original_source), corrected_path, resolution/'review_resolutions.jsonl',
                 corrected_root/'summary.json', resolution/'completion_manifest.json', Path(a.labels)]
        hashes = {str(f): sha(f) for f in paths}
        if (hashes[str(Path(a.original_source))] != report['input_sha256'][a.original_source]
                or hashes[str(corrected_path)] != report['output_sha256']['atom_assigned.corrected.jsonl']
                or hashes[str(resolution/'review_resolutions.jsonl')] != final['derived_sha256']['review_resolutions.jsonl']):
            raise ValueError('review_source_hash_mismatch')
        wanted = {r['record_id'] for r in report['corrections']}
        def selected(path):
            found = {}
            with path.open('rb') as f:
                for line in f:
                    if not any(rid.encode() in line for rid in wanted):continue
                    r = json.loads(line)
                    if r['record_id'] in wanted:
                        if r['record_id'] in found:raise ValueError('duplicate_patch_source_record')
                        found[r['record_id']] = r
            if set(found) != wanted:raise ValueError('missing_patch_source_record')
            return found
        labels = torch.load(a.labels, map_location='cpu', weights_only=False)
        patches = build_patches(report['corrections'], selected(Path(a.original_source)), selected(corrected_path), labels)
        decisions = {}
        for r in map(json.loads, (resolution/'review_resolutions.jsonl').read_text().splitlines()):
            key = ('experimental', r['source'], r['record_id'], r['smiles'])
            if key in decisions:raise ValueError('duplicate_review_decision')
            decisions[key] = {k: r[k] for k in ('split', 'hard_atom_supervision', 'weak_supervision_enabled',
                                               'experimental_spectral_supervision', 'source_record_sha256')}
        artifact = dict(patches=patches, decisions=decisions)
        torch.save(artifact, out/'artifact.pt')
        for path, h in hashes.items():
            if sha(path) != h:raise ValueError('review_artifact_input_changed')
        meta = dict(patches=len(patches), decisions=len(decisions),
                    splits=dict(collections.Counter(v['split'] for v in decisions.values())),
                    input_sha256=hashes, artifact_sha256=sha(out/'artifact.pt'),
                    source_values_modified=False, test_targets_read=False, extra_API_calls=0)
        (out/'manifest.json').write_text(json.dumps(meta, indent=2))
        (out/'EXIT_CODE').write_text('0\n'); load_artifact(out); print(json.dumps(meta), flush=True)
    except Exception:
        (out/'EXIT_CODE').write_text('1\n'); raise


if __name__ == '__main__':main()
