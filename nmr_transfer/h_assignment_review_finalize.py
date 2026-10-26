"""功能概述：核验完整原记录的双会话逐例审核，并生成可追溯的训练分流侧表。

resolve连接dossiers、两路review及独立甲基可识别性结果；逐条检查source哈希、固定
train/val划分、完整审核声明和独立session。分歧必须由显式adjudications给出理由，
不以多数票、置信分数或理论误差自动换标签。任何硬标签修复提案需走独立补丁流程。
overlay_policy流式更新既有train/val策略中的指定记录；其他行按bytes复制。未知来源
记录暂缓实验域谱级监督，局部归属问题保留谱级监督；全部硬原子监督继续屏蔽。
条件弱目标仅作为诊断候选，不因DP唯一或审核action就自动启动弱监督。
main写新目录的逐例裁决、策略与summary/EXIT_CODE；原JSONL、既有cache、候选、test
及模型权重均只读，不创建训练任务。输入哈希前后核验，依赖h_assignment_audit.sha。
"""
import argparse
import collections
import copy
import hashlib
import json
from pathlib import Path

from .h_assignment_audit import sha
from .data import identity
from .h_assignment_identifiability import identify


ACTIONS = ('mask_atom_keep_spectrum', 'hold_experimental_domain_pending_provenance',
           'conditional_weak_supervision_candidate')


def content_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def resolve(cases, bundles, identifiability, adjudications):
    """核验完整双审覆盖；返回训练处置，保留科学原因的不确定性及原标签。"""
    by_id = {r['record_id']: r for r in cases}
    if len(by_id) != len(cases) or not cases:
        raise ValueError('invalid_case_coverage')
    if any(c['split'] not in ('train', 'val') for c in cases):
        raise ValueError('test_or_unknown_partition_in_review')
    for c in cases:
        raw = c['source_record']
        if content_sha(raw) != c['source_record_sha256']:
            raise ValueError('source_record_content_hash_mismatch')
        if (raw['record_id'] != c['record_id'] or raw['source_ref'] != c['source']
                or raw['nucleus'] != '1H' or raw['data_nature'] != 'experimental'
                or identity(raw['smiles'])[2] != c['split']):
            raise ValueError('source_record_scope_mismatch')
    assigned = collections.defaultdict(list)
    for bundle in bundles:
        session = bundle['session']
        rows = bundle['reviews']
        if len({r['record_id'] for r in rows}) != len(rows):
            raise ValueError('duplicate_review_record')
        for review in rows:
            key = review['record_id']
            if key not in by_id:
                raise ValueError('review_outside_case_cohort')
            c = by_id[key]
            if (review['split'] != c['split'] or review['full_source_record_reviewed'] is not True
                    or review['source_record_sha256'] != c['source_record_sha256']):
                raise ValueError('source_or_full_review_contract_mismatch')
            if review['action'] not in ACTIONS or not review['reason_zh'].strip():
                raise ValueError('invalid_review_action_or_reason')
            if review['repair_uniquely_supported'] is not False:
                raise ValueError('hard_repair_requires_separate_evidence_pipeline')
            assigned[key].append(dict(session=session, review=review))
    if set(identifiability) != set(by_id):
        raise ValueError('identifiability_coverage_mismatch')
    if set(adjudications) - set(by_id):
        raise ValueError('adjudication_outside_case_cohort')
    results = []
    for key, c in by_id.items():
        pair = assigned[key]
        if len(pair) != 2 or len({r['session'] for r in pair}) != 2:
            raise ValueError('two_independent_full_reviews_required')
        if content_sha(pair[0]['review']) == content_sha(pair[1]['review']):
            raise ValueError('identical_review_payload_cannot_establish_independence')
        conditional = identifiability[key]
        if (conditional['source'] != c['source'] or conditional['split'] != c['split']
                or conditional['record_id'] != key):
            raise ValueError('conditional_result_scope_mismatch')
        actions = {r['review']['action'] for r in pair}
        correction = adjudications.get(key)
        if len(actions) != 1 and correction is None:
            raise ValueError('review_disagreement_requires_adjudication')
        if correction is not None:
            if correction.get('action') not in ACTIONS or not correction.get('reason_zh','').strip():
                raise ValueError('invalid_adjudication')
            action = correction['action']
        else:
            action = next(iter(actions))
        results.append(dict(record_id=key, source=c['source'], split=c['split'],
            source_database_id=c.get('source_database_id'), smiles=c['canonical_smiles'],
            source_record_sha256=c['source_record_sha256'], review_complete=True,
            independent_full_reviews=2, reviewer_sessions=[r['session'] for r in pair],
            reviewer_actions=[r['review']['action'] for r in pair],
            reviewer_action_agreement=len(actions)==1, adjudication=correction,
            classifications=[r['review']['classification'] for r in pair],
            reviewer_reasons_zh=[r['review']['reason_zh'] for r in pair],
            reviewer_proposals_not_applied=[r['review'].get('repair_proposal') for r in pair],
            disposition=action, hard_atom_supervision=False,
            experimental_spectral_supervision=action!='hold_experimental_domain_pending_provenance',
            weak_supervision_enabled=False, conditional_weak_targets=identifiability[key],
            source_label_values_modified=False, missing_mask_modified=False,
            causal_interpretation='source/observation hypotheses remain separate from training disposition',
            domain_shift_confirmed=False))
    return results


def overlay_policy(prior, output, resolutions):
    """只修改指定训练/验证策略行，原谱及其他策略行保持原值/原bytes。"""
    indexed = {r['record_id']: r for r in resolutions}
    seen = set(); preserved = 0; counts = collections.Counter()
    with open(prior, 'rb') as f, open(output, 'xb') as dest:
        for line in f:
            row = json.loads(line); key = row['record_id']
            if key not in indexed:
                dest.write(line); preserved += 1; continue
            if key in seen:
                raise ValueError('duplicate_policy_case')
            seen.add(key); decision = indexed[key]
            if row['split'] != decision['split'] or row['source'] != decision['source']:
                raise ValueError('policy_source_or_split_mismatch')
            if row['keep_atom_parent_supervision']:
                raise ValueError('review_overlay_requires_preexisting_flag')
            updated = copy.deepcopy(row)
            updated.update(keep_spectral_supervision=decision['experimental_spectral_supervision'],
                keep_atom_parent_supervision=False, weak_supervision_enabled=False,
                review_disposition=decision['disposition'], review_complete=True,
                source_record_sha256=decision['source_record_sha256'],
                source_values_and_missing_mask_unchanged=True)
            dest.write((json.dumps(updated,ensure_ascii=False)+'\n').encode())
            counts[decision['split']+'|'+decision['disposition']] += 1
    if seen != set(indexed):
        raise ValueError('policy_missing_reviewed_case')
    return dict(reviewed_records=len(seen), unaffected_policy_lines_copied_verbatim=preserved,
                dispositions=dict(counts))


def main():
    p=argparse.ArgumentParser()
    for field in ('dossiers','review_manifest','identifiability','prior_policy','out'):
        p.add_argument('--'+field.replace('_','-'),required=True)
    p.add_argument('--adjudications')
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    try:
        paths=[a.dossiers,a.review_manifest,a.identifiability,a.prior_policy]
        if a.adjudications:paths.append(a.adjudications)
        manifest=json.loads(Path(a.review_manifest).read_text());bundles=[];review_hashes=set()
        for path, field in ((a.dossiers,'dossiers_sha256'),(a.identifiability,'identifiability_sha256'),
                            (a.prior_policy,'prior_policy_sha256')):
            if sha(path)!=manifest[field]:raise ValueError('manifest_input_hash_mismatch')
        cases=list(map(json.loads,Path(a.dossiers).read_text().splitlines()))
        indexed={c['record_id']:c for c in cases}
        for item in manifest['reviews']:
            path=Path(a.review_manifest).parent/item['file'];paths.append(str(path))
            if sha(path)!=item['sha256']:raise ValueError('review_file_hash_mismatch')
            if item['sha256'] in review_hashes:raise ValueError('reused_review_file')
            review_hashes.add(item['sha256'])
            data=json.loads(path.read_text())
            batch=Path(a.review_manifest).parent/item['input_batch'];paths.append(str(batch))
            if sha(batch)!=item['input_batch_sha256'] or data['input_sha256']!=item['input_batch_sha256']:
                raise ValueError('review_input_batch_hash_mismatch')
            subset=list(map(json.loads,batch.read_text().splitlines()))
            if (len({c['record_id'] for c in subset}) != len(subset)
                    or {r['record_id'] for r in data['reviews']} != {c['record_id'] for c in subset}
                    or any(c != indexed.get(c['record_id']) for c in subset)):
                raise ValueError('review_batch_content_or_coverage_mismatch')
            bundles.append(dict(session=item['session'],reviews=data['reviews']))
        hashes={str(path):sha(path) for path in paths}
        ident_rows=list(map(json.loads,Path(a.identifiability).read_text().splitlines()))
        ident={r['record_id']:r for r in ident_rows}
        if len(ident) != len(ident_rows):raise ValueError('duplicate_conditional_result')
        for c in cases:
            expected=json.loads(json.dumps(identify(c['source_record'])))
            if expected!=ident.get(c['record_id']):raise ValueError('stale_or_invalid_conditional_result')
        adjudications=json.loads(Path(a.adjudications).read_text()) if a.adjudications else {}
        rows=resolve(cases,bundles,ident,adjudications)
        with (out/'review_resolutions.jsonl').open('x') as f:
            for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
        policy=overlay_policy(a.prior_policy,out/'training_policy.jsonl',rows)
        for path,h in hashes.items():
            if sha(path)!=h:raise ValueError('review_input_modified_during_finalization')
        report=dict(status='full_review_and_policy_complete',records=len(rows),
            train=sum(r['split']=='train' for r in rows),val=sum(r['split']=='val' for r in rows),
            full_independent_reviews=2*len(rows),action_agreements=sum(r['reviewer_action_agreement'] for r in rows),
            explicit_adjudications=sum(r['adjudication'] is not None for r in rows),
            disposition_counts=dict(collections.Counter(r['disposition'] for r in rows)),
            conditional_solver_status=dict(collections.Counter(r['conditional_weak_targets']['status'] for r in rows)),
            policy=policy,new_hard_label_repairs=0,weak_supervision_activated=False,
            original_values_or_masks_modified=False,test_labels_used=False,
            existing_caches_modified=False,training_started=False,extra_API_calls=0,
            input_sha256=hashes,output_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()})
        (out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        (out/'EXIT_CODE').write_text('0\n');print(json.dumps(report,ensure_ascii=False),flush=True)
    except Exception:
        (out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':main()
