"""功能概述：用明确准入的 JSON 元数据判定四格历史 baseline 能否整体复用。

本工具只服务本轮 fixed_single/fixed_dual × seed193/211 的预注册矩阵，不选模型。
qualify 比较冻结命名映射、实际执行协议/收据、1371 身份/顺序/mask、候选库、
两角色 checkpoint/alpha/O/36 中心/full33 scale/source 以及历史预测的已登记 SHA。
所有引用的大张量和预测只与独立 inventory 的始末 SHA 声明比较；本工具不会打开、
哈希或解析这些文件，也不导入 torch/RDKit/研究模块。角色状态的证据不足时不补造。

四格全部满足才 qualified=true、registered_baseline_jobs_count=0；任何一格缺少
完整证据或条件不符，四格全部不复用、登记 4 个 baseline（总共 12 个新 eval）。
这两种有效政策结果都为 status=PASS；未锁定 validation、未获 root 明确 metadata
准入、损坏/漂移的输入字节、非法路径等为真实 EXIT1，不静默伪装资格不满足。

main 必须先核独审实现 SHA、validation lock 与 root admission，才能读取旧元数据。
request 使用独立生成的规范化 expected_cells；旧证据可以显式为空，得到有原因的
不复用结果。真实 inventory/qualification 只能在完整 val 锁定和 root 显式准入后运行。
输出唯一目录 summary.json/EXIT_CODE（失败保留 failure.json），全部已读元数据和
本源码均记录实际始末 SHA。科学预测和张量的字节验证职责保留给受审 inventory。
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

SCHEMA = 'cap10_baseline_reuse_qualification_20261024'
REQUEST_SCHEMA = 'cap10_baseline_reuse_request_20261024'
CELLS = {(config, seed) for config in ('fixed_single', 'fixed_dual') for seed in (193, 211)}
LEGACY = {'fixed_single': 'fixedcenter_mixed_listwise_single',
          'fixed_dual': 'fixedcenter_mixed_listwise_dual'}
VARIANTS = {'observed': 1000, 'frequency_crop': 200, 'peak_dropout': 171}
IDENTITIES = {'query_sha256', 'identity_content_sha256', 'variant_mask_content_sha256'}
ROLE_DIGESTS = {'observation_state_sha256', 'fixed_centers_tensor_sha256', 'full33_scale_sha256'}
SOURCE_NAMES = {'h_pool_ranker.py', 'h_research_ranker.py', 'h_research_contract.py',
                'h_research_observation.py', 'h_global_ranker.py', 'h_global_features.py',
                'data.py', 'h_pool_fixed_center_ranker.py', 'evaluate_pool_fixed_center_sealed.py'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def content_sha(value):
    return digest(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def canonical_ref(value):
    require(isinstance(value, dict) and set(value) >= {'path', 'sha256'}, 'artifact_ref_required')
    path = Path(value['path'])
    require(path.is_absolute() and '..' not in path.parts and str(path) == value['path'], 'canonical_reference_path')
    require(valid_sha(value['sha256']), 'artifact_reference_SHA')
    return value['path'], value['sha256']


def safe_metadata(path):
    raw = Path(path).absolute()
    require('..' not in raw.parts, 'metadata_parent_traversal')
    resolved = raw.resolve(strict=True)
    for p in (raw, resolved):
        name = p.name.lower()
        require(p.suffix.lower() == '.json' and name not in
                ('summary.json', 'test.json', 'predictions.json', 'artifact_map.json')
                and 'report' not in name and 'prediction' not in name and 'metrics' not in name,
                'forbidden_non_metadata_input')
    require(not raw.is_symlink(), 'metadata_symlink_not_allowed')
    return resolved


class MetadataError(ValueError):
    """声明的元数据本身无法安全消费，禁止静默改成复用资格不足。"""


class Metadata:
    def __init__(self):
        self.before = {}
        self.local = {}
        self.source_before = {}

    def pin_source(self):
        own = Path(__file__).resolve(strict=True)
        require(not Path(__file__).is_symlink(), 'qualifier_source_symlink')
        current = digest(own.read_bytes())
        require(str(own) not in self.source_before or self.source_before[str(own)] == current,
                'qualifier_source_changed_before_metadata')
        self.source_before[str(own)] = current
        return str(own), current

    def load(self, ref):
        try:
            remote, wanted = canonical_ref(ref)
            local = safe_metadata(ref.get('local_path', remote))
            require(remote not in self.local or self.local[remote] == str(local), 'metadata_alias_conflict')
            raw = local.read_bytes()
            require(digest(raw) == wanted, 'metadata_actual_SHA:' + remote)
            require(remote not in self.before or self.before[remote] == wanted, 'metadata_read_changed:' + remote)
            self.before[remote], self.local[remote] = wanted, str(local)
            value = json.loads(raw)
            require(isinstance(value, dict), 'metadata_object_required:' + remote)
            return value
        except ValueError as exc:
            raise MetadataError(str(exc)) from exc

    def finish(self):
        # 对所有路径先准入再读取，防始末检查跟随后来替换的 symlink。
        for p in self.local.values():
            require(str(safe_metadata(p)) == p, 'metadata_canonical_redirect')
        after = {remote: digest(Path(p).read_bytes()) for remote, p in self.local.items()}
        require(after == self.before, 'metadata_changed_at_finish')
        require(all(not Path(p).is_symlink() and digest(Path(p).read_bytes())==h
                    for p,h in self.source_before.items()), 'qualifier_source_changed_at_finish')
        return after


def ref_equal(left, right):
    return canonical_ref(left) == canonical_ref(right)


def strict_zero(value):
    return type(value) is int and value == 0


def utc(value):
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    require(parsed.utcoffset() is not None and parsed.utcoffset().total_seconds() == 0, 'UTC_required')
    return parsed


def validate_expected(cells):
    require(isinstance(cells, list) and len(cells) == 4, 'four_expected_baseline_cells')
    indexed = {}
    for cell in cells:
        key = (cell['config'], cell['seed'])
        require(type(cell['seed']) is int and key in CELLS and key not in indexed, 'unique_registered_baseline_cell')
        require(cell['recall'] == cell['config'].removeprefix('fixed_'), 'baseline_recall_identity')
        canonical_ref(cell['cache'])
        require(set(cell['checkpoint_roles']) == {'no_aug', 'aug'}, 'both_checkpoint_roles_required')
        for role in cell['checkpoint_roles'].values():
            for name in ('checkpoint', 'centers', 'center_rows'):
                canonical_ref(role[name])
        require(set(cell['fixed_test_identity']) == IDENTITIES and
                all(valid_sha(v) for v in cell['fixed_test_identity'].values()), 'three_full1371_identity_SHAs')
        require(valid_sha(cell['query_order_sha256']) and valid_sha(cell['candidate_order_sha256']), 'explicit_order_SHAs')
        require(cell['conditions'] == dict(cohort='test', query_count=1371, base_queries=1000,
                variant_counts=VARIANTS, candidate_count=114982, natural_pool_size=512,
                appended_training_positives=0), 'registered_natural1371_conditions')
        require(isinstance(cell['source_sha256'], dict) and cell['source_sha256'] and
                all(Path(p).is_absolute() and '..' not in Path(p).parts and valid_sha(h)
                    for p, h in cell['source_sha256'].items()), 'expected_scientific_source_refs')
        indexed[key] = cell
    require(set(indexed) == CELLS, 'exact_four_baseline_matrix')
    # 同 seed 的 single/dual 必须使用同两份角色权重和中心，只有自然候选池不同。
    for seed in (193, 211):
        single, dual = indexed['fixed_single', seed], indexed['fixed_dual', seed]
        require(single['checkpoint_roles'] == dual['checkpoint_roles'] and
                single['fixed_test_identity'] == dual['fixed_test_identity'] and
                single['query_order_sha256'] == dual['query_order_sha256'] and
                single['candidate_order_sha256'] == dual['candidate_order_sha256'],
                'same_seed_baseline_roles_and_identity')
    return indexed


def parse_legacy_argv(argv):
    require(isinstance(argv, list) and len(argv) >= 2 and len(argv) % 2 == 0, 'legacy_exact_argv_layout')
    require(Path(argv[1]).name == 'evaluate_pool_fixed_center_sealed.py', 'legacy_fixed_evaluator_required')
    flags = argv[2:]
    require(all(isinstance(flags[i], str) and flags[i].startswith('--') for i in range(0, len(flags), 2)),
            'legacy_exact_argv_flags')
    result = {flags[i][2:].replace('-', '_'): flags[i+1] for i in range(0, len(flags), 2)}
    require(len(result) == len(flags)//2 and set(result) ==
            {'source','config','cache','no_aug_checkpoint','aug_checkpoint','seed','out','frozen_selection'},
            'legacy_no_extra_or_duplicate_flags')
    result['seed'] = int(result['seed'])
    result['device'] = 'cuda'  # 原 adapter 的固定 CLI default；GPU 物理路由不改变计算条件。
    return result


def qualify(request, metadata):
    """只读规范化四格与已签元数据；不以旧指标或预测内容作任何决策。"""
    own, own_sha = metadata.pin_source()
    require(request['schema'] == REQUEST_SCHEMA, 'reuse_request_schema')
    expected = validate_expected(request['expected_cells'])
    expected_sha = content_sha(request['expected_cells'])
    protocol = metadata.load(request['protocol_ref'])
    require(protocol['schema'] == 'cap10_controlled_protocol_20261024' and
            protocol['test_preregistration']['baseline_reuse_mode'] == 'all_four_or_none' and
            protocol['test_preregistration']['matrix_cells'] == 12, 'protocol_all_four_or_none')
    lock = metadata.load(request['validation_lock_ref'])
    require(lock['schema'] == 'cap10_validation_lock_20261024' and lock['status'] == 'LOCKED' and
            lock['selection_scope'] == 'train_validation_only' and lock['test_reselection_allowed'] is False and
            lock['promotion'] is False and lock['protocol_sha256'] == request['protocol_ref']['sha256'] and
            lock['preselected'] in ('fixedcenter_mixed_listwise_single', 'cap10_dual'), 'validation_must_be_locked')
    admission = metadata.load(request['admission_ref'])
    require(admission['schema'] == 'cap10_baseline_metadata_admission_20261024' and
            admission['status'] == 'ADMITTED' and admission['explicit_root_authorization'] is True and
            admission['phase'] == 'baseline_reuse_metadata_only' and admission['expected_cells_sha256'] == expected_sha and
            admission['protocol_sha256'] == request['protocol_ref']['sha256'] and
            admission['validation_lock_sha256'] == request['validation_lock_ref']['sha256'],
            'explicit_root_metadata_admission_required')
    implementation = metadata.load(request['implementation_review_ref'])
    require(implementation['status'] == 'PASS' and implementation['independent'] is True and
            implementation['scope'] == 'baseline_reuse_metadata_implementation' and
            implementation['reviewed_sha256'].get(own) == own_sha, 'implementation_review_required')
    # admission 明确绑定所有将被打开的元数据，不含其自身，避免循环 SHA。
    refs = [request[k] for k in ('protocol_ref','validation_lock_ref','implementation_review_ref')]
    historical = request.get('historical', {})
    require(isinstance(historical, dict), 'historical_metadata_object')
    refs += [historical[k] for k in ('frozen_ref','queue_ref','execution_review_ref','inventory_ref',
                                    'state_proof_ref','state_review_ref') if historical.get(k) is not None]
    for cell in historical.get('cells', []):
        refs += [cell[k] for k in ('protocol_ref','receipt_ref') if cell.get(k) is not None]
    for ref in refs:
        p,h = canonical_ref(ref)
        require(admission['metadata_sha256'].get(p) == h, 'metadata_not_admitted:' + p)
    failures = []
    loaded = {}
    for name in ('frozen','queue','execution_review','inventory','state_proof','state_review'):
        ref = historical.get(name+'_ref')
        if ref is None:
            failures.append(dict(cell='all', reason='missing_'+name+'_metadata'))
        else:
            loaded[name] = metadata.load(ref)
    outputs = []
    historical_cells = historical.get('cells', [])
    require(isinstance(historical_cells, list), 'historical_cells_list')
    index = {}
    for cell in historical_cells:
        key = (cell['config'],cell['seed'])
        require(key in CELLS and key not in index and type(cell['seed']) is int, 'historical_exact_cell_identity')
        index[key] = cell
    # 缺证据是明确的不复用，不为凑资格补读 checkpoint 或预测。
    if not failures:
        frozen, queue, review, inventory, states, state_review = (loaded[k] for k in
            ('frozen','queue','execution_review','inventory','state_proof','state_review'))
        require(review['status'] == 'PASS' and review['independent'] is True, 'prior_execution_independent_review')
        require(state_review['status'] == 'PASS' and state_review['independent'] is True and
                state_review['scope'] == 'fixed_baseline_state_and_identity_metadata' and
                state_review['reviewed_sha256'].get(historical['state_proof_ref']['path']) == historical['state_proof_ref']['sha256'],
                'independent_state_proof_binding')
        require(inventory['schema'] == 'cap10_baseline_reuse_hash_inventory_20261024' and
                inventory['status'] == 'PASS' and strict_zero(inventory['actual_exit']) and
                inventory['method'] == 'actual_sha256_before_and_after' and
                inventory['before_sha256'] == inventory['after_sha256'] and
                inventory['test_predictions_parsed'] is False and inventory['model_forward_calls'] == 0,
                'actual_stable_hash_inventory_required')
        require(states['schema'] == 'fixed_baseline_role_state_proof_20261024', 'role_state_proof_schema')
        require(frozen['schema'] == 'nmr_pool_ranker_20261023' and
                queue['schema'] == 'nmr_research_jobs_20261022', 'legacy_frozen_queue_schema')
        for ref in (historical['frozen_ref'],historical['queue_ref']):
            require(review['reviewed_sha256'].get(ref['path']) == ref['sha256'], 'prior_review_frozen_queue_binding')
    global_missing = bool(failures)
    for key in sorted(CELLS):
        cell = expected[key]
        reasons = []
        old = index.get(key)
        if old is None:
            reasons.append('missing_prior_cell')
        elif not global_missing:
            try:
                check_cell(cell,old,historical,loaded,metadata)
            except Ineligible as exc:
                reasons.append(str(exc))
        else:
            reasons.append('incomplete_global_evidence')
        outputs.append(dict(config=key[0],seed=key[1],recall=cell['recall'],
            cell_evidence_complete=not reasons,qualified=False,reason_codes=reasons,
            predictions=old.get('predictions') if old else None,
            checkpoint_role_refs=cell['checkpoint_roles'],cache_ref=cell['cache'],
            query_identity=cell['fixed_test_identity'],query_order_sha256=cell['query_order_sha256'],
            candidate_order_sha256=cell['candidate_order_sha256'],
            producer_run_refs={k:old.get(k) for k in ('protocol_ref','receipt_ref')} if old else {}))
        failures += [dict(cell=f'{key[0]}/s{key[1]}',reason=reason) for reason in reasons]
    qualified = not failures and all(row['cell_evidence_complete'] for row in outputs)
    for row in outputs:
        row['qualified'] = qualified  # 禁止把其中三格单独标记可复用。
    return dict(schema=SCHEMA,status='PASS',qualified=qualified,all_four_or_none=True,
        baseline_cells=outputs,registered_baseline_jobs_count=0 if qualified else 4,
        registered_new_cap_jobs_count=8,new_evaluation_jobs_count=8 if qualified else 12,
        logical_test_cells=12,findings=failures,expected_cells_sha256=expected_sha,
        validation_preselected=lock['preselected'],validation_lock_sha256=request['validation_lock_ref']['sha256'],
        selection_performed=False,test_reselection_allowed=False,promotion=False,
        independent_acceptance_status='unavailable',test_predictions_opened=False,test_tensors_opened=False,
        source_sha256={own:own_sha},metadata_only=True,scientific_hashes_verified_by_inventory=qualified)


class Ineligible(Exception):
    """已读证据明确缺项/不符；与读路径/字节损坏等操作错误区分。"""


def match(ok, reason):
    if not ok:
        raise Ineligible(reason)


def check_cell(expected, old, historical, loaded, metadata):
    frozen, queue, review, inventory, states = (loaded[k] for k in
        ('frozen','queue','execution_review','inventory','state_proof'))
    config,seed = expected['config'],expected['seed']
    legacy = LEGACY[config]
    try:
        prior = frozen['fixed_center_configurations'][legacy][str(seed)]
        match(prior['seed'] == seed and type(prior['seed']) is int and prior['cohort'] == 'test' and
              prior['recall'] == expected['recall'], 'prior_named_identity')
        match((prior['cache'],prior['cache_sha256']) == canonical_ref(expected['cache']), 'cache_identity_changed')
        match(frozen['fixed_test_identity'] == expected['fixed_test_identity'] and
              frozen['candidate_order_sha256'] == expected['candidate_order_sha256'] and
              frozen['candidate_count'] == 114982 and frozen['test_variants'] == VARIANTS,
              'query_mask_variants_or_candidate_order_changed')
        match(len(expected['source_sha256']) == len(SOURCE_NAMES) and
              set(Path(p).name for p in expected['source_sha256']) == SOURCE_NAMES,
              'incomplete_or_changed_scientific_source_set')
        job_id=f'sealed_{legacy}_s{seed}'
        jobs=[j for j in queue['jobs'] if j['id']==job_id]
        match(len(jobs)==1, 'prior_unique_job')
        job=jobs[0]
        args=parse_legacy_argv(job['argv'])
        match(args['config']==legacy and args['seed']==seed and args['cache']==expected['cache']['path'] and
              args['out']==job['out'] and args['frozen_selection']==historical['frozen_ref']['path'], 'prior_job_semantic_argv')
        protocol_ref,receipt_ref=old.get('protocol_ref'),old.get('receipt_ref')
        match(protocol_ref is not None and receipt_ref is not None, 'missing_protocol_or_receipt')
        for ref in (protocol_ref,receipt_ref):
            match(review['reviewed_sha256'].get(ref['path'])==ref['sha256'], 'prior_review_protocol_receipt_binding')
        protocol=metadata.load(protocol_ref);receipt=metadata.load(receipt_ref)
        match(protocol['args']==args, 'prior_protocol_matches_actual_argv')
        match(receipt['state']=='complete' and strict_zero(receipt['internal_exit']) and
              strict_zero(receipt['external_exit']) and type(receipt['pid']) is int and receipt['pid']>0 and
              receipt['argv']==job['argv'] and receipt['cwd']==job['cwd'] and receipt['output']==job['out'] and
              utc(receipt['started_utc'])<=utc(receipt['finished_utc']), 'prior_real_double0_argv_PID_UTC')
        match(receipt['queue_source_sha256'].get(historical['queue_ref']['path']) == historical['queue_ref']['sha256'],
              'prior_actual_queue_binding')
        merged=dict(expected['source_sha256'])
        for role in ('no_aug','aug'):
            wanted=expected['checkpoint_roles'][role]
            match((prior[role+'_checkpoint'],prior[role+'_sha256'])==canonical_ref(wanted['checkpoint']),
                  'checkpoint_role_or_byteSHA_changed')
            match(args[role+'_checkpoint']==wanted['checkpoint']['path'], 'prior_argv_checkpoint_role')
            center=prior['center_artifacts'][0 if role=='no_aug' else 1]
            match((center['centers'],center['centers_sha256'])==canonical_ref(wanted['centers']) and
                  (center['row_audit'],center['row_audit_sha256'])==canonical_ref(wanted['center_rows']),
                  'center36_or_center_rows_changed')
            state=states['roles'][str(seed)][role]
            match(ref_equal(state['checkpoint'],wanted['checkpoint']) and
                  ref_equal(state['centers'],wanted['centers']) and ref_equal(state['center_rows'],wanted['center_rows']),
                  'state_proof_not_bound_to_exact_checkpoint')
            # 完整 proof 必须显式给出这些状态信息；不因为 checkpoint 名字相同而补造。
            match(type(state['alpha']) in (int,float) and state['alpha']==wanted.get('alpha') and
                  set(state['tensor_sha256'])==ROLE_DIGESTS and
                  all(valid_sha(v) for v in state['tensor_sha256'].values()) and
                  state['tensor_sha256']==wanted.get('tensor_sha256'), 'alpha_O_centers_full33scale_not_proven_equal')
            for name in ('checkpoint','centers','center_rows'):
                path,h=canonical_ref(wanted[name]);merged[path]=h
        path,h=canonical_ref(expected['cache']);merged[path]=h
        for path,h in merged.items():
            match(frozen['evaluation_inputs_sha256'].get(path)==h and job['inputs_sha256'].get(path)==h and
                  protocol['inputs_sha256'].get(path)==h and receipt['requested_inputs_sha256'].get(path)==h and
                  review['reviewed_sha256'].get(path)==h and inventory['before_sha256'].get(path)==h,
                  'scientific_input_or_source_identity_changed')
        identity=states['query_identity']
        match(identity['fixed_test_identity']==expected['fixed_test_identity'] and
              identity['query_order_sha256']==expected['query_order_sha256'] and
              identity['conditions']==expected['conditions'], 'full1371_order_mask_conditions_not_proven')
        pred_path,pred_sha=canonical_ref(old['predictions'])
        match(Path(pred_path)==Path(job['out'])/'predictions.jsonl', 'prior_prediction_output_identity')
        match(review['reviewed_sha256'].get(pred_path)==pred_sha and
              inventory['before_sha256'].get(pred_path)==pred_sha, 'registered_prior_prediction_byteSHA_not_matched')
        evidence=[v for v in review['job_evidence'] if v['config']==legacy and v['seed']==seed]
        match(len(evidence)==1 and evidence[0]['pid']==receipt['pid'] and
              strict_zero(evidence[0]['internal_exit']) and strict_zero(evidence[0]['external_exit']) and
              evidence[0]['predictions_sha256']==pred_sha, 'independent_prior_job_identity')
    except MetadataError:
        raise
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise Ineligible('missing_or_malformed_prior_field:'+str(exc)) from exc


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--request',required=True)
    parser.add_argument('--request-sha256',required=True)
    parser.add_argument('--out',required=True)
    args=parser.parse_args(argv)
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    metadata=Metadata()
    try:
        metadata.pin_source()  # request 自身也是第一次metadata读取，必须在它之前固定源码。
        request=metadata.load(dict(path=str(Path(args.request).absolute()),sha256=args.request_sha256))
        result=qualify(request,metadata)
        after=metadata.finish()
        result.update(utc=datetime.now(timezone.utc).isoformat(),input_sha256=dict(metadata.before),
                      input_end_sha256=after,source_end_sha256=dict(metadata.source_before),
                      inputs_unchanged=True,actual_exit=0)
        (out/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
        (out/'EXIT_CODE').write_text('0\n')
        print(json.dumps({k:result[k] for k in ('status','qualified','registered_baseline_jobs_count','new_evaluation_jobs_count')}))
    except BaseException as exc:
        (out/'failure.json').write_text(json.dumps(dict(error=repr(exc),input_sha256=metadata.before),indent=2)+'\n')
        (out/'EXIT_CODE').write_text('1\n')
        raise


if __name__=='__main__':
    main()
