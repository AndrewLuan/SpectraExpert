"""功能概述：本轮C10/C5评估与汇总共用的纯数据契约，不训练、不调用模型。
need/finite/ratio用于严格类型和空分母处理；Pins按同次字节读取登记SHA并在结束复核，
safe限制当前允许的阶段路径；identity/metrics/signals把完整原始query先合并cohort、
再按两seed等权汇总。select仅接受C10dual为候选，C5dual仅为控制，single仅为机制；
失败直接保留fixedcenter_mixed_listwise_single。driver_receipts核真实内外退出和UTC。
本目录仅供resource-ready恢复后的纯汇总读取；科学评估仍由native_cost_evaluation原叶源码执行。
只写调用方指定的新输出，不修改任何训练权重、旧实验或正式系统。
"""
import collections
import datetime
import hashlib
import json
import math
import struct
from pathlib import Path

SEEDS=(193,211)
CAPS=(5,10)
POOLS=('single','dual')
VARIANTS=('observed','frequency_crop','peak_dropout')
SOURCES=('HMDB','NMRexp','WanJuan','nmrshiftdb2_2018','nmrshiftdb2_2024')
COHORT_OBSERVED={'original_val':320,'tune':1024}
MERGED_OBSERVED=1344
CANDIDATES=114982
ORDER_SHA='72b72140c09ae14ae8d0aa4edad624a151093c9df246b2052de8a3a4ea616bad'
BASELINE='fixedcenter_mixed_listwise_single'
EPS=1e-7
GUARDS=('top1','frequency_crop_MRR','peak_dropout_MRR','macro_MRR')
BASELINE_SIGNALS={'MRR':0.5848916643734858,'top1':0.5163690476190477,
 'frequency_crop_MRR':0.26751386238989017,'peak_dropout_MRR':0.2196237084887636,'macro_MRR':0.43440429414067594}
IDENTITY=('query_order','variant','source','record_id','smiles','group')

def need(value,message):
    if not value:raise ValueError(message)
def finite(x):return type(x) in (int,float) and math.isfinite(x)
def ratio(n,d):return n/d if d else None
def safe(path,allow_test=False):
    p=Path(path).expanduser()
    need('..' not in p.parts,'parent_traversal')
    p=p.resolve()
    if not allow_test:
        need(not any(x in ('test','sealed') or x.startswith(('sealed-','sealed_resume_')) for x in p.parts),'test_path_forbidden')
        need(p.name not in ('research_report.md','supplementary_report.md','sealed_test_report.md','cohorts.pt'),'legacy_test_or_payload_forbidden')
    return p
def sha(path,allow_test=False):
    h=hashlib.sha256()
    with safe(path,allow_test).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def parse_utc(x):
    need(type(x) is str,'utc_string')
    t=datetime.datetime.fromisoformat(x.replace('Z','+00:00'))
    need(t.utcoffset()==datetime.timedelta(0),'utc_required')
    return t
def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def identity(row):
    q=row.get('query_identity',row)
    need(all(k in q for k in IDENTITY),'query_identity_complete')
    need(type(q['query_order']) is int and q['variant'] in VARIANTS and q['source'] in SOURCES,'query_identity_values')
    return tuple(q[k] for k in IDENTITY)
def identity_sha(row):
    return hashlib.sha256(json.dumps(identity(row),ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def ordered_sha(values):return hashlib.sha256('\n'.join(values).encode()).hexdigest()

class Pins:
    """登记已读小输入和大文件SHA；finish重核已登记内容。deferred只供outer核大缓存。"""
    def __init__(self,allow_test=False):self.values={};self.allow_test=allow_test;self.deferred={}
    def add(self,path,expected=None):
        p=safe(path,self.allow_test)
        if str(p) in self.deferred:
            need(expected==self.deferred[str(p)],'deferred_input_requires_exact_expected_SHA')
            return expected
        actual=sha(p,self.allow_test)
        need(expected is None or expected==actual,'input_sha:'+str(p))
        need(str(p) not in self.values or self.values[str(p)]==actual,'input_changed_during_read')
        self.values[str(p)]=actual;return actual
    def raw(self,path,expected=None):
        p=safe(path,self.allow_test);data=p.read_bytes();actual=hashlib.sha256(data).hexdigest()
        need(expected is None or expected==actual,'input_sha:'+str(p))
        need(str(p) not in self.values or self.values[str(p)]==actual,'input_changed_during_read')
        self.values[str(p)]=actual;return data
    def json(self,path,expected=None):return json.loads(self.raw(path,expected))
    def defer(self,path,expected):
        p=str(safe(path,self.allow_test));need(p not in self.values or self.values[p]==expected,'deferred_pin_conflict')
        self.values[p]=expected;self.deferred[p]=expected
    def finish(self):
        for path,digest in self.values.items():
            if path not in self.deferred:need(sha(path,self.allow_test)==digest,'ending_input_sha:'+path)
        return True

def validate_protocol(p):
    need(p['schema']=='cap10_controlled_protocol_20261024','protocol_schema')
    need(p['seeds']==list(SEEDS) and p['caps']==list(CAPS),'fixed_seeds_caps')
    need(p['pool_size']==512 and p['candidate_library_size']==CANDIDATES,'fixed_universe')
    need(p['promotion'] is False and p['test_allowed'] is False,'no_test_or_promotion')
    need(p['final_epoch']==64 and p['alpha']==1.0 and type(p['alpha']) is not bool and p['epoch_offset']==0,'fixed_final_alpha')
    need(p['only_eligible_new_config']=='cap10_dual' and p['control_config']=='cap5_dual' and p['preselected']==BASELINE,'registered_selection_roles')
    return p

def validate_prediction(row):
    identity(row)
    need(type(row['target_in_pool']) is bool and type(row['rank']) is int and 1<=row['rank']<=CANDIDATES,'rank')
    need(finite(row['RR']) and abs(row['RR']-1/row['rank'])<1e-15,'RR')
    if row['target_in_pool']:
        need(1<=row['rank']<=512 and row['in_pool_target_rank']==row['rank'],'hit_rank')
    else:
        need(row['rank']>512 and row['cached_outside_pool_rank']==row['rank'] and row['in_pool_target_rank'] is None,'native_outside_fallback')
    return row

def metrics(rows):
    n=len(rows)
    if not n:return {'queries':0,'MRR':None,'top1':None,'top5':None,'top10':None,'recall512':None,'in_pool_MRR':None}
    hit=[r for r in rows if r['target_in_pool']]
    return dict(queries=n,MRR=math.fsum(r['RR'] for r in rows)/n,
        **{'top'+str(k):sum(r['rank']<=k for r in rows)/n for k in (1,5,10)},
        recall512=len(hit)/n,in_pool_MRR=ratio(math.fsum(r['RR'] for r in hit),len(hit)))

def signal_for_seed(rows):
    normal=[r for r in rows if r['query_identity']['variant']=='observed']
    need(len(normal)==MERGED_OBSERVED,'observed_merged1344')
    bysource={s:metrics([r for r in normal if r['query_identity']['source']==s]) for s in SOURCES}
    need(all(v['queries'] for v in bysource.values()),'five_sources_required')
    crop=metrics([r for r in rows if r['query_identity']['variant']=='frequency_crop'])
    drop=metrics([r for r in rows if r['query_identity']['variant']=='peak_dropout'])
    need(crop['queries'] and drop['queries'],'stress_required')
    return dict(MRR=metrics(normal)['MRR'],top1=metrics(normal)['top1'],
        frequency_crop_MRR=crop['MRR'],peak_dropout_MRR=drop['MRR'],
        macro_MRR=math.fsum(v['MRR'] for v in bysource.values())/5)

def signals(per_seed):
    need(set(per_seed)==set(SEEDS),'two_seeds_required')
    parts={str(s):signal_for_seed(per_seed[s]) for s in SEEDS}
    return {k:math.fsum(parts[str(s)][k] for s in SEEDS)/2 for k in BASELINE_SIGNALS},parts

def select(candidate,control,baseline):
    need({k:baseline[k] for k in BASELINE_SIGNALS}==BASELINE_SIGNALS,'exact_registered_baseline')
    need(all(finite(x[k]) for x in (candidate,control,baseline) for k in BASELINE_SIGNALS),'finite_signals')
    gates={'MRR_gt_fixed_single':candidate['MRR']>baseline['MRR']+1e-12,
           'MRR_gt_matched_C5dual':candidate['MRR']>control['MRR']+1e-12}
    gates.update({k:candidate[k]>=baseline[k]-1e-12 for k in GUARDS})
    eligible=all(gates.values())
    return dict(candidate='C10dual',matched_control='C5dual',gates=gates,eligible=eligible,
        preselected='cap10_dual' if eligible else BASELINE,selection_scope='train_validation_only',
        alternate_selection_allowed=False,single_pool_role='mechanism_only',promotion=False,
        delta_fixed={k:candidate[k]-baseline[k] for k in BASELINE_SIGNALS},
        delta_matched_C5={k:candidate[k]-control[k] for k in BASELINE_SIGNALS})

def driver_receipts(pins,queue_path,run_path):
    """核本轮supervise_controlled_queue producer、isolation回执及始末大输入SHA。"""
    qpath=safe(queue_path);queue=pins.json(qpath);run=safe(run_path)
    need(queue['schema']=='cap10_controlled_queue_20261024','controlled_queue_schema')
    need(run==qpath.parent/'runs'/queue.get('run_name',queue['phase']),'controlled_run_path')
    launch=pins.json(run/'launch_receipt.json');start=pins.json(run/'driver_start.json')
    complete=pins.json(run/'completion.json');qsha=pins.values[str(qpath)]
    need(start['queue_sha256']==complete['queue_sha256']==launch['queue_sha256']==qsha,'driver_queue_SHA')
    need(launch['source_sha256']==queue['source_sha256'],'outer_source_binding')
    need(start['argv']==launch['argv'][2:],'outer_actual_child_command')
    operator_paths=queue.get('operator_paths')
    supervisor=qpath.parent/'supervise_controlled_queue.py'
    if operator_paths is not None:
        need(type(operator_paths) is dict and set(operator_paths)=={'launcher','supervisor'},'exact_operator_path_roles')
        wait=queue.get('resource_wait')
        need(type(wait) is dict and set(wait)=={'maximum_seconds','poll_seconds'} and type(wait['maximum_seconds']) is int and wait['maximum_seconds']==120 and type(wait['poll_seconds']) in (int,float) and wait['poll_seconds']==.2,'exact_bounded_resource_wait')
        names={'launcher':'launch_resource_ready_stage.py','supervisor':'supervise_resource_ready_queue.py'}
        for role,name in names.items():
            ref=operator_paths[role]
            need(type(ref) is dict and set(ref)=={'path','sha256'},'explicit_operator_path_SHA_ref')
            expected=qpath.parent/name
            need(type(ref['path']) is str and ref['path']==str(expected) and safe(ref['path'])==expected,'canonical_resource_operator_path')
            need(type(ref['sha256']) is str and len(ref['sha256'])==64 and all(ch in '0123456789abcdef' for ch in ref['sha256']),'operator_SHA256_hex')
            need(queue['source_sha256'].get(ref['path'])==ref['sha256'],'operator_source_SHA_binding')
            pins.add(ref['path'],ref['sha256'])
        for name in ('cap_evaluation_common.py','summarize_cap_validation.py','analyze_cap_mechanism.py'):
            reader_path=Path(__file__).with_name(name).resolve()
            need(queue['source_sha256'].get(str(reader_path))==sha(reader_path),'resource_analysis_sources_pinned')
        need(all(record.get('operator_paths')==operator_paths for record in (start,complete,launch)),'resource_operator_receipt_refs')
        need(start.get('resource_wait')==complete.get('resource_wait')==wait,'resource_wait_receipt_policy')
        supervisor=Path(operator_paths['supervisor']['path'])
    need(launch['argv']==[queue['python'],'-B',str(supervisor),'--queue',str(qpath),'--out',str(run),'--execute'],'outer_exact_argv')
    need(start['input_sha256']==queue['input_sha256'] and start['source_sha256']==queue['source_sha256'],'driver_pins')
    need(bool(queue['source_sha256']),'nonempty_driver_sources')
    for path,digest in queue['source_sha256'].items():pins.add(path,digest)
    before=pins.json(run/'input_fingerprints.before.json');after=pins.json(run/'input_fingerprints.after.json')
    need(before==after==queue['input_sha256'],'driver_complete_science_start_end_SHA')
    smallbefore=pins.json(run/'operator_fingerprints.before.json');smallafter=pins.json(run/'operator_fingerprints.after.json')
    need(smallbefore==smallafter and smallbefore.get(str(qpath))==qsha,'operator_start_end_SHA')
    need(set(smallbefore)==set(queue['source_sha256'])|{str(qpath)}|set(queue['review_paths']),'complete_small_pin_set')
    required=dict(queue['source_sha256']);required[str(qpath)]=qsha
    need(bool(queue['review_paths']),'independent_admission_required')
    for path in queue['review_paths']:
        review=pins.json(path,smallbefore[path])
        need(review['status']=='PASS' and review['independent'] is True and review['admission_phase']==queue['phase'],'stage_admission_review')
        for rp,rh in required.items():
            local=str(Path(queue['local_root'])/Path(rp).relative_to(qpath.parent)) if Path(rp).is_relative_to(qpath.parent) else rp
            need(review['reviewed_sha256'].get(rp,review['reviewed_sha256'].get(local))==rh,'review_exact_source_queue')
    for path,digest in smallbefore.items():pins.add(path,digest)
    for key in ('internal_exit','external_exit'):need(type(launch[key]) is int and launch[key]==0,'outer_real_double_exit0')
    need(pins.raw(run/'EXIT_CODE').decode().strip()=='0' and pins.raw(run/'EXTERNAL_EXIT_CODE').decode().strip()=='0','driver_markers0')
    need(type(complete['external_exit']) is int and complete['external_exit']==0 and complete['status']=='COMPLETE' and complete['inputs_unchanged'] is True,'driver_complete')
    need(type(start['pid']) is int and start['pid']>0 and complete['pid']==launch['child_pid']==start['pid'],'actual_driver_pid')
    need(type(launch['wrapper_pid']) is int and launch['wrapper_pid']>0 and start['parent_pid']==launch['wrapper_pid'],'actual_wrapper_pid')
    need(start['started_utc']==complete['started_utc'],'driver_start_completion_UTC')
    need(parse_utc(launch['started_utc'])<=parse_utc(start['started_utc'])<=parse_utc(complete['finished_utc'])<=parse_utc(launch['finished_utc']),'driver_UTC_order')
    records={r['id']:r for r in complete['jobs']}
    need(len(records)==len(complete['jobs']) and set(records)=={j['id'] for j in queue['jobs']},'complete_job_set')
    for job in queue['jobs']:
        rec=pins.json(run/'jobs'/(job['id']+'.receipt.json'));access=pins.json(job['access_log'])
        need(rec==records[job['id']],'job_receipt_completion')
        need(all(type(rec[k]) is int and rec[k]==0 for k in ('internal_exit','external_exit')),'job_real_double_exit0')
        need(rec['argv']==job['argv'] and rec['cwd']==job['cwd'] and rec['job_out']==job['job_out'] and rec['access_log']==job['access_log'],'job_exact_argv_paths')
        need(type(rec['pid']) is int and rec['pid']>0 and rec['parent_pid']==start['pid'],'job_pid_parent')
        need(rec['queue_sha256']==qsha and rec['requested_inputs_sha256']=={p:before[p] for p in job['allowed_reads']},'job_source_binding')
        need(access['pid']==rec['pid'] and access['parent_pid']==start['pid'] and access['argv']==job['command'],'isolation_actual_process')
        need(type(access['internal_exit']) is int and access['internal_exit']==0 and not access['rejected_paths'],'isolation_scope_EXIT0')
        need(parse_utc(start['started_utc'])<=parse_utc(rec['started_utc'])<=parse_utc(access['started_utc'])<=parse_utc(access['finished_utc'])<=parse_utc(rec['finished_utc'])<=parse_utc(complete['finished_utc']),'job_access_UTC_order')
        need(pins.raw(Path(job['job_out'])/'EXIT_CODE').decode().strip()=='0','job_internal_EXIT0')
    # 后处理工具只读取元数据；tensor SHA由唯一实际operator的始末hash闭环证明。
    for path,digest in before.items():
        if Path(path).suffix=='.pt':pins.defer(path,digest)
    return queue,dict(launch=launch,start=start,completion=complete,verified_input_sha256=before)

def read_evaluation(pins,out,cap,seed,pool,cohort,bindings_path=None,protocol_path=None,keep_candidates=False):
    """逐prediction重算rank/RR和metric，不把producer聚合值当成输入真值。"""
    out=safe(out,pins.allow_test);summary=pins.json(out/'summary.json');protocol=pins.json(out/'protocol.json')
    need(pins.raw(out/'EXIT_CODE').decode().strip()=='0','eval_EXIT0')
    need(summary['schema']=='cap_ranker_evaluation_20261024' and summary['status']=='PASS','eval_schema_status')
    need((summary['cap'],summary['seed'],summary['pool'],summary['cohort'])==(cap,seed,pool,cohort),'eval_role')
    args=protocol['args']
    need((args['cap'],args['seed'],args['pool'],args['cohort'])==(cap,seed,pool,cohort),'eval_args_role')
    need(safe(args['out'],pins.allow_test)==out,'eval_actual_out')
    if bindings_path is not None:need(safe(args['bindings'])==safe(bindings_path),'eval_bound_bindings')
    if protocol_path is not None:need(safe(args['protocol'])==safe(protocol_path),'eval_bound_protocol')
    need(summary['inputs_unchanged'] is True and summary['model_state_before']==summary['model_state_after'],'eval_state_unchanged')
    need(summary['scope']==('fixed_train_pairs_mechanism_only' if cohort=='train' else 'complete_train_validation_evaluation'),'eval_exact_scope')
    need(summary['test_read'] is False and summary['training'] is False and summary['selection_performed'] is False and summary['promotion'] is False,'eval_no_selection_scope')
    need(summary['forward_bank_calls']==summary['external_api_calls']==0,'eval_no_forward_or_API')
    need(summary['pid']==protocol['pid'] and type(summary['pid']) is int and summary['pid']>0,'eval_actual_pid')
    need(parse_utc(summary['started_utc'])<=parse_utc(summary['finished_utc']),'eval_time')
    need(all(summary['input_sha256'].get(p)==h for p,h in protocol['input_sha256'].items()),'eval_initial_pin_subset')
    for p,h in summary['source_sha256'].items():need(summary['input_sha256'].get(p)==h,'eval_source_in_inputs')
    data=pins.raw(out/'predictions.jsonl',summary['predictions_sha256']);rows=[json.loads(line) for line in data.splitlines()]
    need(len(rows)==summary['records'] and len({identity(r) for r in rows})==len(rows),'eval_records_unique')
    need(ordered_sha(sorted(identity_sha(r) for r in rows))==summary['identity_sha256'],'eval_identity_sha')
    for r in rows:
        validate_prediction(r)
        ids=r['natural_candidate_indices'];costs=r['natural_candidate_costs'];tr=r['natural_candidate_trace_sha256']
        need(len(ids)==512 and len(set(ids))==512 and all(type(i) is int and 0<=i<CANDIDATES for i in ids),'natural512_ids')
        need(len(costs)==512 and all(finite(x) for x in costs),'natural512_costs')
        dtype=validate_cost_dtypes(r)
        need(all(cast_native(x,dtype)==x for x in costs),'native_cost_dtype_bytes')
        need(len(r['natural_candidate_coarse_costs'])==512 and all(finite(x) for x in r['natural_candidate_coarse_costs']),'natural512_coarse')
        need(all(cast_native(x,r['coarse_dtype'])==x for x in r['natural_candidate_coarse_costs']),'native_coarse_dtype_bytes')
        need(len(tr)==2 and all(len(x)==512 and all(type(h) is str and len(h)==64 and all(c in '0123456789abcdef' for c in h) for h in x) for x in tr),'candidate_trace_SHA')
        if r['target_in_pool']:
            need(r['target_candidate_index'] in ids,'target_in_natural')
            pos=ids.index(r['target_candidate_index']);target_cost=costs[pos]
            # 原producer执行同dtype tensor加epsilon；浮点重算先还原fp32容忍区间，下面native函数另核。
            threshold=native_threshold(target_cost,dtype)
            need(r['rank']==sum(x<=threshold for x in costs),'natural_rank_cost_recomputed')
            need(r['target_cost']==target_cost,'target_cost')
        need(r['gamma'] in (.25,.75),'public_gamma')
        need(finite(r['reranking_seconds']) and r['reranking_seconds']>=0,'query_timing')
        if cohort!='train':need(r['appended_training_positive'] is False,'validation_no_appended_positive')
        r['natural_candidate_identity_sha256']=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
        r['natural_coarse_cost_sha256']=hashlib.sha256(json.dumps(r['natural_candidate_coarse_costs'],separators=(',',':')).encode()).hexdigest()
        for k in (1,5,10):need(r['top'+str(k)] is (r['rank']<=k),'TopK_rank')
    need(summary['metrics']=={v:metrics([r for r in rows if r['query_identity']['variant']==v]) for v in VARIANTS},'summary_metrics_recomputed')
    for k in ('reranking_seconds_mean','reranking_seconds_p95','elapsed_seconds'):need(finite(summary[k]) and summary[k]>=0,'timing_finite')
    times=sorted(r['reranking_seconds'] for r in rows);need(bool(times),'nonempty_evaluation')
    z=(len(times)-1)*.95;i=int(z);p95=times[i]+(times[min(i+1,len(times)-1)]-times[i])*(z-i)
    need(abs(summary['reranking_seconds_mean']-math.fsum(times)/len(times))<1e-12 and abs(summary['reranking_seconds_p95']-p95)<1e-12,'timing_raw_records_recomputed')
    if not keep_candidates:
        for r in rows:
            for k in ('natural_candidate_indices','natural_candidate_costs','natural_candidate_coarse_costs','natural_candidate_trace_sha256'):r.pop(k)
    return summary,protocol,rows

CACHE_MANIFEST_KEYS=('schema','cohort','recall','candidate_count','pool_size','candidate_order_sha256',
 'query_sha256','checkpoint_sha256','source_inputs_sha256','identity_content_sha256',
 'variant_mask_content_sha256','records','base_queries','variants')

def verify_cache_manifest(loaded,bound,producer,cache_sha):
    """native producer在torch.save后补payloadSHA；逐payload字段强核，额外producer字段有字节pin。"""
    need(all(k in loaded and k in bound and k in producer for k in CACHE_MANIFEST_KEYS),'cache_core_identity_complete')
    need(all(bound.get(k)==v and producer.get(k)==v for k,v in loaded.items()),'loaded_manifest_exact_binding')
    need(all(producer.get(k)==v for k,v in bound.items()),'bound_producer_manifest_exact')
    need(producer.get('feature_cache_sha256')==cache_sha,'producer_payload_SHA')
    if 'feature_cache_sha256' in bound:need(bound['feature_cache_sha256']==cache_sha,'bound_payload_SHA')
    need(loaded['candidate_count']==CANDIDATES and loaded['pool_size']==512,'manifest_fixed_universe')
    return True

def f32(x):
    need(finite(x),'finite_fp32_input')
    return struct.unpack('f',struct.pack('f',x))[0]

def native_cost_fp32(coarse,delta0,delta1,scale0,scale1,gamma):
    """逐步重放eager native fp32乘、加顺序；不将FP64解释代入实际评分或排名。"""
    need(gamma in (.25,.75),'native_gamma')
    a=f32(f32(delta0)*f32(scale0));b=f32(f32(delta1)*f32(scale1))
    return f32(f32(f32(coarse)+f32((1-gamma)*a))+f32(gamma*b))

def fixed_pair_numerics(evidence,gamma):
    """同一target/固定other的mixed-dtype native重放和FP64实数解释；舍入残差显式保留。"""
    role={t['role']:t for t in evidence['role_saturation']}
    need(set(role)=={'no_aug','aug'} and len(evidence['role_saturation'])==2,'fixed_numeric_roles')
    scales=evidence['role_scale0'];need(len(scales)==2 and all(finite(x) and x>0 for x in scales),'fixed_numeric_scales')
    t0,t1=role['no_aug'],role['aug']
    dtype=validate_cost_dtypes(evidence)
    target=native_cost_replay(evidence['target_coarse_cost'],t0['target_delta'],t1['target_delta'],*scales,gamma,evidence['coarse_dtype'],evidence['correction_dtypes'])
    other=native_cost_replay(evidence['original_other_coarse_cost'],t0['original_other_delta'],t1['original_other_delta'],*scales,gamma,evidence['coarse_dtype'],evidence['correction_dtypes'])
    coarse_gap=evidence['target_coarse_cost']-evidence['original_other_coarse_cost']
    correction=math.fsum(w*scale*(t['target_delta']-t['original_other_delta']) for w,scale,t in zip((1-gamma,gamma),scales,(t0,t1)))
    real_margin=coarse_gap+correction
    native_margin=target-other
    return dict(target_native_replay=target,original_other_native_replay=other,
        real_arithmetic_margin=real_margin,real_arithmetic_correction=correction,
        native_margin=native_margin,native_minus_real_margin=native_margin-real_margin)


NATIVE_DTYPES=('torch.float32','torch.float64')
def cast_native(value,dtype):
    need(dtype in NATIVE_DTYPES and finite(value),'native_dtype_finite')
    return struct.unpack('f',struct.pack('f',value))[0] if dtype=='torch.float32' else float(value)
def promoted_dtype(coarse,corrections):
    need(coarse in NATIVE_DTYPES and type(corrections) is list and len(corrections)==2 and all(x in NATIVE_DTYPES for x in corrections),'native_input_dtypes')
    return 'torch.float64' if 'torch.float64' in [coarse]+corrections else 'torch.float32'
def native_threshold(target,dtype):
    return cast_native(cast_native(target,dtype)+cast_native(EPS,dtype),dtype)
def validate_cost_dtypes(row):
    need(row['final_cost_dtype']==promoted_dtype(row['coarse_dtype'],row['correction_dtypes']),'native_cost_promotion')
    return row['final_cost_dtype']
def native_cost_replay(coarse,delta0,delta1,scale0,scale1,gamma,coarse_dtype,correction_dtypes):
    need(gamma in (.25,.75),'native_gamma')
    final=promoted_dtype(coarse_dtype,correction_dtypes)
    a=cast_native(delta0*scale0,correction_dtypes[0]);b=cast_native(delta1*scale1,correction_dtypes[1])
    a=cast_native((1-gamma)*a,correction_dtypes[0]);b=cast_native(gamma*b,correction_dtypes[1])
    first_dtype='torch.float64' if 'torch.float64' in (coarse_dtype,correction_dtypes[0]) else 'torch.float32'
    return cast_native(cast_native(cast_native(coarse,coarse_dtype)+a,first_dtype)+b,final)
