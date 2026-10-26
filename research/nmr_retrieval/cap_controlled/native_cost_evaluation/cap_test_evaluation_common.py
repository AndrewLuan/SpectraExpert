"""功能概述：独立sealed阶段的固定身份、metadata准入和原始预测契约。
本模块由cap_test_evaluate_ranker调用，与已执行validation四源码分开保存。
先读取冻结/计划/独审/validation lock等JSON并核全闭包声明；全部通过后才返回
允许读取登记test缓存的Pins。只运行固定12逻辑cells，既有预选不再变更。
纯指标/自然512规则取自已审validation实现的固定快照，无选择函数。
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
CANDIDATES=114982
ORDER_SHA='72b72140c09ae14ae8d0aa4edad624a151093c9df246b2052de8a3a4ea616bad'
BASELINE='fixedcenter_mixed_listwise_single'
EPS=1e-7
IDENTITY=('query_order','variant','source','record_id','smiles','group')
CONFIGS=('fixed_single','fixed_dual','cap5_single','cap5_dual','cap10_single','cap10_dual')
TEST_COUNTS={'observed':1000,'frequency_crop':200,'peak_dropout':171}
CACHE_MANIFEST_KEYS=('schema','cohort','recall','candidate_count','pool_size','candidate_order_sha256',
 'query_sha256','checkpoint_sha256','source_inputs_sha256','identity_content_sha256',
 'variant_mask_content_sha256','records','base_queries','variants')
IDENTITY_KEYS=('query_sha256','identity_content_sha256','variant_mask_content_sha256')


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

def verify_cache_manifest(loaded,bound,producer,cache_sha):
    """native producer在torch.save后补payloadSHA；逐payload字段强核，额外producer字段有字节pin。"""
    need(all(k in loaded and k in bound and k in producer for k in CACHE_MANIFEST_KEYS),'cache_core_identity_complete')
    need(all(bound.get(k)==v and producer.get(k)==v for k,v in loaded.items()),'loaded_manifest_exact_binding')
    need(all(producer.get(k)==v for k,v in bound.items()),'bound_producer_manifest_exact')
    need(producer.get('feature_cache_sha256')==cache_sha,'producer_payload_SHA')
    if 'feature_cache_sha256' in bound:need(bound['feature_cache_sha256']==cache_sha,'bound_payload_SHA')
    need(loaded['candidate_count']==CANDIDATES and loaded['pool_size']==512,'manifest_fixed_universe')
    return True


def hexsha(value):return type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value)

def sha_map(value,label):
    need(type(value) is dict and bool(value),label+'_nonempty')
    need(all(type(k) is str and str(safe(k,True))==k and hexsha(v) for k,v in value.items()),label+'_canonical_hex_SHA')
    return value

def metadata_json(pins,path,expected=None):
    p=safe(path,True)
    need(p.suffix=='.json' and not p.name.startswith('predictions'),'gate_metadata_JSON_only')
    return pins.json(p,expected)

def read_ref(pins,ref):
    need(type(ref) is dict and type(ref.get('path')) is str and hexsha(ref.get('sha256')),'metadata_ref')
    return metadata_json(pins,ref['path'],ref['sha256'])

def reviewed(review,required,label):
    need(review.get('status')=='PASS' and review.get('independent') is True and bool(review.get('reviewer')),label+'_independent_PASS')
    need(type(review.get('scope')) is str and bool(review['scope']),label+'_scope')
    have=review.get('reviewed_sha256',{})
    need(all(have.get(p)==h for p,h in required.items()),label+'_complete_exact_SHA')

def sealed_gate(args,source_paths,require_cell=True):
    """只读明确JSON与源码进行准入；返回之前不hash/打开任何test payload或prediction。"""
    pins=Pins(allow_test=True)
    for p in source_paths:pins.add(p)
    protocol=validate_protocol(metadata_json(pins,args.protocol))
    frozen=metadata_json(pins,args.frozen_selection)
    need(frozen.get('schema')=='cap10_sealed_selection_20261024','sealed_frozen_schema')
    need(frozen.get('selection_locked') is True and frozen.get('promotion') is False,'frozen_locked_no_promotion')
    need(frozen.get('preselected') in (BASELINE,'cap10_dual'),'registered_frozen_preselected')
    bindings=metadata_json(pins,args.bindings)
    need(bindings.get('schema')=='cap10_test_bindings_20261024','test_registry_schema')
    need(safe(args.source,True)==safe(bindings['source_checkout'],True),'source_checkout_binding')
    plan=read_ref(pins,frozen['test_plan'])
    need(plan.get('entries')==frozen.get('entries')==bindings.get('entries'),'plan_frozen_registry_exact_entries')
    need(plan.get('preselected')==frozen['preselected'] and plan.get('selection_locked') is True,'plan_selection_lock')
    need(plan.get('promotion') is False and plan.get('selection_performed') is False,'plan_no_reselection')
    required={}
    def merge(mapping,label):
        for p,h in sha_map(mapping,label).items():
            need(p not in required or required[p]==h,'conflicting_registered_SHA')
            required[p]=h
    merge(frozen['input_sha256'],'frozen_inputs');merge(frozen['source_sha256'],'frozen_sources')
    merge(bindings['input_sha256'],'registry_inputs');merge(bindings['source_sha256'],'registry_sources')
    need(all(frozen['input_sha256'].get(p)==h for p,h in bindings['input_sha256'].items()),'registry_inputs_frozen')
    need(all(frozen['source_sha256'].get(p)==h for p,h in bindings['source_sha256'].items()),'registry_sources_frozen')
    entries=frozen['entries'];need(type(entries) is list and len(entries)==12,'twelve_logical_cells')
    keys=[]
    for entry in entries:
        key=(entry['config'],entry['seed']);keys.append(key)
        need(entry['config'] in CONFIGS and type(entry['seed']) is int and entry['seed'] in SEEDS,'entry_config_seed')
        need(entry['pool']==entry['config'].rsplit('_',1)[1],'entry_named_pool')
        need(entry.get('execution') in ('evaluate','reuse'),'entry_execution')
        need(hexsha(entry['cache_sha256']),'entry_cache_SHA')
        reg=bindings['caches'][str(entry['seed'])]['test'][entry['pool']]
        need(safe(entry['cache'],True)==safe(reg['path'],True) and entry['cache_sha256']==reg['sha256'],'entry_cache_registry')
        need(required.get(str(safe(entry['cache'],True)))==entry['cache_sha256'],'cache_in_frozen_inputs')
        need(set(entry['checkpoints'])=={'no_aug','aug'},'exact_checkpoint_roles')
        for role,r in entry['checkpoints'].items():
            need(type(r) is dict and hexsha(r.get('sha256')),'checkpoint_ref')
            path=r.get('path');need(type(path) is str and Path(path).suffix=='.pt','checkpoint_path')
            need(required.get(str(safe(path,True)))==r['sha256'],'checkpoint_in_frozen_inputs')
            need(str(safe(Path(path).parent/'EXIT_CODE',True)) in required,'checkpoint_producer_EXIT_frozen')
            for k in ('centers','center_rows','producer_summary'):
                need(type(r.get(k)) is str and hexsha(r.get(k+'_sha256')),'checkpoint_metadata_ref')
                need(required.get(str(safe(r[k],True)))==r[k+'_sha256'],'checkpoint_metadata_frozen')
            need(type(r.get('epoch')) is int and r['epoch']>=0 and finite(r.get('alpha')) and 0<=r['alpha']<=1,'registered_epoch_alpha')
            if not entry['config'].startswith('fixed_'):need(r['epoch']==64 and r['alpha']==1.,'newcap_final64_alpha1')
    need(len(set(keys))==12 and set(keys)=={(c,s) for c in CONFIGS for s in SEEDS},'complete_six_by_two')
    for family in ('fixed','cap5','cap10'):
        for seed in SEEDS:
            paired=[next(e for e in entries if (e['config'],e['seed'])==(family+'_'+pool,seed)) for pool in POOLS]
            need(paired[0]['checkpoints']==paired[1]['checkpoints'],'same_weights_across_single_dual')
    modes={e['execution'] for e in entries if e['config'].startswith('fixed_')}
    need(len(modes)==1,'all_four_baselines_or_none_reused')
    need(all(e['execution']=='evaluate' for e in entries if not e['config'].startswith('fixed_')),'new_eight_evaluated')
    identity_fixed=frozen['fixed_test_identity']
    need(type(identity_fixed) is dict and set(identity_fixed)==set(IDENTITY_KEYS) and all(hexsha(v) for v in identity_fixed.values()),'exact_three_test_identity_SHA')
    for seed in SEEDS:
        for pool in POOLS:
            reg=bindings['caches'][str(seed)]['test'][pool]
            need(required.get(str(safe(reg['producer']['path'],True)))==reg['producer']['sha256'],'cache_producer_frozen')
            need(str(safe(reg['producer']['exit_path'],True)) in required,'cache_producer_EXIT_frozen')
            manifest=reg['manifest']
            need(all(manifest.get(k)==v for k,v in identity_fixed.items()),'cache_registry_test_identity')
            need(manifest.get('cohort')=='test' and manifest.get('recall')==pool and manifest.get('candidate_count')==CANDIDATES and manifest.get('candidate_order_sha256')==ORDER_SHA and manifest.get('pool_size')==512,'cache_registry_universe')
            need(manifest.get('records')==sum(TEST_COUNTS.values()) and manifest.get('base_queries')==TEST_COUNTS['observed'] and manifest.get('variants')==TEST_COUNTS,'cache_registry_full1371')
    for name in ('cap_module','fixed_module','evaluator'):
        ref=bindings[name];need(frozen['source_sha256'].get(str(safe(ref['path'],True)))==ref['sha256'],'runtime_module_frozen_source')
    val=read_ref(pins,frozen['validation_summary']);selection=read_ref(pins,frozen['validation_selection'])
    valreview=read_ref(pins,frozen['validation_review'])
    need(val.get('status')=='PASS' and val.get('test_read') is False and val.get('promotion') is False,'validation_summary_SCOPE_PASS')
    need(val.get('preselected')==selection.get('preselected')==frozen['preselected'] and selection.get('selection_locked') is True,'validation_preselected_unchanged')
    reviewed(valreview,{frozen['validation_summary']['path']:frozen['validation_summary']['sha256']},'validation_review')
    need('validation' in valreview['scope'],'validation_result_review_scope')
    prereview=read_ref(pins,frozen['freeze_review'])
    reviewed(prereview,{frozen['test_plan']['path']:frozen['test_plan']['sha256']},'test_plan_review')
    final=metadata_json(pins,args.freeze_review)
    need('freeze' in final.get('scope','') and 'test' in final['scope'],'specific_test_freeze_review_scope')
    # 最终专项独审覆盖实际frozen本身、全部metadata/read sources及全部注册输入；它不写回frozen。
    merge(pins.values,'gate_metadata_and_sources')
    required.pop(str(safe(args.freeze_review,True)),None)
    reviewed(final,required,'specific_test_freeze_review')
    entry=None
    if require_cell:
        entry=next(e for e in entries if (e['config'],e['seed'])==(args.config,args.seed))
        need(entry['execution']=='evaluate','reused_baseline_must_not_evaluate_again')
        need(safe(args.cache,True)==safe(entry['cache'],True),'CLI_cache_exact_entry')
        need(safe(args.out,True)==safe(entry['job_out'],True),'CLI_out_exact_entry')
    pins.finish()
    return pins,protocol,bindings,frozen,entry


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
