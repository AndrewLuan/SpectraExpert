"""功能概述：冻结计数消融后组合不重叠观测路由，核验全库对照并生成研究汇总。
compose按查询来源在HMDB报告模型和其他来源宽度流程间选择整条结果，不拼候选
分数；两流程共享冻结均值，分别只改变HMDB/稀疏原子观测路由。row_key检查身份，
cluster_bootstrap按非立体连接结构整组重采样，保留同物质多谱的相关性。
main等待36臂及所选评分完成；宽度可靠性策略只用原320-H-val选择，完整H组合
与上一轮生产参照也只按同一val选择。随后复用h_fusion_prepare/workflow在独立
320配对val选融合；C控制、候选次序、物理输入、查询和压力变体必须一致。
audit_means核验所有36臂、权重SHA及严格参数/前向配置别名；不同标签口径的
理论误差由既有独立诊断报告，不在此混比。所有test结果只在相应选择写盘后汇总。
输出独占JSON/JSONL、置信区间、源码/状态和EXIT_CODE；源数据及旧结果只读，
不训练、不改均值、不从DMF测试位移拟合参数，无外部服务。
--prepared及--previous-reference允许独立后续消融显式复用相同缓存和完整生产参照，
默认仍按原36臂的相对路径解析；不根据测试指标选择参照。
"""
import argparse,collections,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from .data import identity
from .evaluate import summarize
from .h_observation_study import summarize_sources,write_rows
from .prepare import sha

H_ORDER='72b72140c09ae14ae8d0aa4edad624a151093c9df246b2052de8a3a4ea616bad'
JOINT_ORDER='161187a22c98077d654ad185fe277cc1b423641bb5424620aa4f65f44f738ac8'
PAIR_VAL='390c28106601864073ddf11e0ea1aab30295be1e9912d8074f605f2f8e93c307'
C_SHA='d968bcd773405c1406d4bc21b39c196ca11ab72e3c302fc71ad0975d6493d799'
H_TEST='cb761a76675e3e7107f85c3fb443ee55b3c4da513988f4e65d057c9282a0fcdb'
PAIRED_H_TEST='2723b3cf16c4881dd541bba751b5c321fb46a66ea85064d03a006b5363e8fe04'


def read(path):return json.loads(Path(path).read_text())
def rows(path):return [json.loads(l) for l in Path(path).read_text().splitlines()]


def row_key(r,stress=False):
    source=r['H_source'] if 'H_source' in r else r.get('source','');rid=r['H_record_id'] if 'H_record_id' in r else r['record_id']
    return (r['variant'],rid,r['smiles']) if stress else (source,rid,r['smiles'])


def index(rs,stress=False):
    result={row_key(r,stress):r for r in rs}
    if len(result)!=len(rs):raise ValueError('duplicate_result_identity')
    return result


def compose(width,reporting,stress=False):
    other=index(reporting,stress);base=index(width,stress)
    if base.keys()!=other.keys():raise ValueError('composition_cohort_mismatch')
    return [dict(other[row_key(r,stress)] if (r['H_source'] if 'H_source' in r else r.get('source'))=='HMDB' else r) for r in width]


def cluster_bootstrap(new,old,stress=False,seed=241,repeats=5000):
    previous=index(old,stress);current=index(new,stress)
    if current.keys()!=previous.keys():raise ValueError('bootstrap_cohort_mismatch')
    order=sorted(current);a=np.array([current[k]['rank'] for k in order],float);b=np.array([previous[k]['rank'] for k in order],float)
    if not np.isfinite(a).all() or not np.isfinite(b).all() or min(a.min(),b.min())<1:raise ValueError('invalid_rank_for_bootstrap')
    groups=[identity(current[k]['smiles'])[1] for k in order];unique=sorted(set(groups));lookup={g:i for i,g in enumerate(unique)};ids=np.array([lookup[g] for g in groups]);counts=np.bincount(ids,minlength=len(unique));draws=np.random.default_rng(seed).integers(0,len(unique),(repeats,len(unique)));denominator=counts[draws].sum(1);metrics={}
    for name,diff in [('MRR',1/a-1/b),('top1',(a==1).astype(float)-(b==1)),('top10',(a<=10).astype(float)-(b<=10))]:
        sums=np.bincount(ids,weights=diff,minlength=len(unique));sample=sums[draws].sum(1)/denominator;metrics[name]=dict(difference=float(diff.mean()),cluster_95_percentile_interval=np.quantile(sample,[.025,.975]).tolist())
    return dict(queries=len(order),connectivity_clusters=len(unique),replicates=repeats,seed=seed,metrics=metrics)


def check_ids(rs,reference,expected,stress=False):
    if len(rs)!=expected or index(rs,stress).keys()!=index(reference,stress).keys():raise ValueError('frozen_query_identity_changed')


def check_carbon(rs,reference):
    previous=index(reference)
    for r in rs:
        c=r['13C'];old=previous[row_key(r)]['13C']
        if c['rank']!=old['rank'] or c['top10']!=old['top10']:raise ValueError('carbon_rank_or_top10_changed')


def flatten_joint(rs):return [dict(r['joint'],H_source=r['H_source'],H_record_id=r['H_record_id'],smiles=r['smiles']) for r in rs]


def audit_means(base,root,frozen):
    reference=root/'evaluation/control';normal=rows(reference/'test/predictions.jsonl');missing=rows(reference/'test/missing_predictions.jsonl');paired=rows(reference/'joint/predictions.jsonl');paired_missing=rows(reference/'joint/missing_predictions.jsonl');carbon=rows(base/'hydrogen_analysis/observation_comparison/joint/predictions.jsonl');h0=read(reference/'test/summary.json');j0=read(reference/'joint/summary.json');report={};aliases=collections.defaultdict(list)
    if sha(base/'atom_emission/13C/domain/best.pt')!=C_SHA:raise ValueError('carbon_checkpoint_changed')
    if h0['query_sha256']!=H_TEST or j0['H_query_sha256']!=PAIRED_H_TEST:raise ValueError('original_test_queries_changed')
    for name,training in frozen['arms'].items():
        folder=root/'evaluation'/name
        if (folder/'test/EXIT_CODE').read_text().strip()!='0' or (folder/'joint/EXIT_CODE').read_text().strip()!='0':raise ValueError('mean_test_incomplete:'+name)
        h=read(folder/'test/summary.json');j=read(folder/'joint/summary.json');hh=rows(folder/'test/predictions.jsonl');hm=rows(folder/'test/missing_predictions.jsonl');jj=rows(folder/'joint/predictions.jsonl');jm=rows(folder/'joint/missing_predictions.jsonl');check_ids(hh,normal,1000);check_ids(hm,missing,371,True);check_ids(jj,paired,792);check_ids(jm,paired_missing,711,True);check_carbon(jj,carbon)
        if h['candidate_order_sha256']!=H_ORDER or j['candidate_order_sha256']!=JOINT_ORDER or h['physics_sha256']!=h0['physics_sha256'] or j['physics_sha256']!=j0['physics_sha256']:raise ValueError('candidate_or_physics_drift:'+name)
        if h['query_sha256']!=h0['query_sha256'] or any(j[k]!=j0[k] for k in ['H_query_sha256','C_query_sha256','C_checkpoint_sha256']):raise ValueError('query_or_C_checkpoint_drift:'+name)
        path=root/training/name/'train/best.pt'
        if sha(path)!=h['checkpoint_sha256'] or h['checkpoint_sha256']!=j['H_checkpoint_sha256']:raise ValueError('mean_checkpoint_drift:'+name)
        saved=torch.load(path,map_location='cpu',weights_only=False);digest=hashlib.sha256();config={k:v for k,v in saved['args'].items() if k not in ['seed','site_quality_mode','experimental_selection_mode']};digest.update(json.dumps(config,sort_keys=True).encode())
        for key,t in sorted(saved['state_dict'].items()):digest.update(key.encode());digest.update(str(t.dtype).encode());digest.update(str(list(t.shape)).encode());digest.update(t.detach().cpu().contiguous().numpy().tobytes())
        aliases[digest.hexdigest()].append(name);report[name]=dict(validation=frozen['validation'][name]['metrics'],H=h['metrics'],joint=j['optimized'],H_missing=h['missing_stress'],joint_missing=j['missing_stress'],checkpoint_sha256=h['checkpoint_sha256'],best_epoch=saved.get('epoch'),best_stage=saved.get('stage'),C_rank_top10_exact=True)
    return dict(arms=report,strict_state_config_aliases=[v for v in aliases.values() if len(v)>1],normal_H_queries=1000,H_stress_queries=371,joint_queries=792,joint_stress_queries=711,H_candidate_order_sha256=H_ORDER,joint_candidate_order_sha256=JOINT_ORDER),normal,paired


def chosen_fusion(folder):
    f=read(folder/'frozen_selection.json');mode=f['selected_joint_mode'];weight=str(f['selected_H_C_weights'][mode]);return f,mode,weight,f['validation'][mode][weight]['MRR']


def main():
    p=argparse.ArgumentParser()
    for name in ['base','root','out']:p.add_argument('--'+name,required=True)
    p.add_argument('--prepared');p.add_argument('--previous-reference')
    a=p.parse_args();base=Path(a.base);root=Path(a.root);out=Path(a.out);out.mkdir(parents=True,exist_ok=False);start=time.monotonic();torch.set_num_threads(1)
    def status(stage,**kw):(out/'status.json').write_text(json.dumps(dict(stage=stage,seconds=time.monotonic()-start,**kw)))
    def run(module,args,log):
        with (out/log).open('x') as f:subprocess.run([sys.executable,'-u','-m',module,*map(str,args)],cwd=root/'formal_driver/source',stdout=f,stderr=subprocess.STDOUT,check=True)
    try:
        status('waiting_formal_evaluation');sentinel=root/'formal_driver/EXIT_CODE'
        while not sentinel.exists():
            if time.monotonic()-start>18000:raise TimeoutError('formal_evaluation_timeout')
            pid=read(root/'formal_driver_pid.json')['pid']
            try:os.kill(pid,0)
            except ProcessLookupError:
                if sentinel.exists():break
                raise RuntimeError('formal_driver_missing_without_exit')
            time.sleep(30)
        if sentinel.read_text().strip()!='0':raise RuntimeError('formal_evaluation_failed')
        oldroot=Path(a.previous_reference) if a.previous_reference else root.parent/'acyl_message_comparison';oldwidth=oldroot/'selected_combined/width';previous=oldroot/'selected_combined/fusion_reporting';scoring=root/'selected_scoring';policies={q:read(scoring/('width_'+q)/'frozen_selection.json') for q in ['original','methyl_record_mask']};quality=max(policies,key=lambda q:policies[q]['validation'][str(policies[q]['selected_lambda'])]['overall']['MRR']);vm={q:policies[q]['validation'][str(policies[q]['selected_lambda'])]['overall']['MRR'] for q in policies}
        for dependency in [oldwidth,previous,oldroot/'original_late/reporting']:
            if (dependency/'EXIT_CODE').read_text().strip()!='0':raise ValueError('previous_production_reference_incomplete')
        if vm[quality]-vm['original']<=.0001:quality='original'
        width=scoring/('width_'+quality);reporting=scoring/'reporting';rf=read(reporting/'frozen_selection.json')
        if read(oldroot/'original_late/reporting/frozen_selection.json')['selected_config']['kind']!='baseline':raise ValueError('previous_full_H_reference_requires_composition')
        if policies[quality]['checkpoint_sha256']!=rf['H_checkpoint_sha256']:raise ValueError('composition_checkpoint_mismatch')
        validation=compose(rows(width/'validation_selected.jsonl'),rows(reporting/'validation_selected.jsonl'));pv=rows(oldwidth/'validation_selected.jsonl');check_ids(validation,pv,320);metrics={'new':summarize_sources(validation),'previous':summarize_sources(pv)};promoted=metrics['new']['overall']['MRR']-metrics['previous']['overall']['MRR']>.0001;selection=dict(width_policy=quality,width_policy_validation_MRR=vm,H_pipeline='new' if promoted else 'previous',H_validation=metrics,mean_variant=read(root/'evaluation/frozen_selection.json')['selected_variant'],H_checkpoint_sha256=rf['H_checkpoint_sha256'],reporting_choice=rf['selected_config'],selection='matching320 H validation only; numerical ties prefer previous; no test values used');(out/'frozen_selection.json').write_text(json.dumps(selection,indent=2));write_rows(out/'validation_selected.jsonl',validation)
        status('paired_validation');common=['--base',base,'--prepared',Path(a.prepared) if a.prepared else root.parent/'solvent_resolved_prepared','--reference-latent',scoring/'latent'];run('nmr_transfer.h_fusion_prepare',common+['--out',out/'paired_validation'],'paired_validation.log');vp=read(out/'paired_validation/protocol.json')
        if vp['query_sha256']!=PAIR_VAL or vp['candidate_order_sha256']!=JOINT_ORDER or vp['queries']!=320:raise ValueError('paired_validation_drift')
        status('joint_fusion');args=common+['--width-evaluation',width,'--validation',out/'paired_validation','--joint-width-grid'];run('nmr_transfer.h_fusion_workflow',args+['--out',out/'fusion'],'fusion.log')
        if rf['selected_config']['kind']=='baseline':(out/'fusion_reporting').symlink_to(out/'fusion',target_is_directory=True);(out/'reporting_alias.json').write_text(json.dumps(dict(reason='validation selected identity reporting; exact same scoring configuration',source='fusion')))
        else:run('nmr_transfer.h_fusion_workflow',args+['--out',out/'fusion_reporting','--reporting-evaluation',reporting],'fusion_reporting.log')
        jf,mode,weight,jval=chosen_fusion(out/'fusion_reporting');pf,pm,pw,pval=chosen_fusion(previous)
        if jf['validation_query_sha256']!=pf['validation_query_sha256'] or jf['validation_query_sha256']!=PAIR_VAL:raise ValueError('joint_selection_validation_mismatch')
        selection['joint_pipeline']='new' if jval-pval>.0001 else 'previous';selection['joint_validation_MRR']=dict(new=jval,previous=pval);selection['joint_new_mode']=mode;selection['joint_new_weight']=weight;selection['joint_previous_mode']=pm;selection['joint_previous_weight']=pw;(out/'frozen_selection.json').write_text(json.dumps(selection,indent=2))
        status('invariant_audit');frozen=read(root/'evaluation/frozen_selection.json');audit,normal,paired=audit_means(base,root,frozen);(out/'raw_ablation_audit.json').write_text(json.dumps(audit,indent=2));combined={}
        for target,relative,stress in [('H','predictions.jsonl',False),('H_missing','missing_predictions.jsonl',True),('joint_equal','joint/predictions.jsonl',False),('joint_missing','joint/missing_predictions.jsonl',True)]:
            rs=compose(rows(width/relative),rows(reporting/relative),stress);combined[target]=rs;dest=out/'composed'/relative;dest.parent.mkdir(parents=True,exist_ok=True);write_rows(dest,rs)
        oldH=rows(oldwidth/'predictions.jsonl');oldJ=rows(oldwidth/'joint/predictions.jsonl');check_ids(combined['H'],oldH,1000);check_ids(combined['joint_equal'],oldJ,792);check_carbon(combined['joint_equal'],oldJ);newcal=rows(out/'fusion_reporting'/(mode+'_weight'+weight+'_predictions.jsonl'));oldcal=rows(previous/(pm+'_weight'+pw+'_predictions.jsonl'));check_ids(newcal,oldcal,792)
        oldmissing={'H_missing':rows(oldwidth/'missing_predictions.jsonl'),'joint_missing':rows(oldwidth/'joint/missing_predictions.jsonl')};check_ids(combined['H_missing'],oldmissing['H_missing'],371,True);check_ids(combined['joint_missing'],oldmissing['joint_missing'],711,True);missing_metrics={}
        for name,oldrows in oldmissing.items():
            missing_metrics[name]={}
            for variant in sorted({r['variant'] for r in combined[name]}):
                nn=[r for r in combined[name] if r['variant']==variant];oo=[r for r in oldrows if r['variant']==variant];missing_metrics[name][variant]=dict(new=summarize(nn),previous=summarize(oo),difference=cluster_bootstrap(nn,oo,True))
        carbon=index(rows(base/'hydrogen_analysis/observation_comparison/joint/predictions.jsonl'))
        for f in (out/'fusion_reporting').glob('*_weight0.0_predictions.jsonl'):
            for r in rows(f):
                cc=carbon[row_key(r)]['13C']
                if r['rank']!=cc['rank'] or r['top10']!=cc['top10']:raise ValueError('fusion_carbon_control_changed')
        selected_mean=root/'evaluation'/frozen['selected_variant'];raw_delta=dict(H=cluster_bootstrap(rows(selected_mean/'test/predictions.jsonl'),normal),joint=cluster_bootstrap(flatten_joint(rows(selected_mean/'joint/predictions.jsonl')),flatten_joint(paired)))
        summary=dict(selection=selection,raw_ablations=audit['arms'],strict_aliases=audit['strict_state_config_aliases'],raw_selected_vs_control=raw_delta,H=dict(new=summarize_sources(combined['H']),previous=summarize_sources(oldH),difference=cluster_bootstrap(combined['H'],oldH)),joint_equal=dict(new=summarize(flatten_joint(combined['joint_equal'])),previous=summarize(flatten_joint(oldJ)),difference=cluster_bootstrap(flatten_joint(combined['joint_equal']),flatten_joint(oldJ))),joint_calibrated=dict(new=summarize(newcal),previous=summarize(oldcal),difference=cluster_bootstrap(newcal,oldcal)),missing=missing_metrics,C_rank_and_top10_exact=True,source_sha256={f.name:sha(f) for f in Path(__file__).parent.glob('*.py')},scope='all choices use train/val; same reused development test cohorts; bootstrap conditional on frozen choices');(out/'summary.json').write_text(json.dumps(summary,indent=2));status('complete');(out/'EXIT_CODE').write_text('0\n')
    except Exception as exc:status('failed',error=repr(exc));(out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':main()
