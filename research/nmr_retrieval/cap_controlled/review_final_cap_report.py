"""功能概述：非作者只读终审本轮报告、小文件SHA清单和已独审汇总/导出。
不读取原始预测或tensor，不复跑科学统计；核对文字表格与直接JSON字段和TSV。
输出唯一核验receipt，由非作者审阅人另写正式报告终审；不改报告与状态。
"""
import csv,json,hashlib,re
from pathlib import Path
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parent
REPORT_SHA='1b7060e7a50812021ffbe308ff01866783e729fee60662d1a653c653d61be264'
MANIFEST_SHA='14708631b5fde65de65e189e9ca3d2b1c3028accd39cb7cbdd512bd9151e1e8a'
EXPECTED_TEST_REVIEW='8ed1bdce56493eeceaf3f24c176998d5b137422b19bfe08ee74a62261970d333'
def need(x,label):
    if not x:raise ValueError(label)
def raw(path):
    p=Path(path);need(p.is_absolute() and p==p.resolve(strict=True),'noncanonical_or_symlink')
    need(p.suffix.lower() not in ('.pt','.pth','.ckpt','.npy','.npz'),'tensor_forbidden')
    need(not any(word in p.name.lower() for word in ('prediction','rankings')) and p.name!='ranks.jsonl','raw_prediction_forbidden')
    return p.read_bytes()
def sha(path):return hashlib.sha256(raw(path)).hexdigest()
def obj(name):return json.loads(raw(ROOT/name))
def check_value(actual,want,label):
    if want is None:need(actual=='','null_not_empty:'+label)
    elif type(want) is str:need(actual==want,'text:'+label)
    else:need(abs(float(actual)-want)<=1e-14,'numeric:'+label)
def six(actual,want,label):need(actual==f'{want:.6f}','six_decimal:'+label)
def tables(report):
    all_tables=[];current=[]
    for line in report.splitlines()+['']:
        if line.startswith('|'):current.append([x.strip() for x in line.strip('|').split('|')])
        elif current:
            need(len(current)>=2,'table_shape');all_tables.append((current[0],current[2:]));current=[]
    return all_tables
def main():
    manifest_path=ROOT/'research_report.input_sha256.json';report_path=ROOT/'research_report.md'
    need(sha(report_path)==REPORT_SHA and sha(manifest_path)==MANIFEST_SHA,'exact_final_report_manifest')
    manifest=obj('research_report.input_sha256.json');inputs=manifest['input_sha256'];outputs=manifest['output_sha256']
    need(len(inputs)==337 and len(outputs)==2 and not set(inputs)&set(outputs),'exact_manifest_shape')
    allpins={**inputs,**outputs};before={}
    for p,h in allpins.items():before[p]=sha(p);need(before[p]==h,'manifest_actual_SHA:'+p)
    need(not any(Path(p) in {ROOT/name for name in ('completion.json','research_report_final_review.json','orchestrator-state.json','commands.jsonl','research_report.input_sha256.json')} for p in allpins),'acyclic_manifest')
    report=raw(report_path).decode()
    for link in re.findall(r'\[[^\]]+\]\(([^)]+)\)',report):
        p=(ROOT/link).resolve(strict=True);need(str(p) in allpins or p==manifest_path,'linked_file_not_manifested:'+link)
    test=obj('test_results_independent_review.json');train=obj('training_results_independent_review.json');val=obj('validation_results_independent_review.json')
    need(sha(ROOT/'test_results_independent_review.json')==EXPECTED_TEST_REVIEW,'exact_actual_test_review')
    for r in (test,train,val):need(r['status']=='PASS' and r['independent'] is True,'actual_reviews_PASS_independent')
    need(train['models']==8 and train['total_history_rows']==512 and train['total_draws']==4194304 and train['total_optimizer_steps']==131072,'actual_train_budget')
    need(test['metric_rows']==324 and test['comparison_strata']==180 and test['bootstrap_seed']==317 and test['bootstrap_replicates']==5000,'actual_test_registry')
    need(test['current_runtime_input_pins']==85 and test['current_queue_source_pins']==212 and sum(len(x['inputs_sha256']) for x in test['historical_frozen_provenance'])==20,'current_vs_historical_scope')
    need(all(x['rehashed_during_test_result_review'] is False for x in test['historical_frozen_provenance']),'historical_rehash_not_claimed')
    need(val['fixed_train_pairs400'] is True and val['original87'] is True and val['common_candidate_invariance'] is True,'actual_mechanism_scope')
    lock=obj('validation_lock.json');selection=obj('validation_selection.json');proto=obj('protocol.json')
    need(lock['selection_locked'] is True and selection['selection_locked'] is False,'acyclic_selection_lock')
    baseline='fixedcenter_mixed_listwise_single'
    need(all(r['preselected']==baseline for r in (lock,selection,test,val)) and all(r['promotion'] is False for r in (lock,selection,test,val,train)),'fixed_preselection_no_promotion')
    need(selection['selection']['eligible'] is False and selection['selection']['gates']==dict(MRR_gt_fixed_single=True,MRR_gt_matched_C5dual=False,top1=True,frequency_crop_MRR=False,peak_dropout_MRR=False,macro_MRR=False),'selection_guards')
    analysis=obj('collected/test/remote/runs/test/derived_summary/analysis.json');summary=obj('collected/test/remote/runs/test/derived_summary/summary.json')
    mr='/home/user/IRfingerprints/SpectrExpert/outputs/nmr_cap10_controlled_20261024T163633Z'
    for name in ('analysis.json','summary.json','metrics.tsv'):
        p=ROOT/'collected/test/remote/runs/test/derived_summary'/name
        need(test['reviewed_sha256'][mr+'/runs/test/derived_summary/'+name]==sha(p),'actual_review_binding:'+name)
    need(summary['native_final_dtype_records']=={'torch.float32':16248,'torch.float64':204} and sum(summary['native_final_dtype_records'].values())==16452,'native_dtype_count')
    need(len(analysis['metrics'])==324 and len(analysis['comparisons'])==180,'analysis_shape')
    comparisons=analysis['comparisons'];need(sum(c['status']=='EMPTY_STRATUM' for c in comparisons)==20,'empty20_retained')
    need({c['comparison'] for c in comparisons}=={x['id'] for x in proto['test_preregistration']['comparisons']},'all10_comparisons')
    source_rows=list(csv.DictReader(raw(ROOT/'collected/test/remote/runs/test/derived_summary/metrics.tsv').decode().splitlines(),delimiter='\t'))
    need(len(source_rows)==324,'metrics_export324')
    for row,want in zip(source_rows,analysis['metrics']):
        for key,value in want.items():check_value(row[key],value,'metrics.'+key)
        need(all(row[k]=='' for k in set(row)-set(want)),'metrics_extra_fields_empty')
    pair_rows=list(csv.DictReader(raw(ROOT/'paired_comparisons.tsv').decode().splitlines(),delimiter='\t'))
    need(len(pair_rows)==180,'paired_export180')
    metric_names=('MRR','top1','top5','top10','recall512','in_pool_MRR','in_pool_RR_contribution','out_of_pool_RR_contribution')
    for row,c in zip(pair_rows,comparisons):
        expected={k:c[k] for k in ('comparison','left','right','variant','source','status','queries_per_seed','connectivity_clusters')}
        for key in row.keys()-expected.keys():expected[key]=None
        if c['status']!='EMPTY_STRATUM':
            for metric in metric_names:
                expected[metric+'_delta']=c['two_seed']['delta'][metric]
                for dst,src in (('CI_lower','lower'),('CI_upper','upper'),('valid_replicates','valid_replicates'),('undefined_replicates','undefined_replicates')):
                    expected[metric+'_'+dst]=c['paired95CI'][metric][src]
            for category in ('common','new','lost','neither'):
                q=c['categories'][category]
                for dst,src in (('query_count_mean','mean_query_count_per_seed'),('full_MRR_contribution','contribution_to_total_delta_MRR'),('conditional_delta_MRR','conditional_delta_MRR')):
                    expected[category+'_'+dst]=q[src]
        for key,want in expected.items():check_value(row[key],want,'pair.'+key)
    parsed=tables(report);need([len(rows) for _,rows in parsed]==[5,4,18,9],'report_tables')
    signals={'fixed_single':{k:x['value'] for k,x in obj('baseline_validation_reference.json')['signals'].items()},**val['signals']}
    fields=('MRR','top1','frequency_crop_MRR','peak_dropout_MRR','macro_MRR')
    for row in parsed[0][1]:
        for text,key in zip(row[1:],fields):six(text,signals[row[0]][key],'validation.'+row[0]+'.'+key)
    mech=obj('collected/mechanism_fixed_rows/remote/runs/mechanism_fixed_rows/derived_summary/summary.json')['fixed_train_pair_analysis']
    need(mech['all400']['query_seed_pairs']==400 and mech['original87']['query_seed_pairs']==87,'actual_mechanism_counts')
    for row in parsed[1][1]:
        cohort,seed,n,m5,m10,b5,b10,margin=row;v=mech[cohort]['per_seed'][seed]
        need(int(n)==v['queries'],'mechanism_n')
        for text,cap in ((m5,'5'),(m10,'10')):six(text,v['caps'][cap]['metrics']['MRR'],'mechanism_MRR')
        need(int(b5)==v['caps']['5']['bound_released_count'] and int(b10)==v['caps']['10']['bound_released_count'],'mechanism_bound_count')
        six(margin,v['paired_fixed_margin_delta']['mean'],'mechanism_margin')
    allmetrics={(x['config'],x['variant']):x for x in analysis['metrics'] if x['seed']=='two_seed_mean' and x['source']=='all'}
    for row in parsed[2][1]:
        variant,config,n,*values=row;v=allmetrics[config,variant];need(int(n)==v['queries'],'test_n')
        for text,key in zip(values,('MRR','top1','top5','top10','recall512','in_pool_MRR')):six(text,v[key],'test.'+config+'.'+variant+'.'+key)
    cmpmap={(c['comparison'],c['variant']):c for c in comparisons if c['source']=='all'}
    for row in parsed[3][1]:
        comp,variant,mrr,ci,top1,recall=row;v=cmpmap[comp,variant];six(mrr,v['two_seed']['delta']['MRR'],'report_comparison_MRR')
        six(top1,v['two_seed']['delta']['top1'],'report_comparison_top1');six(recall,v['two_seed']['delta']['recall512'],'report_comparison_recall')
        q=v['paired95CI']['MRR'];need(ci==f'[{q["lower"]:.6f}, {q["upper"]:.6f}]','report_CI_six')
    for cohort,value in (('all400','+0.041428'),('original87','+0.000759')):
        need(f'{mech[cohort]["equal_two_seed_mean_C10_minus_C5"]["MRR"]:+.6f}'==value and value in report,'mechanism_text_delta')
    need(mech['original87']['total_bound_release_count']['10']==66 and all(v['caps']['10']['metrics']['top10']==0 for v in mech['original87']['per_seed'].values()),'original87_necessary_not_rank_success')
    need({c['variant']:c['connectivity_clusters'] for c in comparisons if c['comparison']=='cap10dual_minus_cap5dual' and c['source']=='all'}==dict(observed=999,frequency_crop=200,peak_dropout=171),'cluster_counts')
    need('比例' in report and '必要条件' in report and '名义' in report and '历史' in report and '未作多重比较校正' in report,'units_and_inference_scope')
    failure=obj('validation_resource_failure_independent_review.json')
    need(failure['completed_validation_records']==3616 and failure['cancelled_complete_output_lines_lower_bound']==3592 and failure['unlaunched_C10_cells']==8,'failure_exposure_counts')
    need(val['prior_four_semantic_comparison']['rows_compared']==3616 and val['prior12_mechanism_semantic_comparison']['complete_lines_compared']==12,'failed_semantic_consistency')
    need('20格完成验证' in report and '3592' in report and 'readiness仍为实际EXIT1' in report and '20项历史输入' in report and '本轮直接消费的85项输入' in report,'failure_historical_text')
    collection=obj('collected/test/manifest.json');supp=obj('test_results_source_supplement/artifact_map.json')
    need(len(collection['files'])==274 and len(collection['remote_before_sha256'])==322 and len(supp['files'])==61,'collection_counts')
    after={p:sha(p) for p in allpins};need(after==before,'all_input_output_end_SHA')
    result=dict(status='PASS',schema='cap10_final_report_readonly_checks_20261025',utc=datetime.now(timezone.utc).isoformat(),input_count=337,output_count=2,report_sha256=REPORT_SHA,manifest_sha256=MANIFEST_SHA,all_begin_end_sha256=before,checks=dict(all_manifest_SHA=True,all_links_registered=True,acyclic=True,actual_train_val_test_reviews_PASS=True,report_table_cells_six_decimal=True,metrics324_exact=True,paired180_all_direct_fields_exact=True,EMPTY_STRATUM20_preserved=True,preselection_and_budget=True,mechanism400_original87=True,current85_sources212_history20=True,exposure_failure_scope=True),original_prediction_reads=0,tensor_reads=0,scientific_runs=0,source_sha256=sha(Path(__file__).resolve()))
    out=ROOT/'final_report_readonly_checks.json';need(not out.exists(),'unique_check_receipt');out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(status='PASS',receipt=str(out),sha256=sha(out))))
if __name__=='__main__':main()
