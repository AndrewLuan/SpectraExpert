"""功能概述：在16臂完成后分离报告推断收益与联合训练导致的均值变化。

main对指定layout/seed，固定baseline预测bank和训练宽度，仅移植配对report臂的
共享513参数报告头，不重算或修改峰位。先保存控制协议，再评320-val；宽度沿用
baseline已冻结选择，随后评1000-test及371固定变体。此四个事后机制对照不参与
原16臂模型选择，也不根据DMF或test改头、概率、宽度及评分配置。
verify_unaffected核查所有非HMDB查询的rank和Top-10与同宽度baseline逐条相同，
从而检验推断干预的路由边界。输出独占report_transfer/<layout_seed>及真实差异统计；
只读既有权重/bank，不调用API或训练，不改原冻结选择和任何原始数据。
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from .h_condinfer import scoring_inputs
from .h_count_finalization import cluster_bootstrap, index
from .h_joint_conditions import ReportingHead
from .h_joint_evaluate import rank
from .h_metadata import metadata_view
from .prepare import sha


def verify_unaffected(current, previous):
    old = index(previous)
    if index(current).keys()!=old.keys():
        raise ValueError('report_transfer_query_identity_drift')
    checked = 0
    for row in current:
        if row['source']=='HMDB':continue
        other = old[row['source'],row['record_id'],row['smiles']]
        if row['rank']!=other['rank'] or row['top10']!=other['top10']:
            raise ValueError('report_transfer_changed_non_HMDB_rank')
        checked += 1
    return checked


def main():
    p = argparse.ArgumentParser()
    for name in ('root','base','layout','seed'):
        p.add_argument('--'+name,required=True)
    a = p.parse_args()
    root,base = Path(a.root),Path(a.base)
    study = root/'study'
    if (study/'EXIT_CODE').read_text().strip()!='0':raise ValueError('primary_study_not_complete')
    stem = a.layout+'_base_s'+a.seed
    donor = a.layout+'_report_s'+a.seed
    out = root/'report_transfer'/(a.layout+'_s'+a.seed)
    out.mkdir(parents=True,exist_ok=False)
    def status(stage,**kw):(out/'status.json').write_text(json.dumps(dict(stage=stage,**kw)))
    def rows(path):return [json.loads(line) for line in path.read_text().splitlines()]
    try:
        torch.set_num_threads(4)
        torch.set_float32_matmul_precision('high')
        device = torch.device('cuda')
        baseline = study/'evaluation'/stem
        lam = json.loads((baseline/'val/frozen_selection.json').read_text())['width_lambda']
        global_selection = study/'evaluation/frozen_selection.json'
        before = sha(global_selection)
        checkpoint = study/'training'/donor/'train/best.pt'
        protocol = dict(mean_model=stem,report_head=donor,width_lambda=lam,
            report_checkpoint_sha256=sha(checkpoint),original_selection_sha256=before,
            scope='post-selection fixed-means mechanism control; no fit, no model selection or promotion')
        (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
        bank = torch.load(baseline/'bank.pt',map_location='cpu',weights_only=False,mmap=True)
        prepared = base/'hydrogen_analysis/condition_response/solvent_resolved_prepared'
        metadata = metadata_view(torch.load(prepared/'metadata.pt',map_location='cpu',weights_only=False),bank['condition_key_mode'])
        labels = torch.load(base/'hydrogen_analysis/labels_topology/annotations.pt',map_location='cpu',weights_only=False)
        saved = torch.load(checkpoint,map_location='cpu',weights_only=False)
        head = ReportingHead().to(device).eval()
        head.load_state_dict({k.removeprefix('reporting_head.'):v for k,v in saved['state_dict'].items() if k.startswith('reporting_head.')})
        model = SimpleNamespace(reporting_enabled=True,reporting_head=head)
        widths = json.loads((baseline/'val/width_fit.json').read_text())['selected_ppm']
        inputs = scoring_inputs(base)
        val = out/'val'
        val.mkdir()
        validation = rank(bank,bank['val'],metadata,labels,model,widths,inputs,device,val,status)
        (val/'frozen_selection.json').write_text(json.dumps(dict(width_lambda=lam,selection='baseline width retained; no new selection',validation=validation),indent=2))
        test = out/'test'
        test.mkdir()
        tests = rank(bank,bank['test'],metadata,labels,model,widths,inputs,device,test,status,True)
        contrasts = {}
        for width in tests:
            current = rows(test/('width_'+width)/'predictions.jsonl')
            previous = rows(baseline/'test'/('width_'+width)/'predictions.jsonl')
            checked = verify_unaffected(current,previous)
            contrasts[width] = dict(cluster_difference=cluster_bootstrap(current,previous),
                non_HMDB_rank_and_top10_exact=checked,
                DMF_rank=next(r['rank'] for r in current if r['record_id']=='HMDB:HMDB0001888:5029'))
        if sha(global_selection)!=before:raise ValueError('report_transfer_modified_primary_selection')
        (out/'summary.json').write_text(json.dumps(dict(protocol=protocol,validation=validation,tests=tests,contrasts=contrasts,source_sha256=sha(__file__)),indent=2))
        status('complete')
        (out/'EXIT_CODE').write_text('0\n')
    except Exception as exc:
        status('failed',error=repr(exc))
        (out/'EXIT_CODE').write_text('1\n')
        raise


if __name__=='__main__':main()
