"""功能概述：用冻结重排器评估封存验证/测试，避免 alpha=0 重复计数。

main 复用训练时的 ResidualRanker、标准化、排名与指标实现；候选、特征协议、
预测检查点、重排权重、所有输入及本入口源码均与冻结文件核对。只接受 verify/test
缓存，且 alpha 直接读取已选择权重；alpha=0 控制和所选 alpha 去重后各算一次。
这是评估入口的计数修正，不更改训练算法、模型、特征缓存或冻结选择信号。
每份输出检查查询/变体身份唯一、计数相等，保留原始 alpha=0 完整库排名。
不搜索参数，不改写旧结果或数据，不访问任何网络或外部 API。
"""
import argparse
import json
from pathlib import Path

import torch

from .h_global_ranker import ResidualRanker, evaluate, normalized_rows
from .prepare import sha


def main():
    parser=argparse.ArgumentParser()
    for name in ('checkpoint','cache','frozen-selection','out'):
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    try:
        frozen=json.loads(Path(args.frozen_selection).read_text())
        selection_sha256=sha(args.frozen_selection);cache_sha256=sha(args.cache)
        if frozen.get('selection_scope')!='train_val_only':raise ValueError('locked_ranker_invalid_selection')
        source=Path(__file__).resolve()
        if frozen['evaluation_inputs_sha256'].get(str(source))!=sha(source):raise ValueError('locked_ranker_source_changed')
        checkpoint=Path(args.checkpoint).resolve()
        if frozen['ranker_checkpoints_sha256'].get(str(checkpoint))!=sha(checkpoint):raise ValueError('locked_ranker_weight_changed')
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        data=torch.load(args.cache,map_location='cpu',weights_only=False);manifest=data['manifest']
        if manifest['cohort'] not in ('verify','test'):raise ValueError('locked_ranker_wrong_cohort')
        for path,digest in manifest['source_inputs_sha256'].items():
            if frozen['evaluation_inputs_sha256'].get(path)!=digest or sha(path)!=digest:
                raise ValueError('locked_ranker_feature_input_changed:'+path)
        for field in ('coarse_config','pool_size','feature_code_sha256','feature_names','candidate_order_sha256'):
            if saved[field]!=manifest[field]:raise ValueError('locked_ranker_protocol_changed:'+field)
        if saved['prediction_checkpoint_sha256']!=manifest['checkpoint_sha256']:
            raise ValueError('locked_ranker_prediction_checkpoint_changed')
        identities=[(r['query_order'],r['variant']) for r in data['rows']]
        if len(set(identities))!=len(identities):raise ValueError('locked_ranker_duplicate_cached_query')
        if len(identities)!=manifest['queries']:raise ValueError('locked_ranker_cache_count_changed')
        torch.set_num_threads(4);device=torch.device('cuda')
        model=ResidualRanker(saved['columns'],saved['architecture'],saved['typed']).to(device)
        model.load_state_dict(saved['state_dict'])
        alphas=list(dict.fromkeys([0.,saved['alpha']]))
        results,reports=evaluate(model,normalized_rows(data['rows'],saved['scale']),device,alphas)
        for alpha,rows in results.items():
            if len(rows)!=manifest['queries']:raise ValueError('locked_ranker_output_count_changed')
            for row in rows:row['top10']=[data['smiles'][i] for i in row.pop('top10_indices')]
            (out/('predictions_'+alpha+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))
        for path,digest in manifest['source_inputs_sha256'].items():
            if sha(path)!=digest:raise ValueError('locked_ranker_input_changed_during_evaluation:'+path)
        if sha(args.frozen_selection)!=selection_sha256 or sha(args.cache)!=cache_sha256:
            raise ValueError('locked_ranker_selection_or_cache_changed_during_evaluation')
        if sha(checkpoint)!=frozen['ranker_checkpoints_sha256'][str(checkpoint)] or sha(source)!=frozen['evaluation_inputs_sha256'][str(source)]:
            raise ValueError('locked_ranker_weight_or_source_changed_during_evaluation')
        summary=dict(metrics=reports,alpha=saved['alpha'],checkpoint_sha256=sha(checkpoint),
            cache_sha256=cache_sha256,queries=manifest['queries'],source_sha256=sha(source),
            frozen_selection_sha256=selection_sha256,selection_scope='frozen internal train alpha',external_api_calls=0)
        (out/'summary.json').write_text(json.dumps(summary,indent=2));(out/'EXIT_CODE').write_text('0\n')
    except Exception as exc:
        (out/'failure.json').write_text(json.dumps(dict(error=repr(exc))));(out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':
    main()
