"""功能概述：定义24小时检索研究的排序、冻结输入和完成证据契约。

rank_shortlist 将候选短名单整体置于全库前部，再按冻结粗分保留名单外顺序；
该定义支持双通道非连续候选，不沿用“名单外排名不变”的单通道假设。训练补入
的正例只用于损失，评估时剔除。相等分数使用保守名次，Top10按候选序号稳定排序。
input_fingerprints 与 check_locked 核验实际读取文件及本轮阶段/参数；atomic_json
发布完整状态文件。所有函数不访问网络、不修改原始科学输入；调用者负责独占输出。
"""
import hashlib
import json
from pathlib import Path

import torch


def input_fingerprints(paths):
    """返回规范路径与文件SHA，不静默跳过不存在的输入。"""
    result = {}
    for path in sorted({Path(p).resolve() for p in paths if p is not None}):
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 << 20), b''):
                digest.update(chunk)
        result[str(path)] = digest.hexdigest()
    return result


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def check_locked(selection_path, inputs, config, cohort):
    """核验新轮封存；不能以旧轮freeze替代本轮配置或输入绑定。"""
    if cohort not in ('verify', 'test'):
        return None
    if not selection_path:
        raise ValueError('research_locked_selection_required')
    selection = json.loads(Path(selection_path).read_text())
    if (selection.get('schema') != 'nmr_research_20261022'
            or selection.get('selection_scope') != 'train_val_only'
            or not selection.get('frozen_utc')):
        raise ValueError('research_invalid_frozen_selection')
    if config not in selection['allowed_extract_configs']:
        raise ValueError('research_extract_config_not_frozen')
    pinned = selection['evaluation_inputs_sha256']
    for path, digest in inputs.items():
        if pinned.get(path) != digest:
            raise ValueError('research_frozen_input_changed:' + path)
    if cohort == 'verify' and selection.get('verify_eligibility') != 'eligible':
        raise ValueError('research_independent_verify_not_eligible')
    return selection


def pool_background_rank(coarse, pool, target):
    """仅在特征生成后计算标签指标；pool不必是coarse前缀。"""
    pool = torch.as_tensor(pool, device=coarse.device, dtype=torch.long)
    if len(pool.unique()) != len(pool):
        raise ValueError('research_duplicate_candidate_pool')
    if bool((pool == target).any()):
        return None
    before = int((coarse <= coarse[target] + 1e-7).sum())
    selected_before = int((coarse[pool] <= coarse[target] + 1e-7).sum())
    return len(pool) + before - selected_before


def rank_shortlist(row, costs):
    """以同一完整排序定义计算名次和Top10，拒绝训练正例越过召回上限。"""
    indices = torch.as_tensor(row['candidate_indices'], device=costs.device)
    size = int(row.get('evaluation_pool_size', len(indices)))
    if size <= 0 or size > len(indices) or len(indices[:size].unique()) != size:
        raise ValueError('research_invalid_evaluation_pool')
    values = costs[:size]
    if not torch.isfinite(values).all():
        raise ValueError('research_nonfinite_ranking_cost')
    # 首先按候选库序号排列，以稳定排序处理分数ties。
    by_index = torch.argsort(indices[:size], stable=True)
    order = by_index[torch.argsort(values[by_index], stable=True)]
    top = indices[order[:10]].detach().cpu().tolist()
    if row['target_in_coarse_pool']:
        target = int(row['target_position'])
        if not 0 <= target < size:
            raise ValueError('research_recalled_target_position_invalid')
        rank = int((values <= values[target] + 1e-7).sum())
    else:
        rank = int(row['outside_pool_rank'])
        if rank <= size:
            raise ValueError('research_outside_target_crossed_recall_bound')
    return rank, top


def summarize_predictions(rows):
    """统计检索、池内表现和来源，不将seed均值解释为预测集成。"""
    def metrics(part):
        n = len(part)
        if not n:
            return {'queries': 0}
        recalled = [r for r in part if r['target_in_coarse_pool']]
        return dict(queries=n, **{f'top{k}': sum(r['rank'] <= k for r in part)/n for k in (1, 5, 10)},
                    MRR=sum(1/r['rank'] for r in part)/n,
                    recall512=len(recalled)/n,
                    in_pool_MRR=sum(1/r['rank'] for r in recalled)/len(recalled) if recalled else None)
    normal = [r for r in rows if r['variant'] == 'observed']
    return dict(overall=metrics(normal),
                by_source={s: metrics([r for r in normal if r['source'] == s]) for s in sorted({r['source'] for r in normal})},
                stress={v: metrics([r for r in rows if r['variant'] == v]) for v in ('frequency_crop', 'peak_dropout')})
