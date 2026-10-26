"""功能概述：在原观测多重集内判断甲基归属能否靠置换修复及哪些弱监督可识别。

methyl_constraints从已有source_mol原子顺序提取普通无同位素CH3中已报告的2/3个H，
不补缺峰、不改变mask，不把CH2、NH2、跨母节点图对称合并为等价氢。
allocation_support用前后向整数动态规划，检查每个位移值可承载多少个三H/二H甲基组；
其他观测原子可自由保留剩余值。仅当可行分配支持同一值时报告被唯一确定的母组目标，
也可识别多个母节点间唯一的无序目标集合；二者均不是全原子硬标签修复。
identify把源记录与严格条件化结果连接：假设单态快转甲基的各已报告H具有完全相同的
中心位移，固定原观测mask及整谱值multiset。无解只表示该表在这些假设下不能由重排
修复，不证明计算来源或标注人为错误；静态构象、多态/多重峰语义另由逐例审核判断。
输入仅实验train/val源记录；依赖h_assignment_audit的形式契约及data.identity划分。
不读取理论/模型分数，不修改原始文件，不接网络或API；结果供审核侧表显式引用。
"""
import collections

from rdkit import Chem

from .data import identity
from .h_assignment_audit import inspect_record, spectrum_multiset


def methyl_constraints(record):
    """返回源索引母组及观测大小；仅使用结构和原观测mask。"""
    m = Chem.AddHs(Chem.MolFromSmiles(record['smiles']))
    order = record['atom_order']['source_to_canonical']
    reverse = {node: i for i, node in enumerate(order)}
    groups = []
    for a in m.GetAtoms():
        hs = [x for x in a.GetNeighbors() if x.GetAtomicNum() == 1]
        if (a.GetAtomicNum() != 6 or a.GetFormalCharge() or a.GetNumRadicalElectrons()
                or len(hs) != 3 or any(h.GetIsotope() not in (0, 1) for h in hs)):
            continue
        source_h = sorted(reverse[h.GetIdx()] for h in hs)
        observed = [i for i in source_h if record['atom_target_mask'][i]]
        if len(observed) >= 2:
            groups.append(dict(source_parent=reverse[a.GetIdx()], source_H=source_h,
                observed_source_H=observed, observed_size=len(observed),
                current_ppm=[record['atom_targets'][i] for i in observed]))
    return groups


def allocation_support(counts, n3, n2):
    """求3x+2y<=原值计数的所有全局可行分配支持，避免枚举逐H排列。"""
    if n3 < 0 or n2 < 0 or any(c <= 0 for c in counts.values()):
        raise ValueError('invalid_assignment_capacity')
    minimum = 2 if n2 else 3
    items = sorted((float(v), int(c)) for v, c in counts.items() if c >= minimum)
    options = [[(x,y) for x in range(min(n3,c//3)+1)
                for y in range(min(n2,(c-3*x)//2)+1)] for _, c in items]
    forward = [{(0,0)}]
    for choices in options:
        forward.append({(a+x,b+y) for a,b in forward[-1] for x,y in choices
                        if a+x <= n3 and b+y <= n2})
    if (n3,n2) not in forward[-1]:
        return dict(feasible=False, requested_3H_groups=n3, requested_2H_groups=n2,
                    value_support=[], group_size_support={3:[],2:[]})
    backward = [set() for _ in range(len(items)+1)]
    backward[-1] = {(0,0)}
    for i in range(len(items)-1,-1,-1):
        backward[i] = {(a+x,b+y) for a,b in backward[i+1] for x,y in options[i]
                       if a+x <= n3 and b+y <= n2}
    support = []; by_size = {3:[],2:[]}
    for i, (value,count) in enumerate(items):
        supported = {(x,y) for x,y in options[i] if any(
            (n3-a-x,n2-b-y) in backward[i+1] for a,b in forward[i])}
        xs = [x for x,_ in supported]; ys = [y for _,y in supported]
        support.append(dict(ppm=value, available_observations=count,
            group3_min=min(xs), group3_max=max(xs),
            group2_min=min(ys), group2_max=max(ys)))
        if max(xs): by_size[3].append(value)
        if max(ys): by_size[2].append(value)
    return dict(feasible=True, requested_3H_groups=n3, requested_2H_groups=n2,
                value_support=support, group_size_support=by_size)


def identify(record):
    """只输出条件化弱监督及不可置换修复证据，不生成硬原子标签更正。"""
    if (record.get('data_nature') != 'experimental' or record.get('nucleus') != '1H'
            or identity(record['smiles'])[2] not in ('train','val')):
        raise ValueError('identifiability_requires_experimental_train_val_H')
    audit = inspect_record(record)
    if audit['formal_issues']:
        raise ValueError('unverified_source_contract')
    groups = methyl_constraints(record)
    n = collections.Counter(g['observed_size'] for g in groups)
    counts = collections.Counter(spectrum_multiset(record))
    solution = allocation_support(counts,n[3],n[2])
    result = dict(record_id=record['record_id'], source=record['source_ref'], split=audit['split'],
        source_database_id=record.get('source_database_id'),
        assumptions=['fixed source observation mask', 'unchanged exact reported shift multiset',
            'one exactly shared center for each observed ordinary fast-rotating CH3',
            'all remaining observed atoms unconstrained'],
        value_histogram=[dict(ppm=v,count=c) for v,c in sorted(counts.items())],
        constraints=groups, allocation=solution, identified_parent_targets=[],
        pooled_group_targets=[], source_values_modified=False, missing_labels_imputed=False,
        theoretical_values_used=False, use='conditional_weak_supervision_candidate_only')
    if not groups:
        result['status']='no_nontrivial_methyl_constraint'
    elif not solution['feasible']:
        result['status']='no_permutation_solution_under_declared_semantics'
    else:
        result['status']='feasible_assignment_ambiguity'
        for g in groups:
            candidates = solution['group_size_support'][g['observed_size']]
            if len(candidates) == 1:
                result['identified_parent_targets'].append(dict(source_parent=g['source_parent'],
                    observed_source_H=g['observed_source_H'], shared_ppm=candidates[0]))
        for size,tag in [(3,'group3'),(2,'group2')]:
            if not n[size]:continue
            all_values=solution['value_support']
            if all(v[tag+'_min']==v[tag+'_max'] for v in all_values):
                result['pooled_group_targets'].append(dict(observed_group_size=size,
                    source_parents=[g['source_parent'] for g in groups if g['observed_size']==size],
                    group_center_multiset=[v['ppm'] for v in all_values for _ in range(v[tag+'_min'])],
                    interpretation='assignment ambiguity; not physical equivalence across parents'))
        if len(result['identified_parent_targets']) == len(groups):
            result['status']='methyl_targets_identified_conditionally'
    return result
