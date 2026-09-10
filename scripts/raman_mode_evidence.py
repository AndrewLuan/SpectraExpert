"""功能概述：以逐振动模式记录替代混合分子的 OR 规则，生成可溯源的 Raman 候选证据。
主要职责：MODES 保存官方表逐项区间、SMARTS 适用条件和 Raman 活性；retrieve 只匹配
实验峰并可按候选子结构过滤；format_evidence 提供完整与最多八条的紧凑格式。
代码库关系：由 run_raman_kb_ablation 调用，HORIBA RA TN-18 是实际核对来源；SMARTS
是本项目工程编码，不声称由文献给出。不使用标签、候选排名分数或缺峰排除规则。
输入输出与副作用：只读内存输入，无网络及文件写入；非法 SMILES/SMARTS 明确报错。
范围是经验支持区间，不能保证给定分子必有该峰；未记录的物相和仪器条件保持未知。
"""
from rdkit import Chem

SOURCE = 'https://static.horiba.com/fileadmin/Horiba/Products/Scientific/Molecular_and_Microanalysis/MacroRAM/RA-TN18-Raman_Data_and_Analysis.pdf'
CITE = 'horibaRATN18'
# id, vibration, lower, upper, applicability SMARTS, Raman intensity, page.
MODES = [
    ('RM01', 'C-Br stretch', 500, 700, '[#6]-[Br]', 'strong', 1),
    ('RM02', 'C-Cl stretch', 550, 800, '[#6]-[Cl]', 'strong', 1),
    ('RM03', 'C-O-C stretch', 800, 970, '[#6]-[O;X2]-[#6]', 'strong', 1),
    ('RM04', 'C=S stretch', 1000, 1250, '[#6]=[S]', 'strong', 2),
    ('RM05', 'C-NO2 stretch', 1340, 1380, '[#6]-[N+](=O)[O-]', 'strong', 2),
    ('RM06', 'C=N stretch', 1610, 1680, '[#6]=[#7]', 'strong', 2),
    ('RM07', 'C=C stretch', 1500, 1900, '[C]=[C]', 'strong', 2),
    ('RM08', 'C=O stretch', 1680, 1820, '[#6]=[O]', 'medium', 2),
    ('RM09', 'C#C stretch', 2100, 2250, '[#6]#[#6]', 'strong', 2),
    ('RM10', 'C#N stretch', 2220, 2255, '[#6]#[#7]', 'medium', 2),
    ('RM11', 'S-H stretch', 2550, 2600, '[S;H1]', 'strong', 2),
    ('RM12', 'saturated C-H stretch', 2800, 3000, '[C;X4;!H0]', 'strong', 2),
    ('RM13', 'unsaturated C-H stretch', 3000, 3100, '[#6;X3;!H0]', 'strong', 2),
    ('RM14', 'N-H stretch', 3300, 3500, '[#7;!H0]', 'medium', 2),
    ('RM15', 'O-H stretch', 3100, 3650, '[O;H1]', 'weak', 2),
]


def retrieve(positions, candidates, filtered=True, tolerance=20):
    molecules = [Chem.MolFromSmiles(s) for s in candidates]
    if any(m is None for m in molecules):
        raise ValueError('invalid candidate structure')
    hits = []
    for rid, mode, lo, hi, smarts, intensity, page in MODES:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            raise ValueError(smarts)
        matches = [float(p) for p in positions if lo-tolerance <= p <= hi+tolerance]
        if not matches:
            continue
        compatible = [i+1 for i, m in enumerate(molecules) if m.HasSubstructMatch(pattern)]
        if filtered and not compatible:
            continue
        distance = min(max(lo-p, p-hi, 0) for p in matches)
        hits.append(dict(rule_id=rid, mode=mode, interval=[lo, hi], peaks=matches,
            modality='Raman', activity=intensity, phase='unspecified',
            compatible_candidates=compatible if filtered else None,
            applicability_smarts=smarts, source_refs=[CITE], source_url=SOURCE,
            source_locator=f'RA TN-18, page {page}', distance_cm=distance))
    # Same ranking in both arms: no hidden candidate-discriminativeness change.
    return sorted(hits, key=lambda h:(h['distance_cm'], h['interval'][1]-h['interval'][0], h['rule_id']))


def format_evidence(hits, compact=False):
    chosen = hits[:8] if compact else hits
    lines = ['## Mode-specific Raman evidence',
        'These are possible assignments, not proof of identity. Missing bands are not exclusions. '
        'Phase is unspecified; scores are not probabilities. Cite relevant rule IDs and source keys.']
    for h in chosen:
        compatible = h['compatible_candidates']
        scope = ('; compatible candidate indices='+','.join(map(str,compatible))) if compatible is not None else ''
        line = (f"{h['rule_id']}: {h['mode']}; interval={h['interval']} cm^-1; "
                f"observed={h['peaks']}; Raman={h['activity']}{scope}; "
                f"source={CITE}, {h['source_locator']}.")
        lines.append(line)
        if not compact:
            lines.append(f"Applicability predicate: {h['applicability_smarts']}. "
                'Substructure compatibility is necessary context, not a predicted Raman spectrum. '
                'Environmental shifts, symmetry and overlapping modes can affect visibility. '
                f"Source URL: {SOURCE}")
    if not chosen:
        lines.append('No applicable mode record matched; use the original peaks and candidates.')
    return '\n'.join(lines)
