"""功能概述：恢复 Agent 已明确给出的最终候选，统一格式而不改变分子选择。
主要职责：extract 只接受显式最终字段或最后一行独立答案，映射 1..10 索引或唯一候选
canonical SMILES；冲突、越界和歧义返回 None。prepare 创建独立归一化实验并保留审计。
代码库关系：输入为 run_agent_multimodel 产物；输出使用原评分器的 SELECTED_INDEX。
不读取真值参与解析；只使用原始回答与候选列表。解析规则与模型无关，tag 仅选实验目录。
输入输出与副作用：只读父目录，原始响应不覆盖；输出记录响应哈希、证据文本与解析原因。
无法确定的响应保持原样供审查，绝不从正文全部候选中猜测或按真值回退。不调用 API。
"""
import argparse
import copy
from pathlib import Path
import re
import shutil
import sys

from rdkit import Chem, rdBase

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import run_agent_multimodel as base


def canonical(value):
    with rdBase.BlockLogs():
        try:mol=Chem.MolFromSmiles(value)
        except (ValueError,TypeError):return None
    return Chem.MolToSmiles(mol) if mol else None


def parse_value(value,candidates):
    value=value.strip()
    match=re.match(r'^(?:Candidate\s+)?(\d+)(?=$|[\s.:(])',value,re.I)
    if match:
        number=int(match[1])
        if not 1<=number<=len(candidates):return None
        # 若索引后显式附带合法 SMILES，要求它与该索引一致。
        remainder=value[match.end():].strip().lstrip('.:').strip()
        if remainder:
            token=remainder.split()[0]
            normalized=canonical(token)
            if normalized is None and token.startswith('(') and token.endswith(')'):
                normalized=canonical(token[1:-1])
            if normalized and normalized!=canonical(candidates[number-1]):return None
        return number-1
    value=re.sub(r'^SMILES\s*:\s*','',value,flags=re.I)
    normalized=canonical(value.strip('"\''))
    if not normalized:return None
    matches=[i for i,c in enumerate(candidates) if canonical(c)==normalized]
    return matches[0] if len(matches)==1 else None


def extract(text,candidates):
    clean=text.replace('**','').replace('`','')
    lines=clean.strip().splitlines()
    evidence=[]
    for i,line in enumerate(lines):
        match=re.match(r'^\s*(?:#{1,6}\s*)?(?:SELECTED_INDEX|Selected Candidate|Select Candidate|Select|Final Answer|Final Selection|The selected candidate is|The correct candidate is)\s*:\s*(.*?)\s*$',line,re.I)
        if match:
            value=match[1]
            if not value:value=next((s.strip() for s in lines[i+1:] if s.strip()),'')
            evidence.append(value)
    if evidence:
        indices=[parse_value(v,candidates) for v in evidence]
        if None in indices or len(set(indices))!=1:return None,'ambiguous_or_invalid_field',evidence
        return indices[0],'explicit_final_field',evidence
    if not lines:return None,'empty',[]
    last=lines[-1].strip()
    terminal=re.fullmatch(r'Candidate\s*:\s*(.+)',last,re.I)
    if terminal:
        index=parse_value(terminal[1],candidates)
        return index,'terminal_candidate_field',[last]
    # 最后一行“序号 + SMILES”必须同时命中同一个候选，避免把正文列表项当成猜测。
    terminal=re.fullmatch(r'(?:Candidate\s+)?(\d+)\s*[:.]\s*(\S+)',last,re.I)
    if terminal:
        index=parse_value(last,candidates)
        molecule=canonical(terminal[2])
        if index is not None and molecule and molecule==canonical(candidates[index]):
            return index,'terminal_index_and_smiles',[last]
    if re.fullmatch(r'(?:Candidate\s+)?\d+[.]?',last,re.I) or canonical(last):
        index=parse_value(last,candidates)
        if index is not None:return index,'standalone_final_line',[last]
    return None,'no_unambiguous_final_field',[]


def prepare(parent,out,tag):
    protocol=base.load(parent);tasks=base.read(parent/'tasks.json')
    models=[m for m in protocol['config']['models'] if m['tag']==tag]
    if len(models)!=1:raise ValueError('missing/ambiguous model')
    out.mkdir(parents=True,exist_ok=False)
    for sub in ['requests','responses','attempts']:(out/tag/sub).mkdir(parents=True)
    for name in ['queries.json','tasks.json']:shutil.copy2(parent/name,out/name)
    audit=[]
    for i,task in enumerate(tasks):
        source=parent/tag/'responses'/f'{i:04d}.json';r=base.read(source)
        index,reason,evidence=extract(r.get('content',''),task['candidates']) if not r.get('error') else (None,'api_error',[])
        if index is not None:
            r=dict(r,content=f'SELECTED_INDEX: {index+1}',normalization='explicit final answer extraction')
        base.save(out/tag/'responses'/source.name,r)
        shutil.copy2(parent/tag/'requests'/source.name,out/tag/'requests'/source.name)
        audit.append(dict(query_index=i,index=index,reason=reason,evidence=evidence,source_response=str(source.resolve()),source_sha256=base.digest(source)))
    base.save(out/'normalization_audit.json',audit)
    child=copy.deepcopy(protocol);child['config']['models']=models
    child['normalization']=dict(parent=str(parent.resolve()),parser_sha256=base.digest(Path(__file__)),
        policy='model-independent explicit final field / standalone last-line; candidate-only mapping; conflicts rejected; no ground truth')
    child['files_sha256']={str(p.relative_to(out)):base.digest(p) for p in out.rglob('*.json')}
    base.save(out/'protocol.json',child)
    print(dict(normalized=sum(r['index'] is not None for r in audit),unresolved=[r['query_index'] for r in audit if r['index'] is None]))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--tag',required=True)
    a=p.parse_args();prepare(a.parent,a.out,a.tag)
