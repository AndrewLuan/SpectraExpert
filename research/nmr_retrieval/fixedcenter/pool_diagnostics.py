"""功能：对同一冻结排序器在single/dual自然池上的验证输出做配对分解。
读取两个已成功评估目录的summary.json、predictions.jsonl及EXIT_CODE；拒绝test/verify。
query_key核对完整查询身份；decompose按共同入池/新增/丢失/均未入池拆解MRR。
optional_pool检查输出中自然候选及costs，量化共同候选相对名次，避免将池中位数
归一化引起的评分变化误写成固定评分。main独占写summary和逐query配对结果，记录
输入始末SHA和真实内部EXIT；不加载谱、不调模型、不重算CI、不访问外部API。
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def query_key(row):
    return tuple(str(row[k]) for k in ('query_order','variant','source','record_id','group'))


def index_rows(rows):
    result = {}
    for row in rows:
        key = query_key(row)
        if key in result:
            raise ValueError('duplicate_query_identity')
        if row['rank'] < 1 or row['evaluation_pool_size'] != 512:
            raise ValueError('invalid_rank_or_pool')
        if bool(row['target_in_coarse_pool']) != (row['rank'] <= 512):
            raise ValueError('rank_membership_inconsistent')
        result[key] = row
    return result


def optional_pool(a, b):
    a, b = dict(a), dict(b)
    for row in (a, b):
        if 'natural_candidate_indices' in row: row['candidate_indices'] = row['natural_candidate_indices']
        if 'natural_candidate_costs' in row: row['candidate_costs'] = row['natural_candidate_costs']
    if 'candidate_indices' not in a or 'candidate_indices' not in b:
        return {'candidate_overlap_available': False}
    pa, pb = a['candidate_indices'], b['candidate_indices']
    if len(pa) != 512 or len(pb) != 512 or len(set(pa)) != 512 or len(set(pb)) != 512:
        raise ValueError('nonnatural_or_duplicate_pool')
    common = set(pa) & set(pb)
    out = dict(candidate_overlap_available=True, common_candidates=len(common),
               added_candidates=len(set(pb)-set(pa)), removed_candidates=len(set(pa)-set(pb)))
    if 'candidate_costs' in a and 'candidate_costs' in b:
        import math
        ca, cb = a['candidate_costs'], b['candidate_costs']
        if len(ca)!=512 or len(cb)!=512 or not all(math.isfinite(x) for x in ca+cb):
            raise ValueError('invalid_candidate_costs')
        va, vb = dict(zip(pa, ca)), dict(zip(pb, cb))
        ra = {i:n for n,i in enumerate(sorted(common,key=lambda i:(va[i],i)))}
        rb = {i:n for n,i in enumerate(sorted(common,key=lambda i:(vb[i],i)))}
        out['common_candidate_mean_abs_rank_change'] = sum(abs(ra[i]-rb[i]) for i in common)/max(1,len(common))
    return out


def decompose(left, right):
    a, b = index_rows(left), index_rows(right)
    if a.keys() != b.keys():
        raise ValueError('query_identity_mismatch')
    pairs=[]
    for key in sorted(a):
        x,y=a[key],b[key]
        for field in ('smiles','mask_content_sha256'):
            if field in x or field in y:
                if x.get(field)!=y.get(field): raise ValueError('query_content_mismatch:'+field)
        category = {(True,True):'common',(False,True):'new',(True,False):'lost',(False,False):'neither'}[(bool(x['target_in_coarse_pool']),bool(y['target_in_coarse_pool']))]
        pairs.append(dict(query_key=key, source=x['source'], variant=x['variant'], group=x['group'],
            category=category, left_rank=x['rank'], right_rank=y['rank'],
            delta_MRR=1/y['rank']-1/x['rank'], **optional_pool(x,y)))
    def summarize(part):
        n=len(part)
        if not n:return {'queries':0}
        cells={}
        for category in ('common','new','lost','neither'):
            cell=[r for r in part if r['category']==category]
            cells[category]=dict(queries=len(cell), contribution_to_total_delta_MRR=sum(r['delta_MRR'] for r in cell)/n,
                conditional_delta_MRR=sum(r['delta_MRR'] for r in cell)/len(cell) if cell else None)
        delta=sum(r['delta_MRR'] for r in part)/n
        assert abs(delta-sum(c['contribution_to_total_delta_MRR'] for c in cells.values()))<1e-12
        return dict(queries=n, left_MRR=sum(1/r['left_rank'] for r in part)/n,
            right_MRR=sum(1/r['right_rank'] for r in part)/n, delta_MRR=delta,
            delta_recall512=(cells['new']['queries']-cells['lost']['queries'])/n,
            **{'delta_top'+str(k):sum((r['right_rank']<=k)-(r['left_rank']<=k) for r in part)/n for k in (1,5,10)}, categories=cells)
    return pairs,dict(by_variant={v:summarize([r for r in pairs if r['variant']==v]) for v in sorted({r['variant'] for r in pairs})},
        by_source={s:summarize([r for r in pairs if r['source']==s and r['variant']=='observed']) for s in sorted({r['source'] for r in pairs})})


def read_run(path):
    p=Path(path)
    if (p/'EXIT_CODE').read_text().strip()!='0':raise ValueError('evaluation_not_complete')
    summary=json.loads((p/'summary.json').read_text())
    if summary.get('cohort') not in ('original_val','tune'):raise ValueError('validation_only')
    rows=[json.loads(x) for x in (p/'predictions.jsonl').read_text().splitlines() if x]
    return summary,rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--single',required=True);parser.add_argument('--dual',required=True)
    parser.add_argument('--out',required=True)
    args=parser.parse_args(); out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    paths=[Path(__file__)]+[Path(p)/n for p in (args.single,args.dual) for n in ('summary.json','predictions.jsonl','EXIT_CODE')]
    before={str(p.resolve()):sha(p) for p in paths}
    try:
        sa,a=read_run(args.single);sb,b=read_run(args.dual)
        for key in ('cohort','query_sha256','candidate_order_sha256','checkpoints'):
            if key not in sa or sa[key]!=sb.get(key):raise ValueError('fixed_model_or_identity_mismatch:'+key)
        pairs,result=decompose(a,b)
        after={str(p.resolve()):sha(p) for p in paths}
        if before!=after:raise ValueError('input_changed')
        result.update(cohort=sa['cohort'],scope='train_val_only',inputs_sha256=before,inputs_unchanged=True,
            external_api_calls=0,interpretation='Same frozen model; natural pool median centering can change common-candidate scores. No causal attribution to missing peaks or atom errors.')
        (out/'paired.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in pairs))
        (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
        (out/'EXIT_CODE').write_text('0\n')
    except Exception:
        (out/'EXIT_CODE').write_text('1\n');raise

if __name__=='__main__':main()
