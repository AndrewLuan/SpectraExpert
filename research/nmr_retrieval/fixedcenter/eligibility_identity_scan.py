"""功能概述：仅抽取JJ实验数据身份，核查新验收的候选支持。
main 读取prepared缓存身份并流式扫描experimental_pretrain；不访问value中的谱值、
不计算排名。identity沿用nmr_transfer.data，聚合原始SMILES后只规范化一次。
输出JSON含输入SHA、来源计数和未入缓存结构身份，但任何身份均不自动获得历史隔离资格。
本脚本由eligibility_launch.py经run_screened SSH输入执行，JJ不落盘。
"""
import sys,json,hashlib,collections,time,os
from pathlib import Path
from datetime import datetime,timezone
sys.path.insert(0,'/home/user/IRfingerprints/SpectrExpert')
import torch
from nmr_transfer.data import identity
torch.set_num_threads(1)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 start=datetime.now(timezone.utc).isoformat()
 p=Path('/home/user/IRfingerprints/SpectrExpert/outputs/nmr_transfer_20261017/prepared_coverage/cache.pt')
 cache=torch.load(p,map_location='cpu',weights_only=False,mmap=True)
 groups={x['group'] for x in cache['rows']}
 exp_test=[{k:x[k] for k in ['group','source','split','smiles']} for x in cache['rows'] if x['domain']=='experimental' and x['nucleus']=='1H' and x['split']=='test']
 known={x['smiles']:x['group'] for x in cache['rows']}
 counts=collections.Counter()
 raw=Path('/data/SpecGPT/dataset/NMR_exprimental/experimental_pretrain/data.jsonl')
 h=hashlib.sha256(); unique={};schemas=collections.Counter()
 with raw.open('rb') as f:
  for i,line in enumerate(f):
   h.update(line);x=json.loads(line)
   counts['all_records']+=1
   if x.get('nucleus')!='1H':continue
   counts['1H_records']+=1
   src=str(x.get('source'));counts['source:'+src]+=1
   smi=x.get('smiles')
   if not isinstance(smi,str):counts['missing_smiles']+=1;continue
   key=(smi,src)
   if key not in unique:
    unique[key]=[0,str(x.get('record_id')),str(x.get('topology_key')),x.get('provenance',{}).get('source_path')]
   unique[key][0]+=1
   schemas['|'.join(sorted(x))]+=1
   if i and i%500000==0:print(json.dumps({'progress_records':i,'unique_pairs':len(unique)}),file=sys.stderr,flush=True)
 unmatched={};bad=collections.Counter(); group_sources=collections.defaultdict(set)
 for (smi,src),v in unique.items():
  try:g=known.get(smi) or identity(smi)[1]
  except Exception as e:bad[type(e).__name__]+=v[0];continue
  group_sources[g].add(src)
  if g not in groups:
   if g not in unmatched:unmatched[g]={'group':g,'smiles':smi,'sources':[],'records':0,'example_record_id':v[1],'raw_topology_key':v[2],'source_path':v[3]}
   unmatched[g]['sources']=sorted(set(unmatched[g]['sources'])|{src});unmatched[g]['records']+=v[0]
 result={'started_utc':start,'ended_utc':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),'actual_exit_code':0,'input_sha256':{str(raw):h.hexdigest(),str(p):sha(p)},'counts':dict(counts),'unique_smiles_source_pairs':len(unique),'invalid_records':dict(bad),'schema_counts':dict(schemas),'raw_1H_structures':len(group_sources),'overlap_prepared_structures':len(set(group_sources)&groups),'not_in_prepared_structures':len(unmatched),'unmatched_identities':sorted(unmatched.values(),key=lambda x:x['group']),'native_experimental_test_records':len(exp_test),'native_experimental_test_structures':len({x['group'] for x in exp_test}),'native_experimental_test_identities':exp_test,'qualified_independent_groups':[],'eligibility':'not_cleared_history_audit_required','access_scope':'identity/source/schema only; no spectrum values or ranks emitted','external_api_calls':0}
 print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
