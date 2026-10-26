"""功能概述：补齐祖先初始化链，并核对固定候选库的历史实验测试曝光。
main 仅读配置、权重元数据、缓存身份和历史输出身份；不读取或输出谱值与排名。
证据SHA随读取记录，末尾复核；只有身份交集统计输出。固定候选全被暴露时，
独立验收清单保持空，不因找到新原始记录而改变候选库。
"""
from pathlib import Path
import sys,json,hashlib,os,time,collections
from datetime import datetime,timezone
sys.path.insert(0,'/home/user/IRfingerprints/SpectrExpert')
import torch
from nmr_transfer.data import identity
torch.set_num_threads(1)
R=Path('/home/user/IRfingerprints/SpectrExpert');B=R/'outputs/nmr_transfer_20261017';C=B/'hydrogen_analysis/condition_response'
evidence={}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def pin(p):
 evidence[str(p)]=sha(p);return evidence[str(p)]
def read(p):
 pin(p);return json.loads(p.read_text())
def main():
 started=datetime.now(timezone.utc).isoformat();chain=[];m=read(C/'solvent_resolved_ablation/training/metadata_only/train/config.json')
 rec=read(C/'solvent_resolved_ablation/training/recovered/train/config.json')
 p=Path(rec['args']['init']);assert pin(p)==m['init_checkpoint_sha256']==rec['init_sha256']
 seen=set()
 while str(p) not in seen:
  seen.add(str(p));cp=p.parent/'config.json';entry={'checkpoint':str(p),'sha256':pin(p)}
  if cp.exists():
   cfg=read(cp);entry.update(config_path=str(cp),config_sha256=evidence[str(cp)],args=cfg.get('args'),selection=cfg.get('selection'),test_used=cfg.get('test_used'))
   init=cfg.get('args',{}).get('init')
   if not init and cfg.get('baseline_sha256'):
    init=str(B/'hydrogen_analysis/site_supervision/train/best.pt')
    assert sha(init)==cfg['baseline_sha256']
    entry['implicit_init_source']='nmr_transfer/h_condtrain.py baseline path, checkpoint SHA matched'
    pin(R/'nmr_transfer/h_condtrain.py')
   if init:
    ip=Path(init);ip=ip if ip.is_absolute() else R/ip
    if cfg.get('init_sha256'):assert sha(ip)==cfg['init_sha256']
   else:ip=None
  else:
   assert p==C/'gap_calibration/gap_unshrunk/train/best.pt'
   fit=read(C/'gap_calibration/fit.json');ip=Path(fit['reference_checkpoint']);assert sha(ip)==fit['reference_sha256']
   payload=torch.load(p,map_location='cpu',weights_only=False);assert payload['calibration']['reference_checkpoint_sha256']==fit['reference_sha256']
   entry['calibration']={k:payload['calibration'][k] for k in ['kind','reference_checkpoint_sha256']}
  chain.append(entry)
  if ip is None:break
  p=ip
 print('lineage resolved',file=sys.stderr,flush=True)
 cache_path=B/'prepared_coverage/cache.pt';pin(cache_path);cache=torch.load(cache_path,map_location='cpu',weights_only=False,mmap=True)
 rows=cache['rows'];bykey={(x['source'],x['record_id']):x['group'] for x in rows if x['domain']=='experimental' and x['nucleus']=='1H' and x['split']=='test'}
 exptrainval={x['group'] for x in rows if x['domain']=='experimental' and x['nucleus']=='1H' and x['split'] in ['train','val']}
 alltrainval={x['group'] for x in rows if x['split'] in ['train','val']}
 history=[]
 used=set()
 for p in sorted((B/'1H/evaluation').glob('*_forward.jsonl')):
  pin(p);ids=set()
  for line in p.open():
   x=json.loads(line);ids.add((x['source'],x['record_id']))
  matched=ids&set(bykey);used|={bykey[k] for k in matched}
  history.append({'path':str(p),'sha256':evidence[str(p)],'records':len(ids),'native_test_exact_identity_matches':len(matched),'native_test_missing':len(set(bykey)-ids)})
 bankpath=R/'outputs/nmr_joint_conditions_20261020/study/evaluation/local_base_s211/bank.pt';pin(bankpath)
 bank=torch.load(bankpath,map_location='cpu',weights_only=False,mmap=True);smiles=[bank['smiles'][int(i)] for i in bank['single_indices']];assert len(smiles)==114982;assert hashlib.sha256(chr(10).join(smiles).encode()).hexdigest()=='72b72140c09ae14ae8d0aa4edad624a151093c9df246b2052de8a3a4ea616bad';lookup={x['smiles']:x['group'] for x in rows}
 cg={lookup[s] if s in lookup else identity(s)[1] for s in smiles}
 uncovered=cg-(alltrainval|used)
 result={'started_utc':started,'ended_utc':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),'actual_exit_code':0,'lineage':chain,'lineage_reaches_random_initialization':chain[-1]['args'].get('init') is None,'historical_test_identity_checks':history,'native_test_records':len(bykey),'native_test_structures':len(set(bykey.values())),'historical_test_structures':len(used),'candidate_count':len(smiles),'candidate_order_sha256':hashlib.sha256(chr(10).join(smiles).encode()).hexdigest(),'candidate_connectivity_groups':len(cg),'candidate_groups_in_ancestor_trainval':len(cg&alltrainval),'candidate_groups_in_historical_test':len(cg&used),'candidate_groups_without_exposure_evidence':len(uncovered),'uncovered_groups':sorted(uncovered),'qualified_independent_groups':[],'independent_verify':'unavailable' if not uncovered else 'requires_further_history','input_sha256':evidence,'scope':'identity and source fields only; historical errors and ranks ignored','external_api_calls':0}
 for p,h in evidence.items():assert sha(p)==h,p
 result['input_sha_start_end_equal']=True
 print(json.dumps(result))
if __name__=='__main__':main()
