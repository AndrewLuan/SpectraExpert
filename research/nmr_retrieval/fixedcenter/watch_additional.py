"""功能概述：只读检查唯一固定中心训练队列，保留紧凑的真实状态和历史。
main通过既有screened wrapper读取JJ的driver、退出码、PID及训练history，不启动或重试job。
"""
import subprocess,sys,shlex,json
from pathlib import Path
D=Path(__file__).resolve().parent
REMOTE=r"""from pathlib import Path
import json,datetime,collections,os
r=Path('/home/user/IRfingerprints/SpectrExpert/outputs/nmr_pool_ranking_20261023/runs/additional_fixedcenter_20261023T1205Z')
s=json.loads((r/'status.json').read_text())
def readexit(name):
 p=r/name
 return int(p.read_text()) if p.exists() else None
out={'remote_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'driver_status_utc':s['utc'],'driver_pid':s['driver_pid'],'jobs':[],'job_state_counts':dict(collections.Counter(x['state'] for x in s['jobs'].values())),'internal_exit':readexit('EXIT_CODE'),'external_exit':readexit('EXTERNAL_EXIT_CODE')}
for jid,j in s['jobs'].items():
 x={k:j.get(k) for k in ('state','pid','gpu','started_utc','finished_utc','external_exit','internal_exit')}
 x['id']=jid
 if j.get('pid'):
  try:os.kill(j['pid'],0);x['pid_alive']=True
  except ProcessLookupError:x['pid_alive']=False
 if jid.startswith('fit_'):
  o=Path(j['output'])
  for n in ('centers.json','center_rows.jsonl','initial.pt','history.jsonl'):
   p=o/n;x[n+'_exists']=p.exists();x[n+'_bytes']=p.stat().st_size if p.exists() else 0
  p=o/'history.jsonl'
  if p.exists():
   rows=[json.loads(z) for z in p.read_text().splitlines() if z.strip()]
   if rows:x['last_history']={k:rows[-1].get(k) for k in ('epoch','best','seconds')}
 out['jobs'].append(x)
print(json.dumps(out))
"""
a=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','JJ',shlex.join(['/home/user/.venvs/spectrexpert/bin/python','-c',REMOTE])]
r=subprocess.run([sys.executable,str(D/'run_screened.py'),'--']+a,capture_output=True,text=True)
if r.returncode:print(r.stdout);print(r.stderr,file=sys.stderr);sys.exit(r.returncode)
x=json.loads(r.stdout);(D/'additional_fixedcenter_watch_snapshot.json').write_text(json.dumps(x,indent=2)+'\n')
with (D/'additional_fixedcenter_watch_history.jsonl').open('a') as f:f.write(json.dumps(x)+'\n')
state=json.loads((D/'orchestrator-state.json').read_text());a=state['active_runs']['additional_fixedcenter'];a.update(counts=x['job_state_counts'],actual_snapshot_utc=x['remote_utc'])
if x['internal_exit'] is not None or x['external_exit'] is not None:a.update(internal_exit=x['internal_exit'],external_exit=x['external_exit'])
(D/'orchestrator-state.json').write_text(json.dumps(state,indent=2)+'\n')
print(json.dumps(x,indent=2))
