"""功能概述：将本轮已独审的Python工具按source-only流程同步到JJ并核验合成测试。
main核验review与本机字节，依次dryrun、rsync、远端SHA及unittest；run把全部子命令
交run_screened.py审查并保留真实退出/日志。本工具不启动科学job、不修改旧源码或数据。
"""
import argparse,datetime,hashlib,json,re,shlex,subprocess,sys
from pathlib import Path
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser()
 p.add_argument('--review',required=True);p.add_argument('--files',nargs='+',required=True)
 p.add_argument('--tests',nargs='+',required=True);p.add_argument('--expected-tests',type=int,required=True)
 p.add_argument('--receipt',required=True);a=p.parse_args()
 root=Path(__file__).resolve().parent;reviewpath=Path(a.review).resolve();review=json.loads(reviewpath.read_text())
 if review['status']!='PASS':raise ValueError('independent_review_not_PASS')
 reviewed=review['reviewed_sha256'];files=[(root/f).resolve() for f in a.files]
 for f in files:
  if f.parent!=root or f.suffix!='.py' or reviewed.get(str(f))!=digest(f):raise ValueError('reviewed_source_mismatch:'+str(f))
 nr='/home/user/IRfingerprints/SpectrExpert/outputs/nmr_pool_ranking_20261023'
 local_source=Path('/Users/andrewluan/Desktop/Projects/IRfingerprints/SpectrExpert')
 dependencies={}
 for f,h in reviewed.items():
  f=Path(f)
  if f.suffix!='.py':continue
  if f.parent==root:remote=nr+'/'+f.name
  elif f.is_relative_to(local_source):remote='/home/user/IRfingerprints/SpectrExpert/'+str(f.relative_to(local_source))
  else:continue
  if digest(f)!=h:raise ValueError('reviewed_dependency_drift:'+str(f))
  dependencies[remote]=h
 now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
 receipt={'started_utc':now(),'review':str(reviewpath),'review_sha256':digest(reviewpath),'steps':[],
          'actual_remote_SHA_expected':dependencies,'scientific_execution':False}
 out=Path(a.receipt).resolve()
 if out.exists():raise ValueError('exclusive_receipt_required')
 def save():out.write_text(json.dumps(receipt,indent=2)+'\n')
 def run(argv):
  start=now();r=subprocess.run([sys.executable,str(root/'run_screened.py'),'--']+argv,capture_output=True,text=True)
  receipt['steps'].append({'started_utc':start,'finished_utc':now(),'argv':argv,'actual_exit':r.returncode,'stdout':r.stdout,'stderr':r.stderr});save()
  if r.returncode:raise RuntimeError('screened_step_failed:'+str(r.returncode)+':'+r.stderr[-500:])
  return r
 try:
  options=['--itemize-changes','--timeout=30','-e','ssh -o BatchMode=yes -o ConnectTimeout=10']
  for mode in ('-azn','-az'):run(['rsync',mode]+options+[str(f) for f in files]+['JJ:'+nr+'/'])
  script="from pathlib import Path;import hashlib,json;x="+repr(dependencies)+";actual={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in x};assert actual==x;print(json.dumps({'status':'PASS','actual_sha256':actual}))"
  run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','JJ',shlex.join(['/home/user/.venvs/spectrexpert/bin/python','-c',script])])
  command='cd '+shlex.quote(nr)+' && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/user/IRfingerprints/SpectrExpert '+shlex.join(['/home/user/.venvs/spectrexpert/bin/python','-m','unittest','-v']+a.tests)
  result=run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','JJ',command])
  counts=re.findall(r'Ran (\d+) tests?',result.stdout+'\n'+result.stderr)
  if len(counts)!=1 or int(counts[0])!=a.expected_tests:raise ValueError('actual_synthetic_test_count_mismatch:'+repr(counts))
  receipt.update(status='PASS',JJ_synthetic_tests=int(counts[0]),finished_utc=now());save()
  print(json.dumps({'status':'PASS','JJ_synthetic_tests':int(counts[0]),'receipt':str(out)}))
 except BaseException as e:
  receipt.update(status='FAIL',failure=repr(e),finished_utc=now());save();raise
if __name__=='__main__':main()
