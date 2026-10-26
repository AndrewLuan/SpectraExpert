"""功能概述：独立结果审查前，只补复制当前已封存q.source缺少的小文件。
来源是已闭包test metadata_view中的实际queue；逐文件SHA必须等q登记，始末真实SSH重核。
只准py/json/txt/md、ranker history/center_rows JSONL和EXIT_CODE；拒tensor、预测、旧报告。
不读取预测/性能值，不hash85大输入，不启动模型或科学任务。原map和原远端文件不修改。
下载到唯一新目录，真实before SSH/rsync/after SSH退出码各自保留；失败也保存receipt。
"""
from pathlib import Path
import hashlib,json,subprocess,shlex,datetime,sys
HERE=Path(__file__).resolve().parent
ROOT='/home/user/IRfingerprints/SpectrExpert/outputs/nmr_cap10_controlled_20261024T163633Z'
PY='/home/user/.venvs/spectrexpert/bin/python'
def digest(raw):return hashlib.sha256(raw).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
    out=HERE/'test_results_source_supplement';assert not out.exists();out.mkdir()
    source=Path(__file__).resolve();own=digest(source.read_bytes());view=HERE/'collected/test/metadata_view.json';raw=view.read_bytes()
    assert digest(raw)=='02ecab8a650510a9f6ba5d63f61369a6f59755bc3ee0389bc859f34131fa015b'
    m=json.loads(raw);qp=ROOT+'/test_queue.json';qraw=Path(m['files'][qp]).read_bytes();assert digest(qraw)==m['sha256'][qp];q=json.loads(qraw)
    expected={p:h for p,h in q['source_sha256'].items() if p not in m['files']};assert expected
    for p in expected:
        path=Path(p);assert path.is_absolute() and '..' not in path.parts
        assert path.suffix not in ('.pt','.pth','.ckpt','.npz','.npy')
        assert 'prediction' not in path.name.lower() and path.name not in ('sealed_test_report.md','research_report.md','supplementary_report.md')
        assert path.suffix in ('.py','.json','.txt','.md') or path.name in ('center_rows.jsonl','history.jsonl','EXIT_CODE','EXTERNAL_EXIT_CODE')
    (out/'expected_sha256.json').write_text(json.dumps(expected,indent=2)+'\n')
    relative=out/'relative_files.txt';relative.write_text(''.join(p.lstrip('/')+'\n' for p in sorted(expected)))
    remote="""import pathlib,hashlib,json,datetime
expected=json.loads(PAYLOAD)
actual={}
for name,wanted in expected.items():
 p=pathlib.Path(name);assert p.resolve(strict=True)==p and p.is_file()
 assert p.suffix not in ('.pt','.pth','.ckpt','.npz','.npy') and 'prediction' not in p.name.lower()
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 actual[name]=h.hexdigest();assert actual[name]==wanted,(name,actual[name],wanted)
print(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),sha256=actual,model_calls=0,prediction_read=False,tensor_read=False)))
""".replace('PAYLOAD',repr(json.dumps(expected)))
    ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','JJ',shlex.join([PY,'-c',remote])]
    records=[];start=now()
    def run(label,argv,timeout):
        began=now();p=subprocess.run(argv,capture_output=True,text=True,timeout=timeout)
        (out/(label+'.stdout.log')).write_text(p.stdout);(out/(label+'.stderr.log')).write_text(p.stderr)
        records.append(dict(label=label,argv=argv,started_utc=began,finished_utc=now(),actual_exit=p.returncode))
        assert type(p.returncode) is int and p.returncode==0,(label,p.returncode,p.stderr[-1000:]);return p.stdout
    try:
        before=json.loads(run('remote_before',ssh,120));payload=out/'payload';payload.mkdir()
        run('rsync',['rsync','-aR','--files-from',str(relative),'-e','ssh -o BatchMode=yes -o ConnectTimeout=15','JJ:/',str(payload)+'/'],180)
        files={p:str((payload/p.lstrip('/')).resolve()) for p in expected}
        copied={p:digest(Path(local).read_bytes()) for p,local in files.items()};assert copied==expected
        after=json.loads(run('remote_after',ssh,120));assert before['sha256']==after['sha256']==expected
        assert digest(source.read_bytes())==own and digest(view.read_bytes())==digest(raw)
        result=dict(schema='cap10_result_source_supplement_copy_map_20261025',files=files,sha256=copied,hash_only_inputs={},
          remote_before_sha256=before['sha256'],remote_after_sha256=after['sha256'],remote_inventory_before_exit=records[0]['actual_exit'],
          remote_inventory_after_exit=records[2]['actual_exit'],rsync_exit=records[1]['actual_exit'],
          provenance=dict(existing_test_view=dict(path=str(view),sha256=digest(raw)),actual_queue_sha256=digest(qraw),source_sha256=own),
          prediction_values_interpreted=False,tensor_values_interpreted=False,scientific_jobs=0)
        target=out/'artifact_map.json';target.write_text(json.dumps(result,indent=2)+'\n');(out/'EXIT_CODE').write_text('0\n')
        print(json.dumps(dict(status='PASS',map=str(target),sha256=digest(target.read_bytes()),files=len(files),actual_exits=[r['actual_exit'] for r in records])))
    except BaseException as exc:
        (out/'failure.json').write_text(json.dumps(dict(error=repr(exc)),indent=2)+'\n');(out/'EXIT_CODE').write_text('1\n');raise
    finally:
        (out/'execution.json').write_text(json.dumps(dict(started_utc=start,finished_utc=now(),records=records,source_sha256={str(source):own},scientific_jobs=0),indent=2)+'\n')
if __name__=='__main__':main()
