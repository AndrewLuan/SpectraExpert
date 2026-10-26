"""功能概述：用合成产物检验固定中心audit的中心/继承/完整曝光和真实12job来源闭环。
复用旧audit合成fixture，不打开真实缓存、测试数据或执行科学训练；破坏输入必须失败。
"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
import audit_pool_fixed_center_training as a
import audit_pool_training as old
from test_audit_pool_training import fixture as oldfixture
import test_audit_pool_training as legacytests


def fixture():
    s,p,o,c,h,e,x=oldfixture()
    e.update(training_pool='mixed',loss='listwise')
    p['args'].update(training_pool='mixed',loss='listwise')
    s.update(training_pool='mixed',loss='listwise',pool_exposure_sha256='m64',
             center_estimate_sha256='center',center_rows_sha256='rows',centers_frozen=True,
             scale_frozen=True,initial_tensors_inherited_exact=True,center_query_rows=2)
    estimate=dict(row_audit_sha256='rows',seed=193,augment=False,input_sha256=p['inputs_sha256'],
                  query_rows=2,constant_fp32=36,raw_constant_bytes=144,selection_scope='fit_train_only',
                  center33=list(range(33)),O_center3=[1.,2.,3.])
    for cp in c.values():
        cp['schema']=a.SCHEMA;cp['args']=copy.deepcopy(p['args']);cp['center_estimate_sha256']='center'
        cp['pool_exposure_sha256']='m'+str(cp['epoch'])
        cp['state_dict'].update(fixed_x_center=torch.arange(33,dtype=torch.float32),
                                fixed_o_center=torch.tensor([1.,2.,3.]))
    control=copy.deepcopy(c['last']);control['schema']=old.SCHEMA
    return s,p,o,c,h,e,x,estimate,'center','rows',control


class ModelTests(unittest.TestCase):
    def setUp(self):self.v=fixture()
    def check(self):return a.audit_model(*self.v)
    def test_complete_and_epoch_zero_best_pass(self):
        r=self.check();self.assertEqual(r['draws'],524288);self.assertEqual(r['optimizer_steps'],16384)
    def test_center_changed_each_phase(self):
        for phase in ('initial','best','last'):
            self.v=fixture();self.v[3][phase]['state_dict']['fixed_o_center'][0]+=1
            with self.assertRaisesRegex(ValueError,'fixed_center_changed'):self.check()
    def test_center_dtype_change(self):
        self.v[3]['last']['state_dict']['fixed_x_center']=torch.arange(33,dtype=torch.float64)
        with self.assertRaisesRegex(ValueError,'fixed_center_changed'):self.check()
    def test_center_nonfinite(self):
        self.v[7]['center33'][0]=float('nan')
        with self.assertRaisesRegex(ValueError,'center_capacity_finite'):self.check()
    def test_center_evidence_mismatch(self):
        self.v[0]['center_rows_sha256']='tampered'
        with self.assertRaisesRegex(ValueError,'center_evidence_SHA'):self.check()
    def test_center_input_changed(self):
        self.v[7]['input_sha256']={}
        with self.assertRaisesRegex(ValueError,'center_role_inputs'):self.check()
    def test_initial_tensor_changed(self):
        self.v[3]['initial']['state_dict']['network.weight']+=1
        with self.assertRaisesRegex(ValueError,'initial_tensors_changed'):self.check()
    def test_scale_or_alpha_changed(self):
        for field in ('scale','alpha'):
            self.v=fixture();self.v[3]['initial'][field]+=1
            with self.assertRaises(ValueError):self.check()
    def test_O_changed(self):
        self.v[3]['last']['state_dict']['observation_model.width_raw']+=1
        with self.assertRaisesRegex(ValueError,'observation_changed'):self.check()
    def test_short_budget(self):
        self.v[3]['last']['epoch']=56
        with self.assertRaises(ValueError):self.check()
    def test_history_incomplete(self):
        self.v[4].pop()
        with self.assertRaisesRegex(ValueError,'history_incomplete'):self.check()
    def test_control_exposure_changed(self):
        self.v[10]['pool_exposure_sha256']='wrong'
        with self.assertRaisesRegex(ValueError,'control_complete_exposure'):self.check()
    def test_control_role_changed(self):
        self.v[10]['args']['loss']='pairwise16'
        with self.assertRaisesRegex(ValueError,'control_role_budget'):self.check()
    def test_family_pairwise_disallowed(self):
        self.v[5]['loss']='pairwise16'
        with self.assertRaisesRegex(ValueError,'fixed_family_only'):self.check()


def build_tree(root):
    jobs=legacytests.FilesystemTests().make_tree(root)
    module=root/'h_pool_fixed_center_ranker.py';module.write_text('synthetic pinned source')
    queue_path=root/'additional_queue.json'
    source=root/'science.py';sourcepins={str(source):a.sha(source),str(module):a.sha(module)}
    chosen=[];run=root/'runs/additional';run.mkdir()
    for j in jobs:
        if j['kind']!='training' or 'fit_mixed_listwise_' not in j['id']:continue
        control=Path(j['out']);newout=root/'additional/rankers/fixedcenter_mixed_listwise'/control.parent.name/control.name
        newout.mkdir(parents=True)
        protocol=a.read_json(control/'protocol.json');args=protocol['args'];args.update(out=str(newout),source=str(root),cohort_contract=str(root/'contract.json'))
        (root/'contract.json').write_text('{}')
        protocol['inputs_sha256'].update(sourcepins)
        protocol['inputs_sha256'][str(root/'contract.json')]=a.sha(root/'contract.json')
        # Synthetic inherited capacity:10 tensors; original real models use their actual shapes.
        oldpath=Path(args['initial_checkpoint']);inherited=torch.load(oldpath,weights_only=False)
        for i in range(8):inherited['state_dict']['network.extra'+str(i)]=torch.tensor([float(i)])
        inherited['epoch']=0;torch.save(inherited,oldpath)
        protocol['inputs_sha256'][str(oldpath)]=a.sha(oldpath)
        (oldpath.parent/'summary.json').write_text(json.dumps(dict(best=dict(epoch=0,alpha=inherited['alpha']),training_cache_sha256=inherited['train_sha256'])))
        (oldpath.parent/'EXIT_CODE').write_text('0\n')
        estimate=dict(schema=a.SCHEMA,seed=args['seed'],augment=args['augment'],input_sha256=protocol['inputs_sha256'],
                      row_audit_sha256='',query_rows=2,constant_fp32=36,raw_constant_bytes=144,
                      selection_scope='fit_train_only',center33=[1.]*33,O_center3=[2.]*3)
        (newout/'center_rows.jsonl').write_text('{}\n{}\n')
        estimate['row_audit_sha256']=a.sha(newout/'center_rows.jsonl')
        (newout/'centers.json').write_text(json.dumps(estimate))
        center_sha=a.sha(newout/'centers.json')
        summary=a.read_json(control/'summary.json')
        summary.update(input_sha256=protocol['inputs_sha256'],center_estimate_sha256=center_sha,
                       center_rows_sha256=estimate['row_audit_sha256'],centers_frozen=True,
                       scale_frozen=True,initial_tensors_inherited_exact=True,center_query_rows=2)
        for name in ('initial','best','last'):
            cp=torch.load(control/(name+'.pt'),weights_only=False)
            cp['state_dict']=copy.deepcopy(inherited['state_dict'])
            cp.update(args=copy.deepcopy(args),ranker_source_sha256=sourcepins,initialization_sha256=a.sha(oldpath))
            if name=='last':
                baseline=copy.deepcopy(cp);baseline['schema']=old.SCHEMA
                torch.save(baseline,control/'last.pt')
            cp.update(schema=a.SCHEMA,center_estimate_sha256=center_sha)
            cp['state_dict'].update(fixed_x_center=torch.ones(33),fixed_o_center=torch.full((3,),2.))
            torch.save(cp,newout/(name+'.pt'))
        for n,v in (('summary.json',summary),('protocol.json',protocol)):(newout/n).write_text(json.dumps(v))
        (newout/'history.jsonl').write_bytes((control/'history.jsonl').read_bytes());(newout/'EXIT_CODE').write_text('0\n')
        argv=['python',str(module)]
        for k in ('source','train','dual_train','initial_checkpoint','cohort_contract','out','training_pool','loss','seed','epochs','samples','batch_size','anchor','lr'):
            argv+=['--'+k.replace('_','-'),str(args[k])]
        if args['augment']:argv+=['--augment']
        chosen.append(dict(id=j['id'],kind='training',argv=argv,cwd=str(root),out=str(newout),
                           deps=[],inputs_sha256={p:protocol['inputs_sha256'][p] for p in (args['train'],args['dual_train'],args['initial_checkpoint'])}))
    fitlookup={(a.read_json(Path(j['out'])/'protocol.json')['args']['seed'],
                a.read_json(Path(j['out'])/'protocol.json')['args']['augment']):j for j in chosen}
    for seed in (193,211):
        for pool in ('single','dual'):
            for cohort in ('original_val','tune'):
                cache=root/'old'/f's{seed}'/pool/cohort/'features.pt';cache.parent.mkdir(parents=True)
                cache.write_bytes(b'synthetic validation cache metadata only')
                producer=dict(feature_cache_sha256=a.sha(cache),cohort=cohort,recall=pool,
                              query_sha256='query_'+cohort,candidate_order_sha256='candidate')
                (cache.parent/'summary.json').write_text(json.dumps(producer));(cache.parent/'EXIT_CODE').write_text('0\n')
                out=root/'additional/validation'/f's{seed}'/pool/cohort;out.mkdir(parents=True)
                args=dict(stage='evaluate',source=str(root),cache=str(cache),seed=seed,out=str(out),
                          no_aug_checkpoint=str(Path(fitlookup[(seed,False)]['out'])/'best.pt'),
                          aug_checkpoint=str(Path(fitlookup[(seed,True)]['out'])/'best.pt'))
                inputs=dict(sourcepins)
                for path in (str(cache),args['no_aug_checkpoint'],args['aug_checkpoint']):inputs[path]=a.sha(path)
                checkpoints={args[k]:a.sha(args[k]) for k in ('no_aug_checkpoint','aug_checkpoint')}
                centers=[a.read_json(Path(fitlookup[(seed,aug)]['out'])/'summary.json')['center_estimate_sha256'] for aug in (False,True)]
                summary=dict(cohort=cohort,recall=pool,query_sha256=producer['query_sha256'],
                             candidate_order_sha256='candidate',input_sha256=inputs,checkpoints=checkpoints,center_estimate_sha256=centers)
                (out/'protocol.json').write_text(json.dumps(dict(args=args,inputs_sha256=inputs)))
                (out/'summary.json').write_text(json.dumps(summary));(out/'EXIT_CODE').write_text('0\n')
                argv=['python',str(module)]
                for k,v in args.items():argv+=['--'+k.replace('_','-'),str(v)]
                chosen.append(dict(id=f'eval_s{seed}_{pool}_{cohort}',kind='validation',out=str(out),
                                   argv=argv,cwd=str(root),deps=[],inputs_sha256={str(cache):a.sha(cache)}))
    queue=dict(jobs=chosen,source_inputs_sha256=sourcepins);queue_path.write_text(json.dumps(queue))
    bound=dict(sourcepins,**{str(queue_path):a.sha(queue_path)})
    start='2026-10-22T06:00:'
    receipts={}
    for j in chosen:
        r=dict(state='complete',external_exit=0,internal_exit=0,started_utc=start+'02+00:00',
               finished_utc=start+'03+00:00',argv=j['argv'],output=j['out'],cwd=j['cwd'],
               requested_inputs_sha256=j['inputs_sha256'],queue_source_sha256=bound)
        receipts[j['id']]=r;(run/(j['id']+'.receipt.json')).write_text(json.dumps(r))
    launch=dict(external_exit=0,internal_exit=0,started_utc=start+'00+00:00',finished_utc=start+'05+00:00',
                queue_sha256={str(queue_path):a.sha(queue_path)},bundle_sha256=sourcepins,source_sha256=sourcepins,
                child_pid=123,argv=['python','-m','driver','--queue',str(queue_path)])
    driver=dict(pid=123,argv=['driver.py','--queue',str(queue_path)],utc=start+'01+00:00',inputs_sha256=bound)
    completion=dict(success=True,inputs_unchanged=True,utc=start+'04+00:00',jobs=receipts)
    for n,v in (('launch_receipt.json',launch),('driver_start.json',driver),('completion.json',completion)):
        (run/n).write_text(json.dumps(v))
    for n in ('EXIT_CODE','EXTERNAL_EXIT_CODE'):(run/n).write_text('0\n')
    return queue_path,run,chosen


class FilesystemAuditTests(unittest.TestCase):
    def run_fixture(self,mutation=None):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();queue,run,jobs=build_tree(root)
            if mutation:mutation(root,queue,run,jobs)
            with patch.object(a.old,'recompute_exposure',return_value=(oldfixture()[-1],1,1)):
                return a.audit(root,queue,run)
    def test_actual_audit_entry_fourfit_twelve_receipts_pass(self):
        v=self.run_fixture();self.assertEqual(v['training_jobs'],4);self.assertEqual(v['validation_jobs'],8)
        self.assertTrue(all(v['checks'].values()));self.assertTrue(any(p.endswith('/last.pt') for p in v['input_sha256']))
    def test_missing_eval_summary_rejected(self):
        def mutate(root,q,r,jobs):
            for j in jobs:
                if j['kind']=='validation':(Path(j['out'])/'summary.json').unlink()
        with self.assertRaises(FileNotFoundError):self.run_fixture(mutate)
    def test_eval_cache_required_protocol_pin(self):
        def mutate(root,q,r,jobs):
            j=next(j for j in jobs if j['kind']=='validation');out=Path(j['out'])
            protocol=a.read_json(out/'protocol.json');del protocol['inputs_sha256'][protocol['args']['cache']]
            summary=a.read_json(out/'summary.json');summary['input_sha256']=protocol['inputs_sha256']
            (out/'protocol.json').write_text(json.dumps(protocol));(out/'summary.json').write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError,'eval_cache_required_protocol_pin'):self.run_fixture(mutate)
    def test_eval_checkpoint_swapped_rejected(self):
        def mutate(root,q,r,jobs):
            j=next(j for j in jobs if j['kind']=='validation');p=Path(j['out'])/'summary.json'
            v=a.read_json(p);v['checkpoints']={};p.write_text(json.dumps(v))
        with self.assertRaisesRegex(ValueError,'eval_checkpoint_SHA'):self.run_fixture(mutate)
    def test_boolean_exit_rejected(self):
        def mutate(root,q,r,jobs):
            p=r/(jobs[0]['id']+'.receipt.json');v=a.read_json(p);v['internal_exit']=False;p.write_text(json.dumps(v))
            c=a.read_json(r/'completion.json');c['jobs'][jobs[0]['id']]=v;(r/'completion.json').write_text(json.dumps(c))
        with self.assertRaisesRegex(ValueError,'not_double_exit0'):self.run_fixture(mutate)
    def test_synthetic_time_rejected(self):
        def mutate(root,q,r,jobs):
            p=r/'launch_receipt.json';v=a.read_json(p);v['finished_utc']='synthetic';p.write_text(json.dumps(v))
        with self.assertRaisesRegex(ValueError,'invalid_completion_utc'):self.run_fixture(mutate)
    def test_center_file_tamper_rejected(self):
        def mutate(root,q,r,jobs):
            p=Path(jobs[0]['out'])/'centers.json';v=a.read_json(p);v['center33'][0]=99.;p.write_text(json.dumps(v))
        with self.assertRaisesRegex(ValueError,'center_evidence_SHA'):self.run_fixture(mutate)
    def test_control_pool_tamper_rejected(self):
        def mutate(root,q,r,jobs):
            p=root/'formal/rankers/mixed_listwise/s193/no_aug/last.pt';v=torch.load(p,weights_only=False)
            v['pool_exposure_sha256']='tampered';torch.save(v,p)
        with self.assertRaisesRegex(ValueError,'control_summary_binding'):self.run_fixture(mutate)


if __name__=='__main__':unittest.main()
