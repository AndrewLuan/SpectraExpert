"""功能概述：用验证选定的共性观测粗排重新生成难负例，保留原特征算法。

ObservationScorer 仅在旧atom_count_1.0上叠加训练等价报告和HMDB容量；复用
h_global_extract.extract的完整114982候选、512含边界tie短名单、train正例补入、
缺峰生成和特征函数。新粗排协议显式记录，不冒充旧cache。所有新模块、训练侧表、
固定选择文件均加入cache来源SHA，未来locked重排器逐项核验。
main 先用original320+tune选定的共同配置缓存train/val；verify/test需额外冻结，
绝不在封存集搜索。进程内替换extract的Scorer类，无源码/数据改写，无外部API。
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from . import h_global_extract as extractor
from .h_condinfer import scoring_inputs
from .h_global_bank import load_bank
from .h_global_capacity import cluster_predictions
from .h_global_features import FEATURE_NAMES
from .h_global_observation_study import artifact_inputs,configs,corrections
from .h_global_protocol import actual_inputs
from .h_global_scoring import configs as coarse_configs,frozen_choices
from .h_observation_study import uniform_peaks
from .prepare import sha


def inputs(args):
    required=actual_inputs(args.base,args.bank,args.widths,args.cohorts,args.extension)+artifact_inputs(args.artifact)+[Path(args.feature_choice)]
    required += [Path(__file__).with_name(n) for n in ('h_global_observation_extract.py','h_global_observation_study.py','h_global_capacity.py','h_global_reporting.py')]
    return sorted({p.resolve() for p in required})


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    for name in ('base','bank','widths','cohorts','artifact','feature-choice','cohort','out'):p.add_argument('--'+name,required=True)
    p.add_argument('--extension');p.add_argument('--frozen-selection');p.add_argument('--limit',type=int,default=0)
    p.add_argument('--pool-size',type=int,default=512);p.add_argument('--augment',action='store_true')
    args=p.parse_args()
    if args.cohort not in ('train','original_val','tune','verify','test'):raise ValueError('observation_extract_wrong_cohort')
    selected=json.loads(Path(args.feature_choice).read_text())
    if selected['selection_scope']!='train_val_only':raise ValueError('observation_feature_choice_wrong_scope')
    config=selected['observation_config']
    if config not in configs():raise ValueError('observation_feature_choice_unregistered')
    choice=dict(name='observation_'+config['name'],base='atom_count_1.0',observation=config)
    base_choice=next(c for c in coarse_configs() if c['name']=='atom_count_1.0')
    required=inputs(args);pinned={str(p):sha(p) for p in required};selection_sha=None
    if args.cohort in ('verify','test'):
        if not args.frozen_selection or args.limit:raise ValueError('observation_extract_not_frozen')
        if base_choice not in frozen_choices(args.frozen_selection,required):raise ValueError('observation_extract_base_not_frozen')
        frozen=json.loads(Path(args.frozen_selection).read_text());protocol=frozen['feature_extractor_protocol']
        if config not in frozen['observation_configs']:raise ValueError('observation_extract_config_not_frozen')
        if args.pool_size!=protocol['pool_size'] or bool(args.augment)!=protocol['augment'] or FEATURE_NAMES!=protocol['feature_names']:
            raise ValueError('observation_extract_protocol_changed')
        if sha(Path(__file__).with_name('h_global_features.py'))!=protocol['feature_code_sha256']:
            raise ValueError('observation_extract_feature_code_changed')
        selection_sha=sha(args.frozen_selection)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    def status(stage,**kw):
        temp=out/'status.tmp';temp.write_text(json.dumps(dict(stage=stage,**kw)));temp.replace(out/'status.json')
    original_scorer=extractor.Scorer
    try:
        torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
        bank=load_bank(args.bank,args.extension);payload=torch.load(Path(args.cohorts)/'cohorts.pt',map_location='cpu',weights_only=False)
        if sha(bank['checkpoint_path'])!=bank['checkpoint_sha256']:raise ValueError('observation_extract_checkpoint_drift')
        smiles=[bank['smiles'][i] for i in bank['single_indices']]
        manifest=json.loads((Path(args.artifact)/'manifest.json').read_text())
        if manifest['candidate_order_sha256']!=hashlib.sha256('\n'.join(smiles).encode()).hexdigest():
            raise ValueError('observation_extract_candidate_order_changed')
        table=torch.load(Path(args.artifact)/'table.pt',map_location='cpu',weights_only=False)
        if not torch.allclose(table['H'],bank['physics'][bank['single_indices'],0].cpu()*40,atol=.001,rtol=0):
            raise ValueError('observation_extract_H_changed')
        models=json.loads((Path(args.artifact)/'models.json').read_text())
        class ObservationScorer(original_scorer):
            def __init__(self,bank,payload,widths,scoring,ignored,device):
                super().__init__(bank,payload,widths,scoring,base_choice,device)
                self.environments=table['K'].to(device);self.resolutions={}
            def group(self,key,hmdb):
                self.resolutions={};super().group(key,hmdb)
                if hmdb and config['capacity']:
                    self.resolutions[config['width']]=cluster_predictions(uniform_peaks(self.kernels['pred'])*10,config['width'])
            def score(self,query):
                values,baseline=super().score(query)
                delta,capacity=corrections(self,query,self.environments,models,self.resolutions)
                values=values+config['report']*delta
                if config['capacity'] and config['width'] in capacity:values=values+config['capacity']*capacity[config['width']]
                return values,baseline
        extractor.Scorer=ObservationScorer
        queries=bank['test'] if args.cohort=='test' else payload['cohorts'][args.cohort]
        if args.limit:queries=queries[:args.limit]
        widths=json.loads(Path(args.widths).read_text())['selected_ppm']
        (out/'protocol.json').write_text(json.dumps(dict(inputs_sha256=pinned,coarse_config=choice,
            pool_size=args.pool_size,cohort=args.cohort,source_sha256=sha(__file__)),indent=2))
        extractor.extract(bank,queries,payload,widths,scoring_inputs(Path(args.base)),choice,out,status,args.cohort,args.pool_size,args.augment)
        for path,digest in pinned.items():
            if sha(path)!=digest:raise ValueError('observation_extract_input_changed_during_run:'+path)
        if selection_sha and sha(args.frozen_selection)!=selection_sha:raise ValueError('observation_extract_selection_changed')
        if sha(bank['checkpoint_path'])!=bank['checkpoint_sha256']:raise ValueError('observation_extract_weight_changed_during_run')
        status('complete');(out/'EXIT_CODE').write_text('0\n')
    except Exception as exc:
        status('failed',error=repr(exc));(out/'EXIT_CODE').write_text('1\n');raise
    finally:
        extractor.Scorer=original_scorer


if __name__=='__main__':main()
