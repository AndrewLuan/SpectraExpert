"""功能概述：为条件响应消融建立独占训练侧表与冻结GINE表示缓存，不改旧数据。
main复用原train/val结构分区、等量理论重放及原单核/双核候选并集；HMDB条件直接
来自原始记录，其他来源只恢复旧掩码所含信息。audit_metadata统计训练溶剂/pH支持，
列出同结构不同条件的谱形描述；这些比较未做原子对齐，不作纯溶剂因果估计。
EmbeddingSpectra在CPU计算候选局部环境；同一冻结骨干只编码一次，节点表示以
float32保存。检索候选只复用结构表示，条件向量稍后统一取自查询，不能取候选原谱。
输出cache.pt/metadata.pt/metadata_audit.json/manifest.json及进度/退出码；目录独占。
依赖已有prepared_coverage、site权重和固定bank，源文件及输入身份写入manifest。
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from rdkit import Chem
from .data import identity
from .train import Spectra,collate
from .h_conditions import encode_conditions,condition_from_old,local_environment
from .h_labels import make_key
from .h_observed import cluster_lines
from .emission_model import make_model
from .prepare import sha


class EmbeddingSpectra(Spectra):
    def __getitem__(self,i):
        g=super().__getitem__(i);g.local_environment=torch.tensor(local_environment(self.rows[i]['smiles']));return g


def audit_metadata(metadata,rows):
    support=collections.Counter();nonwater_ph=0;by_structure=collections.defaultdict(list);dmf=[]
    for key,meta in metadata.items():
        domain,source,rid,smi=key;split=identity(smi)[2]
        if source=='HMDB':
            support[split+'|'+meta['family']+'|pH_known='+str(meta['pH_used'])]+=1
            if meta['raw_pH'] is not None and not meta['pH_used']:nonwater_ph+=1
            if split=='train':by_structure[smi].append((rid,meta))
            if smi=='CN(C)C=O':dmf.append(dict(record_id=rid,split=split,**{k:meta[k] for k in ['solvent','raw_pH','pH_used','line_centers','raw_positions']}))
    pairs=[]
    for smi,rs in by_structure.items():
        for i,(rid,a) in enumerate(rs):
            for rid2,b in rs[i+1:]:
                if a['family']==b['family'] and a['pH']==b['pH']:continue
                pairs.append(dict(smiles=smi,record_ids=[rid,rid2],solvents=[a['solvent'],b['solvent']],pH=[a['pH'],b['pH']],centers=[a['line_centers'],b['line_centers']],
                    same_solvent=a['family']==b['family'],same_pH=a['pH']==b['pH'],scope='unassigned peak-cluster comparison; missing groups may differ'))
    return dict(HMDB_condition_support=dict(support),pH_recorded_but_not_usable=nonwater_ph,training_condition_pairs=pairs,DMF_test_only_diagnostic=dmf,
        training_rows=dict(collections.Counter(r['domain']+'|'+r['source'] for r in rows if r['split']=='train')),
        missing_metadata='NMRShiftDB 1H conditions empty in all 16756 raw records; frozen integral cache preserves old solvent masks only',
        split_scope='DMF entire connectivity group remains test in every source; no DMF record used for training or numerical selection')


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--out',required=True);a=p.parse_args();base=Path(a.base);out=Path(a.out);out.mkdir(parents=True,exist_ok=False);start=time.time()
    def status(stage,**kw):(out/'status.json').write_text(json.dumps(dict(stage=stage,seconds=time.time()-start,**kw)))
    try:
        torch.set_num_threads(4);torch.set_float32_matmul_precision('high');device=torch.device('cuda');status('loading')
        source=base/'prepared_coverage/cache.pt';cache=torch.load(source,map_location='cpu',weights_only=False,mmap=True)
        bank=torch.load(base/'hydrogen_analysis/evaluation_dropout/site/bank.pt',map_location='cpu',weights_only=False,mmap=True)
        exp=[r for r in cache['rows'] if r['nucleus']=='1H' and r['domain']=='experimental'];train=[r for r in exp if r['split']=='train'];val=[r for r in exp if r['split']=='val']
        theory=[r for r in cache['rows'] if r['nucleus']=='1H' and r['domain']=='theory' and r['split']=='train'];theory.sort(key=lambda r:hashlib.sha256((r['smiles']+r['source']+r['record_id']).encode()).hexdigest())
        rows=[r for r in train+theory[:len(train)]+val if r['physics'][0]>0];metadata={};raw_path=Path('/data/SpecGPT/dataset/NMR_exprimental/experimental_finetune/data.jsonl')
        for r in map(json.loads,raw_path.open()):
            if r['nucleus']!='1H':continue
            try:smi,_,_=identity(r['smiles'])
            except ValueError:continue
            meta=encode_conditions(r);meta['line_centers']=cluster_lines(r['value']['x'],.08)[0].tolist();meta['raw_positions']=r['value']['x']
            metadata['experimental','HMDB',r['record_id'],smi]=meta
        (out/'metadata_audit.json').write_text(json.dumps(audit_metadata(metadata,rows),indent=2))
        for r in rows:
            key=make_key(r)
            if key not in metadata:metadata[key]=condition_from_old(r['cond'])
        torch.save(metadata,out/'metadata.pt')
        required=sorted(set(bank['smiles'])|{r['smiles'] for r in rows});representative={r['smiles']:r for r in cache['rows']};representatives=[dict(representative[s],cond=np.zeros(4,np.float32),q=np.zeros(128,np.float32)) for s in required]
        checkpoint=base/'hydrogen_analysis/site_supervision/train/best.pt';saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if sha(checkpoint)!=bank['checkpoint_sha256']:raise ValueError('baseline_checkpoint_changed')
        model=make_model(saved['args']).to(device).eval();model.load_state_dict(saved['state_dict']);graphs={};offset=0;status('encoding_structures',structures=len(required))
        loader=DataLoader(EmbeddingSpectra(representatives,cache['graphs'],'1H'),batch_size=256,num_workers=8,collate_fn=collate,pin_memory=True)
        with torch.inference_mode():
            for j,b in enumerate(loader):
                gpu=b.to(device);embedding=model.structure_embedding(gpu).cpu();env=gpu.local_environment.cpu();ptr=gpu.ptr.cpu().tolist()
                for i in range(gpu.num_graphs):
                    smi=required[offset+i];g=cache['graphs'][smi].clone();g.frozen_embedding=embedding[ptr[i]:ptr[i+1]].clone();g.local_environment=env[ptr[i]:ptr[i+1]].clone();graphs[smi]=g
                offset+=gpu.num_graphs
                if j%100==0:status('encoding_structures',encoded=offset,total=len(required));print(json.dumps(dict(encoded=offset,total=len(required))),flush=True)
        status('saving_cache');torch.save(dict(rows=rows,graphs=graphs),out/'cache.pt')
        manifest=dict(rows=len(rows),graphs=len(graphs),H_checkpoint_sha256=sha(checkpoint),raw_HMDB_sha256=sha(raw_path),original_cache_sha256=saved['cache_sha256'],
            cache_sha256=sha(out/'cache.pt'),metadata_sha256=sha(out/'metadata.pt'),source_sha256={f.name:sha(f) for f in Path(__file__).parent.glob('*.py')},
            protocol='same original structural train/val partitions and equal theory replay; float32 frozen structure embeddings; no test fitting',seconds=time.time()-start)
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2));status('complete');(out/'EXIT_CODE').write_text('0\n')
    except Exception as exc:status('failed',error=repr(exc));(out/'EXIT_CODE').write_text('1\n');raise


if __name__=='__main__':main()
