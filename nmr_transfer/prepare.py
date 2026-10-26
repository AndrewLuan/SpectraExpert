"""功能概述：准备理论预训练、实验适配及只在训练分区拟合的域差异审计。
主要职责：main 顺序读指定源、统一核种/分位数并做全域连接结构划分；paired_audit
按同结构同核种同来源配对，分子等权拟合稳健位移缩放及分位数残差，不作原子归属MAE。
读取QME14S、FG26、QM9-NMR与最新两份重算谱；实验含全量NMRShiftDB2/HMDB与
既有按记录hash 1/48采样的NMRexp/WanJuan缓存。此采样身份在manifest明确登记。
结构算法registry仅提供结构分组数；同结构冲突不择优。128分位数为谱位置分布代理。
仅训练结构参与配对偏差拟合，test不参与。所有源哈希、排除及划分均审计。
graph_worker CPU并行构建无标签显式氢图；图构建失败排除同时落清单，不静默丢样本。
输出新目录cache.pt/domain_shift.json/manifest.json；拒绝覆盖，失败无complete标记。
"""
import argparse
import collections
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from scipy.optimize import least_squares
from .data import identity, graph, quantiles, physical_features, condition, SCALES
from .observation import coverage
from rdkit import Chem
from torch_geometric.data import Data


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def rows(path):
    with open(path) as f:
        for line in f:
            yield json.loads(line)


def graph_worker(smiles):
    try:
        g=graph(smiles)
        # 通过numpy传结果，避免几十万PyTorch共享storage耗尽服务器1024个FD限额。
        return smiles, (g.x.numpy(),g.edge_index.numpy(),g.edge_attr.numpy()), None
    except Exception as e:
        return smiles, None, str(e)


def environment(atom):
    if atom.GetAtomicNum()==1:
        if atom.GetDegree()!=1:return 'H_other'
        parent=atom.GetNeighbors()[0]
        if parent.GetAtomicNum() in (7,8,16):return 'exchange_H'
        if parent.GetIsAromatic():return 'aromatic_H'
        return 'H_parent_'+str(parent.GetAtomicNum())+'_'+str(parent.GetHybridization())
    if atom.GetIsAromatic():return 'aromatic_C'
    if any(b.GetBondType()==Chem.BondType.DOUBLE and b.GetOtherAtom(atom).GetAtomicNum()==8 for b in atom.GetBonds()):return 'carbonyl_C'
    return 'C_'+str(atom.GetHybridization())


def atom_aligned_audit(pairs):
    report={}
    for nucleus,items in pairs.items():
        x=np.array([i[0] for i in items]);y=np.array([i[1] for i in items]);scale=SCALES[nucleus]
        fitted=least_squares(lambda ab:(ab[0]*x+ab[1]-y)/scale,[1.,0.],loss='soft_l1',f_scale=.02,
                            bounds=([.5,-scale],[1.5,scale]),max_nfev=100)
        a,b=fitted.x;residual=y-(a*x+b);tags=collections.defaultdict(list)
        for item,e in zip(items,residual):tags[item[2]].append(float(e))
        report['recalculated|'+nucleus]=dict(calibration_basis='matched_record_id+nucleus+source_atom_index; training molecules only',
            atoms=len(items),molecules=len({i[3] for i in items}),affine_scale=float(a),affine_offset_ppm=float(b),
            raw_atom_MAE_ppm=float(np.mean(np.abs(x-y))),fitted_atom_MAE_ppm=float(np.mean(np.abs(residual))),
            residual_mad_ppm=float(np.median(np.abs(residual-np.median(residual)))),
            environment_residuals={k:dict(atoms=len(v),median_ppm=float(np.median(v)),MAD_ppm=float(np.median(np.abs(np.array(v)-np.median(v))))) for k,v in tags.items()},
            quantile_residual_mean_ppm=[0.]*128,fit_is_training_description_not_heldout_improvement=True)
    return report


def paired_audit(records):
    theoretical = collections.defaultdict(list)
    experimental = collections.defaultdict(list)
    for r in records:
        if r['split'] != 'train':
            continue
        key = (r['smiles'], r['nucleus'], r['source'])
        (theoretical if r['domain'] == 'theory' else experimental)[key].append(r['q'])
    exp_by_structure = collections.defaultdict(list)
    for (s, n, source), qs in experimental.items():
        exp_by_structure[s, n].append(np.median(qs, axis=0))
    paired = collections.defaultdict(list)
    for (s, n, source), qs in theoretical.items():
        eq = exp_by_structure.get((s, n))
        if eq:
            paired[source, n].append((s, np.median(qs, axis=0), np.median(eq, axis=0)))
    report = {'pairing': 'same isomeric canonical SMILES and nucleus; train only; median per molecule/source; quantile correspondence, not atom correspondence', 'fits': {}}
    for (source, n), pairs in paired.items():
        x = np.stack([p[1] for p in pairs]); y = np.stack([p[2] for p in pairs])
        scale = SCALES[n]
        result = least_squares(lambda ab: ((ab[0]*x + ab[1] - y)/scale).ravel(),
                               [1., 0.], loss='soft_l1', f_scale=.02, bounds=([.5, -scale], [1.5, scale]), max_nfev=100)
        a, b = result.x
        residual = y - (a*x + b)
        raw = np.abs(x-y).mean(1); corrected = np.abs(residual).mean(1)
        strata = collections.defaultdict(list)
        for s, xx, yy in pairs:
            p = physical_features(s)
            tags = [('exchange_H' if p[2] > 0 else 'no_exchange_H'), ('amide' if p[3] > 0 else 'no_amide'), ('aromatic' if p[4] > 0 else 'non_aromatic')]
            for tag in tags:
                strata[tag].append(float(np.mean(yy-(a*xx+b))))
        report['fits'][source+'|'+n] = dict(molecules=len(pairs), affine_scale=float(a), affine_offset_ppm=float(b),
            raw_mean_quantile_L1_ppm=float(raw.mean()), fitted_mean_quantile_L1_ppm=float(corrected.mean()),
            residual_median_ppm=float(np.median(residual)), residual_mad_ppm=float(np.median(np.abs(residual-np.median(residual)))),
            quantile_residual_mean_ppm=residual.mean(0).tolist(),
            strata={k: {'molecules':len(v), 'mean_residual_ppm':float(np.mean(v))} for k,v in strata.items()},
            fit_is_training_description_not_heldout_improvement=True)
    return report


def main():
    p = argparse.ArgumentParser(); p.add_argument('--out', required=True); p.add_argument('--workers', type=int, default=16)
    a = p.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    root = Path('/data/SpecGPT'); spec = root/'dataset'
    sources = [
        ('QME14S', 'theory', spec/'NMR/full_spectra/NMR_QME14S.jsonl'),
        ('FG26', 'theory', spec/'NMR/full_spectra/NMR_fg26_1H.jsonl'),
        ('FG26', 'theory', spec/'NMR/full_spectra/NMR_fg26_13C.jsonl'),
        ('QM9-NMR', 'theory', spec/'NMR/molecular_atom_assigned/data.jsonl'),
        ('recalculated', 'theory', root/'benchmarks/theoretical_1H_NMR.jsonl'),
        ('recalculated', 'theory', root/'benchmarks/theoretical_13C_NMR.jsonl'),
        ('NMRShiftDB2', 'experimental', spec/'NMR_exprimental/atom_assigned/data.jsonl'),
        ('HMDB', 'experimental', spec/'NMR_exprimental/experimental_finetune/data.jsonl')]
    tally = collections.Counter(); exclusions = collections.Counter(); records = []; seen = set(); identity_cache = {}; hashes = {}; start = time.time()
    theory_atoms={};atom_pairs=collections.defaultdict(list);atom_environments={}
    bad = (out/'excluded.jsonl').open('x'); atom_counts={}

    def append(r, source, domain, xs, weights=None, measure='peak_equal'):
        try:
            key = r['smiles']
            if key not in identity_cache:
                identity_cache[key] = identity(key)
            smiles, group, split = identity_cache[key]
            nucleus = r['nucleus']; q = quantiles(xs, weights)
            if smiles not in atom_counts:
                mol=Chem.AddHs(Chem.MolFromSmiles(smiles))
                atom_counts[smiles]={n:sum(a.GetAtomicNum()==z for a in mol.GetAtoms()) for n,z in [('1H',1),('13C',6)]}
            complete_candidate=source in ('QM9-NMR','recalculated')
            observation_state=coverage(r,atom_counts[smiles][nucleus] if complete_candidate else None)
            lo, hi = (-10, 30) if nucleus == '1H' else (-100, 400)
            if q.min() < lo or q.max() > hi:
                raise ValueError('outside_declared_range')
            cond = condition(r)
            observation = (smiles, nucleus, domain, source, q.tobytes(), tuple(cond))
            if observation in seen:
                exclusions['duplicate_observation'] += 1; return
            seen.add(observation)
            item=dict(smiles=smiles, group=group, split=split, nucleus=nucleus, source=source, domain=domain,
                q=q, cond=np.asarray(cond, np.float32), record_id=str(r.get('record_id') or ''), measure=measure,
                observation=observation_state)
            if source=='recalculated':item['atom_values']=list(xs);item['atom_indices']=r['value']['atom_index']
            records.append(item)
            tally[f'{domain}|{source}|{nucleus}|{split}'] += 1
        except Exception as exc:
            reason = str(exc); exclusions[reason] += 1
            bad.write(json.dumps({'source': source, 'record_id': str(r.get('record_id') or ''), 'reason': reason})+'\n')

    for source, domain, path in sources:
        hashes[str(path)] = sha(path)
        for r in rows(path):
            n = r.get('nucleus') or ('1H' if r.get('spectrum_type') == 'theoretical_1H_NMR' else '13C' if r.get('spectrum_type') == 'theoretical_13C_NMR' else None)
            if n not in SCALES:
                exclusions['not_H_or_C'] += 1; continue
            r['nucleus'] = n
            if source == 'QM9-NMR':
                if r.get('quantity_kind') != 'isotropic_shielding' or r.get('conditions', {}).get('phase') != 'gas':
                    raise ValueError('unexpected_qm9_quantity_or_phase')
                ref = 31.7608 if n == '1H' else 186.9704
                xs = [ref-float(x) for x,m in zip(r['atom_targets'],r['atom_target_mask']) if m]
                append(r, source, domain, xs, measure='atom_equal')
            elif source == 'NMRShiftDB2':
                xs = [float(x) for x,m in zip(r['atom_targets'],r['atom_target_mask']) if m]
                append(r, str(r['source_ref']), domain, xs, measure='observed_assigned_atom_equal_partial_possible')
                try:
                    smi,group,split=identity_cache.get(r['smiles']) or identity(r['smiles'])
                    mol=Chem.AddHs(Chem.MolFromSmiles(r['smiles']));order=r['atom_order']['source_to_canonical']
                    if [mol.GetAtomWithIdx(i).GetAtomicNum() for i in order]!=r['atom_order']['atomic_numbers']:raise ValueError('atom_alignment')
                    env=[environment(mol.GetAtomWithIdx(i)) for i in order]
                    for nn in SCALES:
                        if (r['record_id'],nn) in theory_atoms:atom_environments[r['record_id'],nn]=env
                    theory=theory_atoms.get((r['record_id'],n))
                    if theory and split=='train':
                        assert theory['smiles']==r['smiles']
                        values=dict(zip(theory['value']['atom_index'],theory['value']['x']))
                        for i,(v,m) in enumerate(zip(r['atom_targets'],r['atom_target_mask'])):
                            if m and i+1 in values:atom_pairs[n].append((values[i+1],float(v),env[i],group))
                except (ValueError,KeyError,AssertionError):exclusions['atom_pair_unverified']+=1
            else:
                xs = r['value']['x']
                if source=='recalculated':theory_atoms[r['record_id'],n]=r
                append(r, source, domain, xs, measure='atom_equal' if source == 'recalculated' else 'peak_equal_intensity_unused')
        print(json.dumps({'source':str(path), 'retained':len(records), 'seconds':time.time()-start}), flush=True)
    # 复用已冻结采样缓存；不下载、不改写14GB原文件。原始输入sha登记于对应audit.json。
    cache = Path('/home/user/IRfingerprints/SpectrExpert/outputs/nmr_forward_20260922/prepared_v2_mod48/cache.pt')
    hashes[str(cache)] = sha(cache)
    for r in torch.load(cache, map_location='cpu', weights_only=False):
        append(dict(smiles=r['smiles'], nucleus='1H', record_id=r['record_id'], solvent='', frequency=0),
               str(r['source']), 'experimental', r['xs'], r['weights'], 'integral_weighted')
        # 保留已规范化的条件，不从峰位置反推实验条件。
        if records and records[-1]['record_id'] == str(r['record_id']):
            records[-1]['cond'] = np.asarray(r['cond'], np.float32)
    bad.close()
    # 结构registry的计数不得由理论/实验位移推导；跨同结构不一致时去除。
    base = root/'benchmarks/nmr_constraint_count/baseline'
    structures={}
    for r in rows(base/'structures.jsonl'):
        try:structures[r['record_id']]=identity(r['source_smiles'])[0]
        except ValueError:continue
    registry = {}; conflicting = set()
    for r in rows(base/'predictions.jsonl'):
        if r['record_id'] not in structures:continue
        smi = structures[r['record_id']]
        counts = {n:v['count'] for n,v in r['nuclei'].items() if v.get('status') == 'predicted'}
        if smi in registry and registry[smi] != counts:
            conflicting.add(smi)
        registry[smi] = counts
    for s in conflicting:
        registry.pop(s, None)
    for name in ['structures.jsonl', 'predictions.jsonl']:
        hashes[str(base/name)] = sha(base/name)
    smiles_list = sorted({r['smiles'] for r in records})
    atom_report=atom_aligned_audit(atom_pairs)
    initial=paired_audit(records);initial['atom_aligned_fits']=atom_report
    (out/'domain_shift_initial.json').write_text(json.dumps(initial, indent=2))
    graphs = {}; graph_errors = {}
    with ProcessPoolExecutor(max_workers=a.workers) as executor:
        for i,(smiles,g,error) in enumerate(executor.map(graph_worker, smiles_list, chunksize=128),1):
            if error is None:
                graphs[smiles] = Data(x=torch.from_numpy(g[0]),edge_index=torch.from_numpy(g[1]),edge_attr=torch.from_numpy(g[2]))
            else:
                graph_errors[smiles] = error
            if i % 10000 == 0:
                print(json.dumps({'graphs':i,'total':len(smiles_list),'seconds':time.time()-start}),flush=True)
    kept = []; physics_cache = {}
    for r in records:
        if r['smiles'] in graph_errors:
            exclusions['graph:'+graph_errors[r['smiles']]] += 1
        else:
            if r['smiles'] not in physics_cache:
                physics_cache[r['smiles']] = physical_features(r['smiles'], registry.get(r['smiles']))
            r['physics'] = physics_cache[r['smiles']]
            kept.append(r)
    groups = {s:{r['group'] for r in kept if r['split'] == s} for s in ['train','val','test']}
    assert not groups['train'] & groups['val'] and not groups['train'] & groups['test'] and not groups['val'] & groups['test']
    domain = paired_audit(kept)
    domain['atom_aligned_fits']=atom_report
    for key,value in atom_report.items():
        if key in domain['fits']:value['distribution_diagnostic']=domain['fits'][key]
        domain['fits'][key]=value
    for r in kept:
        if r['source']!='recalculated':continue
        fit=atom_report.get('recalculated|'+r['nucleus'])
        env=atom_environments.get((r['record_id'],r['nucleus']))
        if fit and env:
            corrected=[]
            for value,index in zip(r['atom_values'],r['atom_indices']):
                tag=env[index-1];entry=fit['environment_residuals'].get(tag,{})
                residual=entry.get('median_ppm',0.) if entry.get('atoms',0)>=100 else 0.
                corrected.append(fit['affine_scale']*value+fit['affine_offset_ppm']+residual)
            r['domain_q']=quantiles(corrected)
        r.pop('atom_values',None);r.pop('atom_indices',None)
    (out/'domain_shift.json').write_text(json.dumps(domain, indent=2))
    torch.save(dict(rows=kept, graphs=graphs), out/'cache.pt')
    manifest = dict(counts=dict(collections.Counter(f"{r['domain']}|{r['source']}|{r['nucleus']}|{r['split']}" for r in kept)),
        exclusions=dict(exclusions), graph_errors=graph_errors, input_sha256=hashes, graphs=len(graphs), records=len(kept),
        partition='sha256(nonstereo canonical SMILES) first8 hex modulo100, 80/10/10, shared across all sources and nuclei',
        experimental_peak_table_sample='prepared_v2_mod48: deterministic record-hash 1/48; full NMRShiftDB2 and HMDB additionally included',
        physics_registry=dict(structures=len(registry), conflicts=len(conflicting), use='structure-only counts, never exact decimal peak counts'),
        target='128 chemical-shift quantiles; atom, observed-atom, equal-peak or integral measure explicitly recorded; not sampled multiplet lines',
        observation_states=dict(collections.Counter(f"{r['domain']}|{r['source']}|{r['nucleus']}|{r['observation']['kind']}" for r in kept)),
        missing_coverage='unknown metadata stays unknown; never infer measured interval from first/last peak; complete atom targets checked against structural nucleus count',
        references={'QM9-NMR|13C':{'TMS_gas_ppm':186.9704,'url':'https://moldis-group.github.io/qm9nmr/'},
                    'QM9-NMR|1H':{'TMS_gas_ppm':31.7608,'url':'https://pubs.rsc.org/en/content/articlehtml/2024/dd/d4dd00098f','note':'same theory/geometry level; independently computed reference, not a supplied QM9-NMR H reference'}},
        cache_sha256=sha(out/'cache.pt'), seconds=time.time()-start)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)); (out/'complete.json').write_text(json.dumps({'records':len(kept)}))
    print(json.dumps({'complete':True, 'records':len(kept), 'graphs':len(graphs)}),flush=True)


if __name__ == '__main__':
    main()
