"""功能概述：将未进入原子侧表的理论谱按既有train-only跨源仿射拟合对齐。
align_unassigned不重拟合参数；已有物理/原子侧表键保持原值，由apply_variants接管。
仅处理理论train记录，保留观测缺失状态、权重和谱位置次序；明确窗口随正仿射同步
映射，不把已观测首末峰作为窗口。不对实验谱或test谱作校正。
输出新记录列表及每来源参数/数量审计，原缓存只读；供h_count_curriculum的显式
source-calibration消融使用。统计对齐参数不是已验证的仪器或逐原子化学位移校正。
"""
import collections
import numpy as np
from .h_labels import make_key


def align_unassigned(rows,variants,fits):
    result=[];counts=collections.Counter();parameters={}
    for r in rows:
        if r['domain']!='theory' or r['split']!='train':raise ValueError('theory_alignment_requires_train')
        entry=fits.get(r['source']+'|1H')
        if make_key(r) in variants or not entry or entry.get('molecules',0)<20:result.append(r);continue
        scale=float(entry['affine_scale']);offset=float(entry['affine_offset_ppm'])
        if not np.isfinite([scale,offset]).all() or scale<=0:raise ValueError('invalid_source_affine')
        observation=dict(r['observation']);observation['windows']=[[scale*float(l)+offset,scale*float(u)+offset] for l,u in observation.get('windows',[])];rr=dict(r,q=(scale*np.asarray(r['q'],np.float64)+offset).astype(np.float32),observation=observation);result.append(rr);counts[r['source']]+=1;parameters[r['source']]=dict(scale=scale,offset_ppm=offset,train_paired_molecules=entry['molecules'])
    return result,dict(records=dict(counts),parameters=parameters,scope='existing train-only statistical affine fits; no target lookup or fitting on val/test')
