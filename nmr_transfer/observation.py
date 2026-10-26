"""功能概述：区分完整原子目标、明确测量窗口、部分窗口及未知覆盖，不把缺测当零。
主要职责：coverage仅从明确ppm窗口元数据或逐核全原子覆盖证据建立观测状态；
observed_prediction在已知窗口条件化预测分位数；observation_loss对完整/已测窗口
用条件分布误差，对未知/部分标注只做观测位置到预测的单向匹配。
未知覆盖不从首末峰推断测量窗。部分原子标签不等于完整谱。无强度的峰表不造积分。
明确声明peak_coverage为partial/incomplete时保留partial_peaks标志，供观测格式路由
选择在val缺峰变体上校准的评分；它不是根据真结构或观测首末点推断的。
窗外预测不参与匹配；预测窗内无质量时保留到窗口的距离梯度。原始目标不改写。
训练和检索共用此契约；单向匹配保留未观测峰的自由度，另用冻结理论模型保形约束。
"""
import math
import torch

KINDS={'complete_atoms':0,'window':1,'unknown':2,'partial_window':3}


def coverage(record, expected_atoms=None):
    partial=record.get('peak_coverage') in ('partial','incomplete') or record.get('all_peaks_in_measured_window') is False
    def state(value):
        if partial:value['partial_peaks']=True
        return value
    intervals=None
    for parent in [record,record.get('experimental_conditions') or {},record.get('conditions') or {},record.get('coverage') or {}]:
        for key in ['observed_ppm_ranges','measured_ppm_ranges','measurement_window_ppm','spectral_window_ppm','intervals_ppm']:
            if isinstance(parent,dict) and parent.get(key) is not None:
                intervals=parent[key];break
        if intervals is not None:break
    if intervals is None:
        meta=record.get('source_metadata') or {}
        try:
            width=float(meta.get('sweep-width-ppm'));center=float(meta.get('center-frequency-ppm'))
            if math.isfinite(width) and math.isfinite(center) and width>0:
                intervals=[[center-width/2,center+width/2]]
        except (ValueError,TypeError):pass
    if intervals is not None:
        if len(intervals)==2 and all(isinstance(x,(int,float)) for x in intervals):intervals=[intervals]
        windows=[]
        for a,b in intervals:
            a,b=float(a),float(b)
            if not math.isfinite(a+b) or a>=b:raise ValueError('invalid_observation_window')
            windows.append([a,b])
        if not 1<=len(windows)<=8:raise ValueError('unsupported_window_count')
        positions=record.get('value',{}).get('x')
        if positions and any(not any(a<=float(x)<=b for a,b in windows) for x in positions):
            return state({'kind':'unknown','windows':[],'evidence':'explicit_window_conflicts_with_observed_positions; retained_for_audit'})
        # 测了某频段不等于峰表完整；只有明确的峰覆盖声明才允许负证据。
        complete=record.get('peak_coverage')=='complete' or record.get('all_peaks_in_measured_window') is True
        return state({'kind':'window' if complete else 'partial_window','windows':windows,
                'evidence':'explicit_ppm_metadata; peak completeness '+('declared' if complete else 'unknown')})
    if expected_atoms is not None:
        mask=record.get('atom_target_mask')
        targets=record.get('value',{}).get('atom_index')
        observed=sum(bool(v) for v in mask) if mask is not None else len(targets) if targets is not None else -1
        if expected_atoms>0 and observed==expected_atoms:
            return state({'kind':'complete_atoms','windows':[],'evidence':'all_target_nucleus_atoms_present; atom-shift distribution, not multiplet lines'})
    return state({'kind':'unknown','windows':[],'evidence':'measurement_window_not_supplied; first_and_last_peak_not_used_as_window'})


def observed_prediction(pred, windows, window_mask):
    active=(((pred[:,:,None]>=windows[:,None,:,0]) & (pred[:,:,None]<=windows[:,None,:,1])) & window_mask[:,None,:]).any(-1)
    has_window=window_mask.any(-1)
    active=torch.where(has_window[:,None],active,torch.ones_like(active))
    n=active.sum(1)
    ordered=pred.masked_fill(~active,float('inf')).sort(1).values
    q=(torch.arange(pred.shape[1],device=pred.device)+.5)/pred.shape[1]
    indices=(n[:,None]*q).long().clamp(0,pred.shape[1]-1)
    conditional=ordered.gather(1,indices)
    distance=(windows[:,None,:,0]-pred[:,:,None]).clamp_min(0)+(pred[:,:,None]-windows[:,None,:,1]).clamp_min(0)
    distance=distance.masked_fill(~window_mask[:,None,:],float('inf')).min(-1).values
    penalty=torch.where(has_window,distance.min(-1).values,torch.zeros_like(n,dtype=pred.dtype))
    conditional=torch.where((n>0)[:,None],conditional,pred)
    return conditional,penalty,n


def observation_loss(pred,batch):
    conditional,penalty,n=observed_prediction(pred,batch.windows,batch.window_mask)
    target=batch.q
    full=(conditional-target).abs().mean(1)
    # 只要求每个观测点能匹配一个预测；未观测峰不会作为假阳性被惩罚。
    directed=(target[:,:,None]-conditional[:,None,:]).abs().min(-1).values.mean(1)
    unknown=(batch.coverage_kind==KINDS['unknown']) | (batch.coverage_kind==KINDS['partial_window'])
    error=torch.where(unknown,directed,full)
    return error+torch.where(n==0,penalty,torch.zeros_like(penalty))
