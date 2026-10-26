"""功能概述：为2026-10-22前向消融复用既有固定物理组解码，训练真实末层GINE。

ResearchForwardSpecGNN继承PreservedGroupSpecGNN；它没有重新发明固定组算法。
configure_forward_training只解冻原始atom_bond_block_list最后层、对应bond编码器、
equivalence_head与新增sigma头。其余原参数冻结；physical/local使用相同参数布局。
strip_embedding_caches显式从图副本删除旧节点表示；forward拒绝三类旧embedding，
避免prepared缓存绕过已解冻GINE。只在一次forward作用域复用当次live计算图，finally
清空引用，不跨batch/条件保存embedding。纯结构分组memo在每次forward重建，防止两个
布局共享同一Batch时沿用错误分区。完整核质量及128分位数仍由既有解码器保证。

details额外返回[B,MAX_GROUPS]的group_centers/group_Hmass/group_sigma/group_trust
与group_mask，以及逐发射component_sigma/component_group。位移与sigma均为ppm/10；
Hmass保持真实H数量，trust只表示registry结构分区可用，不是测量完整性或正确率。
quantile_sigma_ppm按与128分位数完全相同的成分排序/CDF索引选择sigma并乘10；
quantile_sigma保留ppm/10，供新bank与观测评分使用，不能把归一化sigma误当ppm。
未知分区精确保留原成分/分位数，group_mask/trust为零，sigma走同一个母节点头。
sigma是预测误差宽度，不是J耦合或仪器谱线宽度，由新trainer的观测NLL显式监督。

load_checkpoint_model是新bank构建入口，支持旧Preserved checkpoint暖启动及新F权重；
旧make_model无需修改。输入图仍须调用strip_embedding_caches；拒绝静默使用旧缓存。
函数不读实验目标、source/ID或候选自身条件来生成分组/宽度，没有文件写入或外部API。
"""
from pathlib import Path

import torch
from torch import nn
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_dense_batch

from .h_preserved_group_model import PreservedGroupSpecGNN
from .h_equivalence_model import MAX_GROUPS
from .h_conditions import CONDITION_DIM, LOCAL_DIM
from .h_acyl import ACYL_DIM
from .model import MLP


EMBEDDING_CACHES = ('frozen_embedding', 'final_context_embedding', 'unconditioned_context_embedding')
LAYOUT_CACHES = ('equivalence_ids', 'equivalence_offsets', 'equivalence_duplicates', 'equivalence_known',
                 'preserved_group_orbits', 'preserved_group_ordinal', 'preserved_group_sizes')
FORWARD_VERSION = 1


def quantile_sigmas(details, batches, points=128, fallback=.025):
    """与既有weighted_quantiles同次索引契约选宽度；返回归一化sigma。"""
    values, weights, groups = (details[name] for name in ('values', 'weights', 'groups'))
    if not len(values):
        return values.new_full((batches, points), fallback)
    packed, _ = to_dense_batch(values, groups, fill_value=1e6, batch_size=batches)
    mass, _ = to_dense_batch(weights, groups, fill_value=0., batch_size=batches)
    sigmas, _ = to_dense_batch(details['component_sigma'], groups, fill_value=fallback, batch_size=batches)
    ordered, order = packed.sort(1)
    mass = mass.gather(1, order)
    cdf = mass.cumsum(1) / mass.sum(1, keepdim=True).clamp_min(1)
    quantiles = ((torch.arange(points, device=values.device) + .5) / points)[None, :].expand(batches, -1).contiguous()
    indices = torch.searchsorted(cdf.contiguous(), quantiles).clamp_max(ordered.shape[1] - 1)
    if not torch.equal(ordered.gather(1, indices), details['quantiles']):
        raise ValueError('research_forward_sigma_quantile_index_mismatch')
    return sigmas.gather(1, order).gather(1, indices)


def strip_embedding_caches(graph, clone=True):
    """仅删除图表示及模型布局memo；保留原图、条件、观测与标签，默认不改输入对象。"""
    result = graph.clone() if clone else graph
    for name in EMBEDDING_CACHES + LAYOUT_CACHES:
        if hasattr(result, name):
            delattr(result, name)
    return result


def configure_forward_training(model, enabled=True):
    last = len(model.atom_bond_block_list) - 1
    if last < 0:
        raise ValueError('research_forward_requires_real_GINE')
    prefixes = (f'atom_bond_block_list.{last}.', f'bond_emb_list.{last}.',
                'equivalence_head.', 'research_sigma_head.')
    names = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(bool(enabled and name.startswith(prefixes)))
        if parameter.requires_grad:
            names.append(name)
    if enabled and any(not any(name.startswith(prefix) for name in names) for prefix in prefixes):
        raise ValueError('research_forward_missing_trainable_module')
    model.use_frozen_embedding = False
    return prefixes, names


class ResearchForwardSpecGNN(PreservedGroupSpecGNN):
    def __init__(self, *args, sigma_min=.002, sigma_max=.15, sigma_initial=.025, **kwargs):
        super().__init__(*args, **kwargs)
        width = args[1] if len(args) > 1 else kwargs['width']
        if not 0 < sigma_min < sigma_initial < sigma_max:
            raise ValueError('research_forward_invalid_sigma_bounds')
        self.sigma_min, self.sigma_max = float(sigma_min), float(sigma_max)
        self.research_sigma_head = MLP([2 * width + 12 + CONDITION_DIM + LOCAL_DIM + ACYL_DIM, 128, 1])
        nn.init.zeros_(self.research_sigma_head[-1].weight)
        fraction = torch.tensor((sigma_initial - sigma_min) / (sigma_max - sigma_min))
        nn.init.constant_(self.research_sigma_head[-1].bias, float(torch.logit(fraction)))
        self.use_frozen_embedding = False
        self._research_active_batch = None
        self._research_live_embedding = None
        configure_forward_training(self)

    def structure_embedding(self, batch):
        if any(hasattr(batch, name) for name in EMBEDDING_CACHES):
            raise ValueError('research_forward_cached_embedding_forbidden:call_strip_embedding_caches')
        if self.use_frozen_embedding:
            raise ValueError('research_forward_live_GINE_required')
        if self._research_active_batch == id(batch):
            if self._research_live_embedding is None:
                self._research_live_embedding = super().structure_embedding(batch)
            return self._research_live_embedding
        return super().structure_embedding(batch)

    def forward(self, batch, details=False):
        if self._research_active_batch is not None:
            raise ValueError('research_forward_reentrant_call_forbidden')
        self._research_active_batch = id(batch)
        try:
            return self._forward_live(batch, details)
        finally:
            self._research_live_embedding = None
            self._research_active_batch = None

    def _forward_live(self, batch, details=False):
        if any(hasattr(batch, name) for name in EMBEDDING_CACHES):
            raise ValueError('research_forward_cached_embedding_forbidden:call_strip_embedding_caches')
        # 这些只含结构的memo属于具体registry/mode，不能在两个配对模型间共享。
        for name in LAYOUT_CACHES:
            if hasattr(batch, name):
                delattr(batch, name)
        result = super().forward(batch, details=True)
        if not details:
            return result['quantiles']
        h = self.structure_embedding(batch)
        pooled = global_mean_pool(h, batch.batch)
        features = torch.cat([h, pooled[batch.batch], batch.physics.reshape(-1, 8)[batch.batch],
            batch.cond.reshape(-1, 4)[batch.batch],
            batch.condition_vector.reshape(-1, CONDITION_DIM)[batch.batch],
            batch.local_environment.float(), batch.acyl_environment.float()], 1)
        node_sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * self.research_sigma_head(features)[:, 0].sigmoid()
        total = batch.num_graphs * MAX_GROUPS
        dense_mass = result['values'].new_zeros(total)
        dense_centers = result['values'].new_zeros(total)
        dense_sigma = result['values'].new_zeros(total)
        component_group = torch.full_like(result['groups'], -1)
        known_components = batch.equivalence_known[result['groups']]
        ids = batch.equivalence_ids
        active = ids >= 0
        parents = torch.arange(len(batch.x), device=h.device)[:, None].expand_as(ids)[active]
        gid = ids[active] + batch.batch[parents] * MAX_GROUPS
        if not torch.equal(result['owners'][known_components], parents):
            raise ValueError('research_forward_preserved_component_order_changed')
        component_sigma = node_sigma[result['owners']]
        if len(gid):
            weights = result['weights'][known_components]
            dense_mass = dense_mass.index_add(0, gid, weights)
            dense_centers = dense_centers.index_add(0, gid, result['values'][known_components] * weights)
            dense_sigma = dense_sigma.index_add(0, gid, component_sigma[known_components] * weights)
            dense_centers = dense_centers / dense_mass.clamp_min(1)
            dense_sigma = dense_sigma / dense_mass.clamp_min(1)
            component_sigma = component_sigma.clone()
            component_sigma[known_components] = dense_sigma[gid]
            component_group[known_components] = gid
        mask = dense_mass > 0
        shape = (batch.num_graphs, MAX_GROUPS)
        total_mass = result['values'].new_zeros(batch.num_graphs).index_add(0, result['groups'], result['weights'])
        augmented = dict(result, group_centers=dense_centers.reshape(shape), group_Hmass=dense_mass.reshape(shape),
            group_sigma=dense_sigma.reshape(shape), group_trust=mask.to(h.dtype).reshape(shape),
            group_mask=mask.reshape(shape), component_sigma=component_sigma,
            component_group=component_group, layout_known=batch.equivalence_known,
            total_Hmass=total_mass, ungrouped_Hmass=total_mass - dense_mass.reshape(shape).sum(1),
            sigma_units='ppm/10', trust_semantics='registry_layout_available_only')
        sigma = quantile_sigmas(augmented, batch.num_graphs)
        return dict(augmented, quantile_sigma=sigma, quantile_sigma_ppm=10 * sigma)


def load_checkpoint_model(path, *, device='cpu', equivalence_mode=None, init_seed=317, trainable=False):
    """返回新模型；model.research_config可随新bank保存，未知checkpoint字段拒绝。"""
    payload = torch.load(Path(path), map_location='cpu', weights_only=False)
    config = dict(payload['args'])
    if config.get('model') != 'preserved_group_atom':
        raise ValueError('research_forward_requires_preserved_checkpoint')
    if config.get('research_forward_version', FORWARD_VERSION) != FORWARD_VERSION:
        raise ValueError('research_forward_unsupported_checkpoint_version')
    if any(name.startswith('research_sigma_head.') for name in payload['state_dict']) and 'research_forward_version' not in config:
        raise ValueError('research_forward_sigma_checkpoint_without_version')
    if equivalence_mode is not None:
        if equivalence_mode not in ('physical', 'local'):
            raise ValueError('research_forward_invalid_equivalence_mode')
        config['equivalence_mode'] = equivalence_mode
    config.update(use_frozen_embedding=False, research_forward_version=FORWARD_VERSION)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(init_seed)
        model = ResearchForwardSpecGNN(config['nucleus'], config['width'], config['layers'],
            config['condition_mode'], config['response_mode'], config['context_depth'], config['message_mode'],
            config['equivalence_registry_path'], config['equivalence_registry_sha256'], config['equivalence_mode'],
            domain_conditioning=config.get('domain_conditioning'), joint_reporting=config.get('joint_reporting', False),
            sigma_min=config.get('research_sigma_min', .002), sigma_max=config.get('research_sigma_max', .15),
            sigma_initial=config.get('research_sigma_initial', .025))
    missing, unexpected = model.load_state_dict(payload['state_dict'], strict=False)
    is_research = 'research_forward_version' in payload['args']
    if unexpected or any(not name.startswith('research_sigma_head.') for name in missing) or (is_research and missing):
        raise ValueError('research_forward_incompatible_checkpoint:' + repr((missing, unexpected)))
    for name in ('count_decode', 'amide_gap_scale'):
        if name in config:
            setattr(model, name, config[name])
    config.update(research_sigma_min=model.sigma_min, research_sigma_max=model.sigma_max,
                  research_sigma_initial=config.get('research_sigma_initial', .025))
    model.research_config = config
    configure_forward_training(model, trainable)
    return model.to(device)
