"""功能概述：在同一个验证集合执行光谱到 graph 或 spectrum 候选的结构检索评估。
主要职责：validate_retrieval 汇总 embeddings，分块稳定排序并计算 exact recall、Morgan
Tanimoto 和 top1 重原子数差；exclude_identity 屏蔽全部同 canonical identity 候选。
代码库关系：由训练入口传入无增强 validation loader；v2 使用模型 encode_graph，v1 使用
同一原始 spectrum 作为候选，不需要 graph index。结构标签仅参与指标，不输入查询编码器。
输入输出与副作用：支持短候选集，空集合或任何查询没有可用候选时报错；始终恢复模型训练状态。
"""
import torch


@torch.no_grad()
def validate_retrieval(model, dataloader, device, exclude_identity=False,
                       candidate_modality="graph") -> dict[str, float]:
    if candidate_modality not in ("graph", "spectrum"):
        raise ValueError("candidate_modality must be graph or spectrum")
    was_training = model.training
    model.eval()
    queries, candidates, fingerprints, identities, sizes = [], [], [], [], []
    try:
        for batch in dataloader:
            spectrum = batch["spectrum"].to(device)
            encode = model.encode_spectrum if hasattr(model, "encode_spectrum") else model
            query = encode(spectrum)
            candidate = (model.encode_graph(batch["graph"].to(device))
                         if candidate_modality == "graph" else query)
            if not torch.isfinite(query).all() or not torch.isfinite(candidate).all():
                raise ValueError("non-finite validation embedding")
            queries.append(query.cpu())
            candidates.append(candidate.cpu())
            fingerprints.append(batch["fingerprint"].cpu().bool())
            identities.append(batch["identity_id"].cpu())
            sizes.append(batch["heavy_atoms"].cpu())
        if not queries:
            raise ValueError("validation dataloader is empty")
        query, candidate = torch.cat(queries), torch.cat(candidates)
        bits, ids, heavy = torch.cat(fingerprints).float(), torch.cat(identities), torch.cat(sizes)
        count = len(ids)
        if not count:
            raise ValueError("validation dataloader is empty")
        totals = dict(top1_tanimoto=0., best5_tanimoto=0., exact_recall_at_1=0.,
                      exact_recall_at_5=0., mean_heavy_atom_gap=0.)
        for start in range(0, count, 256):
            stop = min(count, start + 256)
            scores = query[start:stop] @ candidate.T
            eligible = ids[start:stop, None] != ids[None, :] if exclude_identity else torch.ones_like(scores, dtype=torch.bool)
            if not eligible.any(dim=1).all():
                raise ValueError("validation query has no eligible candidates")
            scores = scores.masked_fill(~eligible, -torch.inf)
            ranked = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :min(5, count)]
            valid = eligible.gather(1, ranked)
            selected = bits[ranked]
            intersection = (bits[start:stop, None] * selected).sum(-1)
            union = bits[start:stop].sum(-1, keepdim=True) + selected.sum(-1) - intersection
            similarity = torch.where(union > 0, intersection / union.clamp_min(1), torch.ones_like(union))
            similarity = similarity.masked_fill(~valid, 0.)
            exact = (ids[ranked] == ids[start:stop, None]) & valid
            totals["top1_tanimoto"] += similarity[:, 0].sum().item()
            totals["best5_tanimoto"] += similarity.max(dim=1).values.sum().item()
            totals["exact_recall_at_1"] += exact[:, 0].sum().item()
            totals["exact_recall_at_5"] += exact.any(dim=1).sum().item()
            totals["mean_heavy_atom_gap"] += (heavy[ranked[:, 0]] - heavy[start:stop]).abs().sum().item()
        return {key: value / count for key, value in totals.items()}
    finally:
        model.train(was_training)
