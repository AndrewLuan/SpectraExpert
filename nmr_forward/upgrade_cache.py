"""功能概述：从已有纯二维峰表缓存增量重建18维立体化学特征，避免重读原始数据集。
主要职责：file_sha256 流式计算文件指纹；upgrade 原地更新内存记录的x/adj/hcount，
逐条确认氢计数与旧缓存完全相同，输出到必须尚不存在的新目录；main 提供CLI。
代码库关系：调用prepare.features的CIP R/S与E/Z实现，输入prepare.py的cache.pt，
保留原积分峰表、分位数、split、record_id及全部其他字段，供train.py直接读取。
副作用：读取可信本地torch缓存（含numpy对象，需要weights_only=False），只写新目录；
记录旧/新缓存SHA256、父audit与complete元信息、特征源码指纹及版本。失败不写complete；
不删除或覆盖旧缓存，不读取分子坐标，不检查或改动测试标签。命令形如：
python nmr_forward/upgrade_cache.py --cache OLD/cache.pt --out NEW_DIRECTORY。
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import time

import numpy as np
import torch
from rdkit import Chem, rdBase

try:
    from .prepare import features
except ImportError:
    from prepare import features


def file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def upgrade(cache: str | pathlib.Path, out: str | pathlib.Path) -> dict:
    cache, out = pathlib.Path(cache).resolve(), pathlib.Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    start = time.time()
    before = cache.stat()
    parent_sha = file_sha256(cache)
    records = torch.load(cache, map_location="cpu", weights_only=False)
    dimensions = collections.Counter()
    splits = collections.Counter()
    for index, item in enumerate(records):
        old_hcount = item["hcount"]
        dimensions[str(item["x"].shape[-1])] += 1
        mol = Chem.MolFromSmiles(item["smiles"])
        if mol is None:
            raise ValueError(f"Invalid cached SMILES at record {index}")
        x, adj, hcount = features(mol)
        if not np.array_equal(hcount, old_hcount):
            raise ValueError(f"Hydrogen count or atom order changed at record {index}")
        if int(hcount.sum()) != item["protons"]:
            raise ValueError(f"Proton total inconsistent at record {index}")
        if x.shape[0] != item["x"].shape[0] or not np.array_equal(adj, item["adj"]):
            raise ValueError(f"Connectivity changed at record {index}")
        item["x"], item["adj"], item["hcount"] = x, adj, hcount
        splits[item["split"]] += 1
        if (index + 1) % 2000 == 0:
            print(json.dumps({"upgraded": index + 1, "seconds": time.time() - start}), flush=True)
    after = cache.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("Source cache changed during upgrade")
    temporary = out / "cache.tmp.pt"
    torch.save(records, temporary)
    temporary.replace(out / "cache.pt")
    audit = {
        "operation": "upgrade_2d_features_cip_ez_v2",
        "feature_dim": 18,
        "feature_schema": {"9": "CIP_R", "10": "CIP_S", "16": "attached_E_double_bond", "17": "attached_Z_double_bond"},
        "source_cache": str(cache), "source_cache_sha256": parent_sha,
        "source_feature_dimensions": dict(dimensions),
        "source_metadata": {}, "records": len(records), "splits": dict(splits),
        "preserved": ["all_targets", "split", "record_id", "smiles", "hydrogen_counts", "connectivity", "other_record_fields"],
        "rdkit_version": rdBase.rdkitVersion, "torch_version": torch.__version__,
        "prepare_source_sha256": file_sha256(pathlib.Path(__file__).with_name("prepare.py")),
        "upgrade_source_sha256": file_sha256(pathlib.Path(__file__)),
        "cache_sha256": file_sha256(out / "cache.pt"),
        "seconds": time.time() - start,
    }
    for name in ("audit.json", "complete.json"):
        path = cache.parent / name
        if path.exists():
            audit["source_metadata"][name] = {"sha256": file_sha256(path), "content": json.loads(path.read_text())}
    (out / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False))
    completion = {"records": len(records), "cache_bytes": (out / "cache.pt").stat().st_size,
                  "cache_sha256": audit["cache_sha256"], "source_cache_sha256": parent_sha, "feature_dim": 18}
    (out / "complete.json").write_text(json.dumps(completion, indent=2))
    print(json.dumps(completion), flush=True)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    upgrade(args.cache, args.out)


if __name__ == "__main__":
    main()
