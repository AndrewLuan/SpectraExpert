# 结构感知的光谱—分子图检索实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个训练时联合 Raman 光谱与分子图、推理时只输入光谱的结构感知检索器，使 Top-K 候选的 GED 与 Morgan Tanimoto 显著优于现有 spectral-only SimCLR/KNN。

**Architecture:** 保留 SpectrumTransformer 作为查询 encoder，新增与 GED/Tanimoto 特征对齐的两层 edge-conditioned GIN 作为候选 encoder，通过精确跨模态对齐、光谱增强一致性、Morgan 结构关系软监督和光谱侧描述符辅助任务共同训练。候选索引由 SMILES 图 embedding 构建，Stage 1 只用未知光谱生成 query embedding；旧 spectral-only checkpoint/index 以显式 legacy 模式保留。

**Tech Stack:** Python 3.11、PyTorch、NumPy、Pandas、PyArrow、SciPy、RDKit、NetworkX、pytest；首版图网络为纯 PyTorch，不依赖 PyTorch Geometric。

**Spec:** `docs/superpowers/specs/2026-09-05-structure-aware-spectral-retrieval-design.md`

## Global Constraints

- 推理查询只允许传入一维 Raman/IR 光谱；真实 SMILES、分子式、分子量、原子数和元素种类不得进入查询路径。
- 图为二维重原子图；H 仅以 `total_hydrogen_count` 节点特征表示。
- GED 核心节点标签为 atomic number；GED 核心边标签必须是 `SINGLE/DOUBLE/TRIPLE/AROMATIC/OTHER` 离散类别，禁止 `int(GetBondTypeAsDouble())`。
- Morgan fingerprint 固定为 radius=2、2048 bits、`useBondTypes=True`、`useChirality=False`；配置进入 cache、checkpoint 和 index metadata。
- 图 encoder 主配置只做两轮消息传递，并 readout 0-hop、1-hop、2-hop；每层拼接 mean pooling 与 sum pooling。
- 首版只使用纯 PyTorch 和 `index_add_` 做图聚合，不增加 PyTorch Geometric 或需要编译扩展的依赖。
- 训练数据必须以 `mol_id` 一对一 join，重复、缺失、非法 SMILES、非有限光谱和长度不匹配均 fail fast；不得依赖 spectra/SMILES 的行顺序。
- `legacy_ged_v1` 只复现历史结果；新主指标为 `chemical_ged_v2` 与固定配置的 Morgan Tanimoto，二者不得混列。
- DDP 的 cross-modal loss 使用全局负样本；相同 canonical SMILES 是 multi-positive，不得互为负样本；loss 不依赖构造时固定 batch size。
- 每个新增或修改的 Python/shell 源文件都必须在 shebang/编码声明之后包含中文头部注释，说明职责、公开接口、依赖关系、输入输出和副作用。
- 所有生产代码遵循测试先行：先运行目标测试并确认因缺少新行为而失败，再实现最小代码并确认通过。
- 本机不读取或遍历大型数据目录，只运行手工 SMILES、随机小张量和临时小 Parquet 测试。
- 服务器代码同步先执行 `rsync -an`，正式同步禁止 `--delete`，使用 `--max-size=20m` 并排除数据、二进制、输出、checkpoint、`.git` 和虚拟环境。
- JJ 服务器实验前重新运行 `nvidia-smi`，根据当时空闲显存设置 `CUDA_VISIBLE_DEVICES`；长任务使用 tmux，检查进程、日志和首轮输出。
- 用户仅授权读取 `/data/SpecGPT/data/theory_data_tiny/Raman_QME14S.jsonl` 与 `Raman_fg26.jsonl`；不得扩展遍历 `/data/SpecGPT` 的其他数据。
- 两个 tiny 文件只用于端到端 pilot：共 400 条唯一 SMILES；QME14S 为 3500 点、500–4000 cm⁻¹，fg26 为 4000 点、1–4000 cm⁻¹；统一插值到 3501 点、500–4000 cm⁻¹。
- pilot 结果必须标记为小样本可行性结果，不得直接写成论文最终性能结论。

---

## 文件职责映射

- `spectraexpert/data/structure_features.py`：固定结构 schema、Morgan fingerprint、描述符和 metadata。
- `spectraexpert/data/molecular_graph.py`：SMILES 图化、图 dataclass、变长图 batching。
- `spectraexpert/data/structure_dataset.py`：按 `mol_id` 对齐的多模态 dataset、collate、scaffold split 与 feature cache。
- `spectraexpert/models/graph_encoder.py`：两层 edge-conditioned GIN 与 0/1/2-hop multi-scale readout。
- `spectraexpert/models/multimodal_retriever.py`：光谱/图双 encoder、projection heads 与辅助预测头。
- `spectraexpert/utils/structure_losses.py`：动态 NT-Xent、multi-positive CLIP、Tanimoto 软标签和组合 loss。
- `spectraexpert/utils/distributed.py`：保留本 rank 梯度的 DDP embedding/label gather。
- `spectraexpert/inference.py`：legacy 与 v2 checkpoint 加载、纯光谱查询编码。
- `spectraexpert/retrieval/graph_index.py`：版本化图索引保存、加载、校验与 cosine 检索。
- `spectraexpert/train_structure_aware.py`：训练、验证、checkpoint 和消融 CLI。
- `spectraexpert/eval/metrics.py`：legacy/corrected GED、nGED、Tanimoto 与结构检索统计。
- `spectraexpert/eval/evaluate_structure_retrieval.py`：统一 baseline/v2 评测和 paired bootstrap。
- `scripts/prepare_raman_jsonl.py`：只读指定 Raman JSONL、统一波数轴并写标准 Parquet。
- `scripts/build_graph_index.py`：从 checkpoint 与 SMILES 构建 v2 candidate index。
- `tests/`：不依赖真实数据的单元和端到端测试。

### Task 1: 建立可复现的包与测试基线

**Files:**
- Create: `pytest.ini`
- Create: `tests/__init__.py`
- Create: `tests/test_package_imports.py`
- Create: `spectraexpert/inference.py`
- Modify: `setup.py`
- Modify: `requirements.txt`
- Modify: `spectraexpert/train_contrastive.py`
- Modify: `spectraexpert/data/spectral_dataset.py`
- Modify: `spectraexpert/run_pipeline.py`
- Modify: `spectraexpert/eval/evaluate.py`
- Modify: `spectraexpert/pipeline/stage1_retrieval.py`
- Modify: `spectraexpert/pipeline/stage2_peak_analysis.py`
- Modify: `spectraexpert/pipeline/stage3_llm_reasoning.py`
- Modify: `spectraexpert/pipeline/knowledge/peak_matcher.py`
- Modify: `spectraexpert/pipeline/prompts/stage3_direct_identification.py`

**Interfaces:**
- Consumes: 现有 `SpectrumTransformer` checkpoint 字典和 `(D,)`/`(N,D)` NumPy embedding。
- Produces: `SpectralEncoder(checkpoint_path: str, device: str | None = None)`、`encode_spectrum(spectrum: np.ndarray) -> np.ndarray`、`retrieve(query: np.ndarray, database: np.ndarray, topk: int) -> tuple[np.ndarray, np.ndarray]`；统一可导入的 `spectraexpert` 包。

- [ ] **Step 1: 写包导入与 legacy retrieval 的失败测试**

```python
# tests/test_package_imports.py
"""中文头部：验证公开模块可从安装后的 spectraexpert 包导入，并锁定 legacy cosine retrieval 契约。"""
import importlib
import numpy as np


def test_public_modules_import_without_legacy_proj_namespace():
    for name in (
        "spectraexpert.train_contrastive",
        "spectraexpert.pipeline",
        "spectraexpert.eval.metrics",
        "spectraexpert.inference",
    ):
        importlib.import_module(name)


def test_retrieve_returns_descending_cosine_neighbors():
    from spectraexpert.inference import SpectralEncoder
    query = np.array([1.0, 0.0], dtype=np.float32)
    database = np.array([[0.0, 1.0], [1.0, 0.0], [0.8, 0.2]], dtype=np.float32)
    indices, scores = SpectralEncoder.retrieve(query, database, topk=2)
    assert indices.tolist() == [1, 2]
    assert scores[0] >= scores[1]
```

- [ ] **Step 2: 安装本地开发依赖并验证测试因错误 namespace/缺失 wrapper 失败**

Run:

```bash
source /Users/andrewluan/.venvs/spectrexpert-local/bin/activate
python -m pip install -e '.[dev]'
pytest tests/test_package_imports.py -v
```

Expected: FAIL，原因是 `proj.spectra_expert`/`spectra_expert` 不存在或 `spectraexpert.inference` 尚不存在；不能接受依赖下载失败作为 RED。

- [ ] **Step 3: 固定依赖并统一 package imports**

`setup.py` 的 `install_requires` 必须包含 `networkx>=3.1`、`pyyaml>=6.0`、`tensorboard>=2.14`；新增：

```python
extras_require={"dev": ["pytest>=8.0"]}
```

`requirements.txt` 同步声明 `torch>=2.0`、`networkx>=3.1`、`pyyaml>=6.0`、`tensorboard>=2.14`、`pytest>=8.0`。将所有 `spectra_expert.*` 与 `proj.spectra_expert.*` 内部 import 改为 `spectraexpert.*`；`stage1_retrieval.py` 中缺失的 `analyze_spectra.SpectralEncoder` 具体改为 `from spectraexpert.inference import SpectralEncoder`。每个被修改的源文件把原英文模块 docstring 更新为符合 Global Constraints 的中文头部说明。

- [ ] **Step 4: 实现最小 legacy `SpectralEncoder` wrapper**

`spectraexpert/inference.py` 定义上述接口。`retrieve` 必须检查二维数据库、维度一致、`topk > 0`，对 query/database 做安全 L2 normalization，用稳定降序排序返回 `int64` indices 和 `float32` scores。checkpoint 加载接受 `model_state_dict` 或裸 state dict，并从 checkpoint `args` 恢复旧 SpectrumTransformer 参数；非有限输入直接 `ValueError`。

- [ ] **Step 5: 运行导入测试与静态 namespace 检查**

Run:

```bash
pytest tests/test_package_imports.py -v
rg -n 'from (proj\.)?spectra_expert|import (proj\.)?spectra_expert' spectraexpert
```

Expected: tests PASS；`rg` 无匹配。

- [ ] **Step 6: 提交基线修复**

```bash
git add pytest.ini tests setup.py requirements.txt spectraexpert
git commit -m "fix: restore importable SpectraExpert package"
```

### Task 2: 实现 GED/Tanimoto 对齐的结构特征与图 batching

**Files:**
- Create: `spectraexpert/data/structure_features.py`
- Create: `spectraexpert/data/molecular_graph.py`
- Create: `tests/data/test_structure_features.py`
- Create: `tests/data/test_molecular_graph.py`
- Modify: `spectraexpert/data/__init__.py`

**Interfaces:**
- Consumes: canonical SMILES、`mol_id`；RDKit `Chem.Mol`。
- Produces: `StructureFeatureConfig`、`DescriptorNormalizer`、`MoleculeGraph`、`GraphBatch`、`featurize_smiles()`、`collate_graphs()`、`morgan_fingerprint()`、`molecular_descriptors()`。

- [ ] **Step 1: 写结构 schema 与图化失败测试**

```python
def test_benzene_uses_distinct_aromatic_bond_category():
    graph = featurize_smiles("c1ccccc1", mol_id=7)
    assert graph.node_features.shape == (6, 7)
    assert graph.edge_index.shape == (2, 12)
    assert set(graph.edge_features[:, 0].tolist()) == {BOND_TYPE_TO_ID["AROMATIC"]}
    assert graph.fingerprint.shape == (2048,)
    assert graph.metadata["morgan_radius"] == 2
    assert graph.metadata["use_chirality"] is False


def test_collate_offsets_edges_and_preserves_graph_membership():
    batch = collate_graphs([
        featurize_smiles("CC", mol_id=1),
        featurize_smiles("O", mol_id=2),
    ])
    assert batch.node_features.shape[0] == 3
    assert batch.graph_index.tolist() == [0, 0, 1]
    assert batch.edge_index.max().item() < 3
    assert batch.mol_ids.tolist() == [1, 2]
```

同时测试 `C1CCCCC1` 只产生 `SINGLE` 边、`C=C` 产生 `DOUBLE`、非法 SMILES 抛出 `ValueError`、单原子图产生 `(2,0)` edge index。

`test_structure_features.py` 还要验证 `DescriptorNormalizer.fit(train_descriptors)` 只用传入训练 tensor 计算逐列 mean/std，`transform` 后训练 tensor 均值约为 0；std 小于 `1e-8` 的列使用 1.0，避免除零。

- [ ] **Step 2: 运行结构测试确认因模块缺失失败**

Run: `pytest tests/data/test_structure_features.py tests/data/test_molecular_graph.py -v`

Expected: FAIL with `ModuleNotFoundError`。

- [ ] **Step 3: 实现固定结构特征配置**

```python
@dataclass(frozen=True)
class StructureFeatureConfig:
    schema_version: str = "metric_aligned_graph_v1"
    morgan_radius: int = 2
    morgan_bits: int = 2048
    use_bond_types: bool = True
    use_chirality: bool = False

    def to_metadata(self) -> dict[str, object]:
        return dataclasses.asdict(self)
```

`structure_features.py` 暴露 `BOND_TYPE_TO_ID = {"OTHER": 0, "SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3, "AROMATIC": 4}`。节点七列依次为 atomic number（1–118，其他为 0）、degree（截断到 0–8）、total valence（截断到 0–12）、formal charge bucket（先截断到 -3–3 再加 3）、total H count（截断到 0–8）、aromatic（0/1）、ring（0/1）；边三列依次为 bond type、conjugated（0/1）、ring（0/1）。描述符顺序固定为 `C,N,O,F,H,heavy_atoms,molecular_weight,ring_count,aromatic_ring_count,aromatic_atoms,single_bonds,double_bonds,triple_bonds,aromatic_bonds,formal_charge`。

```python
@dataclass(frozen=True)
class DescriptorNormalizer:
    mean: torch.FloatTensor
    std: torch.FloatTensor

    @classmethod
    def fit(cls, values: torch.Tensor) -> "DescriptorNormalizer":
        mean = values.float().mean(dim=0)
        std = values.float().std(dim=0, unbiased=False)
        return cls(mean=mean, std=torch.where(std < 1e-8, torch.ones_like(std), std))

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        return (values.float() - self.mean) / self.std
```

- [ ] **Step 4: 实现 dataclass、SMILES 图化和无 padding batching**

```python
@dataclass(frozen=True)
class MoleculeGraph:
    node_features: torch.LongTensor
    edge_index: torch.LongTensor
    edge_features: torch.LongTensor
    fingerprint: torch.BoolTensor
    descriptors: torch.FloatTensor
    mol_id: int
    canonical_smiles: str
    metadata: dict[str, object]


@dataclass(frozen=True)
class GraphBatch:
    node_features: torch.LongTensor
    edge_index: torch.LongTensor
    edge_features: torch.LongTensor
    graph_index: torch.LongTensor
    fingerprints: torch.BoolTensor
    descriptors: torch.FloatTensor
    mol_ids: torch.LongTensor
    canonical_smiles: tuple[str, ...]

    def to(self, device: torch.device | str) -> "GraphBatch":
        return dataclasses.replace(
            self,
            node_features=self.node_features.to(device),
            edge_index=self.edge_index.to(device),
            edge_features=self.edge_features.to(device),
            graph_index=self.graph_index.to(device),
            fingerprints=self.fingerprints.to(device),
            descriptors=self.descriptors.to(device),
            mol_ids=self.mol_ids.to(device),
        )
```

`featurize_smiles` 先 canonicalize，再构造双向边；禁止用 bond order 浮点转整数。`collate_graphs` 累加节点 offset 并支持无键分子。

- [ ] **Step 5: 运行结构测试并提交**

Run: `pytest tests/data/test_structure_features.py tests/data/test_molecular_graph.py -v`

Expected: PASS。

```bash
git add spectraexpert/data tests/data
git commit -m "feat: add metric-aligned molecular graph features"
```

### Task 3: 版本化 GED、nGED 与 Tanimoto 计算

**Files:**
- Modify: `spectraexpert/eval/metrics.py`
- Create: `tests/eval/test_structure_metrics.py`

**Interfaces:**
- Consumes: 两个 SMILES 或两个 boolean fingerprint tensor。
- Produces: `graph_edit_distance(smiles_a, smiles_b, timeout=10.0, include_h=False, version="chemical_ged_v2")`、`normalized_graph_edit_distance()`、`pairwise_tanimoto()`；保留 `legacy_ged_v1`。

- [ ] **Step 1: 写芳香键回归测试与 nGED 测试**

```python
def test_chemical_ged_distinguishes_benzene_from_cyclohexane():
    assert graph_edit_distance("c1ccccc1", "C1CCCCC1", version="legacy_ged_v1") == 0.0
    assert graph_edit_distance("c1ccccc1", "C1CCCCC1", version="chemical_ged_v2") > 0.0


def test_chemical_ged_returns_minimum_completed_search_value(monkeypatch):
    monkeypatch.setattr(nx, "optimize_graph_edit_distance", lambda *args, **kwargs: iter([5.0, 3.0, 2.0]))
    assert graph_edit_distance("CC", "CO", version="chemical_ged_v2") == 2.0
    assert graph_edit_distance("CC", "CO", version="legacy_ged_v1") == 5.0


def test_normalized_ged_uses_nodes_and_edges_denominator():
    raw = graph_edit_distance("CC", "CO", version="chemical_ged_v2")
    normalized = normalized_graph_edit_distance("CC", "CO", version="chemical_ged_v2")
    assert normalized == pytest.approx(raw / 6.0)


def test_pairwise_tanimoto_is_symmetric_with_unit_diagonal():
    fps = torch.tensor([[1, 0, 1], [1, 1, 0]], dtype=torch.bool)
    sim = pairwise_tanimoto(fps)
    assert torch.allclose(sim, sim.T)
    assert torch.allclose(sim.diag(), torch.ones(2))
    assert sim[0, 1].item() == pytest.approx(1 / 3)
```

- [ ] **Step 2: 运行测试确认 legacy 芳香键冲突被捕获**

Run: `pytest tests/eval/test_structure_metrics.py -v`

Expected: FAIL，因为尚无 version/nGED/pairwise API。

- [ ] **Step 3: 实现两个 GED schema 与 nGED**

`smiles_to_nx(smiles, version)` 在 legacy 模式保存旧整数 bond order；在 v2 保存 `BOND_TYPE_TO_ID` 的类别 ID。`graph_edit_distance` 明确设置 node/edge substitution、insert、delete cost 为 0/1。legacy 模式保留 `optimize_graph_edit_distance` generator 的第一个上界；v2 必须迭代完成并返回最小值，timeout 时返回 `None`，不得把中间上界报告为精确 GED。nGED 分母严格为 `|Vq|+|Vc|+|Eq|+|Ec|`，任一图非法、GED timeout 或分母 0 时返回 `None`。

- [ ] **Step 4: 实现可微 tensor Tanimoto 并修复 aggregate rank 索引**

```python
def pairwise_tanimoto(fingerprints: torch.Tensor) -> torch.Tensor:
    bits = fingerprints.float()
    intersection = bits @ bits.T
    union = bits.sum(1, keepdim=True) + bits.sum(1).unsqueeze(0) - intersection
    return torch.where(union > 0, intersection / union.clamp_min(1), torch.ones_like(union))
```

同时修复现有 `compute_retrieval_metrics` 使用过滤后的 `valid_ged` 下标映射候选 rank 的错误：保留 `(original_index, ged)` pair 再找最小值。

- [ ] **Step 5: 运行测试和提交**

Run: `pytest tests/eval/test_structure_metrics.py -v`

```bash
git add spectraexpert/eval/metrics.py tests/eval/test_structure_metrics.py
git commit -m "feat: version molecular structure metrics"
```

### Task 4: 实现两跳 GIN 与双编码器模型

**Files:**
- Create: `spectraexpert/models/graph_encoder.py`
- Create: `spectraexpert/models/multimodal_retriever.py`
- Create: `tests/models/test_graph_encoder.py`
- Create: `tests/models/test_multimodal_retriever.py`
- Modify: `spectraexpert/models/spectrum_encoder.py`
- Modify: `spectraexpert/models/__init__.py`

**Interfaces:**
- Consumes: `(B,L)` 光谱 tensor 与 `GraphBatch`。
- Produces: `SpectrumTransformer.encode_backbone()`、`GraphEncoderOutput`、`MultimodalOutput`、`StructureAwareRetriever.forward()`。

- [ ] **Step 1: 写 0/1/2-hop readout 和双编码器 shape 失败测试**

```python
def test_graph_encoder_exposes_three_scales_and_unit_embedding():
    batch = collate_graphs([featurize_smiles("CCO", 1), featurize_smiles("c1ccccc1", 2)])
    model = GraphEncoder(hidden_dim=32, projection_dim=16, num_layers=2)
    out = model(batch)
    assert len(out.node_states) == 3
    assert out.backbone.shape == (2, 32)
    assert out.embedding.shape == (2, 16)
    assert torch.allclose(out.embedding.norm(dim=1), torch.ones(2), atol=1e-5)


def test_multimodal_forward_needs_no_query_structure_on_encode_spectrum():
    spectrum_encoder = SpectrumTransformer(
        signal_length=65, patch_size=8, stride=4, embed_dim=32,
        depth=1, num_heads=4, projection_dim=16,
    )
    graph_encoder = GraphEncoder(hidden_dim=32, projection_dim=16, num_layers=2)
    model = StructureAwareRetriever(
        spectrum_encoder=spectrum_encoder,
        graph_encoder=graph_encoder,
        descriptor_dim=15,
        predict_fingerprint=False,
    )
    query = torch.randn(2, 65)
    embedding = model.encode_spectrum(query)
    assert embedding.shape == (2, 16)
```

- [ ] **Step 2: 运行测试确认模块/接口缺失**

Run: `pytest tests/models/test_graph_encoder.py tests/models/test_multimodal_retriever.py -v`

Expected: FAIL with missing modules/methods。

- [ ] **Step 3: 拆分 SpectrumTransformer backbone 与 projection**

新增：

```python
def encode_backbone(self, x: torch.Tensor) -> torch.Tensor:
    """返回 projection 前的 `(B, embed_dim)` pooled spectrum feature。"""

def project(self, features: torch.Tensor) -> torch.Tensor:
    """返回 L2-normalized `(B, projection_dim)` embedding。"""
```

旧 `forward(x)` 等价于 `project(encode_backbone(x))`，保证 legacy checkpoint key 不变。
构造函数同时保存只读属性 `self.embed_dim = embed_dim` 和 `self.projection_dim = projection_dim`，供双编码器建辅助头和 checkpoint schema 使用。

- [ ] **Step 4: 实现 edge-conditioned GIN 和 multi-scale readout**

`GraphEncoder(hidden_dim, projection_dim, num_layers=2, dropout=0.1)` 强制默认两层。每个 categorical field 使用独立 embedding；消息函数融合 source node 与 edge embedding；聚合只用 `index_add_`。对 h0/h1/h2 分别做 mean/sum graph pooling，将六个向量拼接后映射为 `backbone`，projection head 输出单位向量。

- [ ] **Step 5: 实现双编码器和辅助头**

```python
@dataclass
class MultimodalOutput:
    spectrum_view1_embedding: torch.Tensor
    spectrum_view2_embedding: torch.Tensor
    graph_embedding: torch.Tensor
    spectrum_view1_features: torch.Tensor
    spectrum_view2_features: torch.Tensor
    graph_features: torch.Tensor
    descriptor_prediction: torch.Tensor
    element_presence_logits: torch.Tensor
    fingerprint_logits: torch.Tensor | None


class StructureAwareRetriever(nn.Module):
    def __init__(self, spectrum_encoder: SpectrumTransformer, graph_encoder: GraphEncoder,
                 descriptor_dim: int = 15, predict_fingerprint: bool = False):
        super().__init__()
        self.spectrum_encoder = spectrum_encoder
        self.graph_encoder = graph_encoder
        self.descriptor_head = nn.Linear(spectrum_encoder.embed_dim, descriptor_dim)
        self.element_head = nn.Linear(spectrum_encoder.embed_dim, 5)
        self.fingerprint_head = (nn.Linear(spectrum_encoder.embed_dim, 2048)
                                 if predict_fingerprint else None)

    @classmethod
    def from_config(cls, config: dict[str, object]) -> "StructureAwareRetriever":
        spectrum_encoder = SpectrumTransformer(
            signal_length=int(config["signal_length"]),
            patch_size=int(config["patch_size"]),
            stride=int(config["stride"]),
            embed_dim=int(config["embed_dim"]),
            depth=int(config["depth"]),
            num_heads=int(config["num_heads"]),
            projection_dim=int(config["projection_dim"]),
        )
        graph_encoder = GraphEncoder(
            hidden_dim=int(config["graph_hidden_dim"]),
            projection_dim=int(config["projection_dim"]),
            num_layers=2,
        )
        return cls(
            spectrum_encoder=spectrum_encoder,
            graph_encoder=graph_encoder,
            descriptor_dim=int(config["descriptor_dim"]),
            predict_fingerprint=bool(config["predict_fingerprint"]),
        )

    def forward(self, spectrum_view1, spectrum_view2, graphs) -> MultimodalOutput:
        h1 = self.spectrum_encoder.encode_backbone(spectrum_view1)
        h2 = self.spectrum_encoder.encode_backbone(spectrum_view2)
        graph_output = self.graph_encoder(graphs)
        h_mean = (h1 + h2) / 2
        return MultimodalOutput(
            spectrum_view1_embedding=self.spectrum_encoder.project(h1),
            spectrum_view2_embedding=self.spectrum_encoder.project(h2),
            graph_embedding=graph_output.embedding,
            spectrum_view1_features=h1,
            spectrum_view2_features=h2,
            graph_features=graph_output.backbone,
            descriptor_prediction=self.descriptor_head(h_mean),
            element_presence_logits=self.element_head(h_mean),
            fingerprint_logits=(self.fingerprint_head(h_mean)
                                if self.fingerprint_head is not None else None),
        )

    def encode_spectrum(self, spectrum: torch.Tensor) -> torch.Tensor:
        return self.spectrum_encoder(spectrum)

    def encode_graph(self, graphs: GraphBatch) -> torch.Tensor:
        return self.graph_encoder(graphs).embedding

    def predict_auxiliary(self, spectrum: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        features = self.spectrum_encoder.encode_backbone(spectrum)
        fingerprint = (torch.sigmoid(self.fingerprint_head(features))
                       if self.fingerprint_head is not None else None)
        return self.descriptor_head(features), fingerprint
```

实例同时保存纯标量 `model_config`，键固定为 `signal_length,patch_size,stride,embed_dim,depth,num_heads,projection_dim,graph_hidden_dim,descriptor_dim,predict_fingerprint`；`to_config()` 返回该字典的副本。checkpoint 不序列化 module object，Task 8 必须只通过 `from_config(payload["model_config"])` 重建模型。

辅助头只读取光谱 backbone feature；`encode_spectrum` 签名中不得出现 SMILES、graph 或 descriptor。

- [ ] **Step 6: 运行模型测试和 legacy forward 回归测试后提交**

Run: `pytest tests/models -v tests/test_package_imports.py -v`

```bash
git add spectraexpert/models tests/models
git commit -m "feat: add spectrum-graph dual encoder"
```

### Task 5: 实现 multi-positive、结构关系和 DDP 全局 loss

**Files:**
- Create: `spectraexpert/utils/distributed.py`
- Create: `spectraexpert/utils/structure_losses.py`
- Create: `tests/utils/test_structure_losses.py`
- Create: `tests/utils/test_distributed_gather.py`
- Modify: `spectraexpert/utils/__init__.py`

**Interfaces:**
- Consumes: `MultimodalOutput`、boolean fingerprints、descriptor targets、canonical SMILES identity IDs。
- Produces: `StructureAwareObjective.forward(output: MultimodalOutput, fingerprints: torch.BoolTensor, descriptor_targets: torch.Tensor, element_presence: torch.Tensor, identity_ids: torch.LongTensor) -> dict[str, torch.Tensor]`，字典至少含 `total/cross/aug/struct/descriptor/presence/fingerprint`。

- [ ] **Step 1: 写动态 batch、multi-positive 和结构排序测试**

```python
def test_multi_positive_identical_structures_are_not_negatives():
    logits = torch.tensor([[4.0, 4.0, 0.0], [4.0, 4.0, 0.0], [0.0, 0.0, 4.0]], requires_grad=True)
    ids = torch.tensor([11, 11, 12])
    loss = multi_positive_cross_entropy(logits, ids, ids)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad[0, 1].item() < 0.0
    assert logits.grad[0, 2].item() > 0.0


def test_structure_loss_rewards_tanimoto_ordering():
    fingerprints = torch.tensor([[1,1,0,0], [1,1,1,0], [0,0,0,1]], dtype=torch.bool)
    aligned = torch.tensor([[1.,0.8,0.0],[0.8,1.,0.0],[0.,0.,1.]])
    reversed_similarity = torch.tensor([[1.,0.0,0.8],[0.0,1.,0.8],[0.8,0.8,1.]])
    assert relational_structure_loss(aligned, fingerprints) < relational_structure_loss(reversed_similarity, fingerprints)


def test_dynamic_nt_xent_accepts_last_short_batch():
    loss = dynamic_nt_xent(F.normalize(torch.randn(3, 8), dim=1), F.normalize(torch.randn(3, 8), dim=1), 0.1)
    assert loss.ndim == 0 and torch.isfinite(loss)
```

- [ ] **Step 2: 运行 loss 测试确认模块缺失**

Run: `pytest tests/utils/test_structure_losses.py -v`

- [ ] **Step 3: 实现 loss primitives 与组合 objective**

`multi_positive_cross_entropy` 使用 canonical identity equality 生成每行归一化 target。`relational_structure_loss` 用固定 Tanimoto soft targets，sg/ss/gg 三个分布按规格计算，ss/gg 排除对角自匹配。`StructureAwareObjective` 接受 `lambda_aug/lambda_struct/lambda_aux/lambda_fingerprint` 和温度参数；任一 loss 非有限立即抛出 `FloatingPointError`。

- [ ] **Step 4: 实现 DDP gather 并写两进程 CPU gloo 测试**

```python
def gather_with_local_grad(tensor: torch.Tensor) -> torch.Tensor:
    """单进程原样返回；DDP 下 gather 全 rank tensor，并保留本 rank slice 的梯度。"""

def gather_identity(values: torch.Tensor) -> torch.Tensor:
    """gather 无梯度 identity/fingerprint/target。"""
```

两进程测试各输入两行 embedding，断言 gather 后为四行，本 rank backward 后 local tensor 有非零 gradient。若各 rank batch size 不同，给出包含各 rank size 的 `RuntimeError`；训练 sampler 在 Task 7 保证 DDP rank batch 相等。

- [ ] **Step 5: 运行 loss/DDP 测试和提交**

Run: `pytest tests/utils/test_structure_losses.py tests/utils/test_distributed_gather.py -v`

```bash
git add spectraexpert/utils tests/utils
git commit -m "feat: add structure-aware contrastive objectives"
```

### Task 6: 实现严格数据对齐、scaffold split 与 Raman JSONL 转换

**Files:**
- Create: `spectraexpert/data/structure_dataset.py`
- Create: `scripts/prepare_raman_jsonl.py`
- Create: `scripts/cache_molecular_features.py`
- Create: `tests/data/test_structure_dataset.py`
- Create: `tests/scripts/test_prepare_raman_jsonl.py`
- Create: `tests/scripts/test_cache_molecular_features.py`
- Modify: `spectraexpert/data/__init__.py`

**Interfaces:**
- Consumes: `spectra.parquet`、`smiles.parquet`、`wavenumber.parquet`，或授权的两个 Raman JSONL。
- Produces: `StructureAwareSpectralDataset`、`structure_aware_collate()`、`scaffold_split()` 和标准三件套 Parquet。

- [ ] **Step 1: 写 mol_id join、重复 ID 和乱序输入失败/通过测试**

```python
def test_dataset_joins_spectra_and_smiles_by_mol_id(tmp_path):
    write_tiny_parquets(tmp_path, spectrum_ids=[2, 1], smiles_ids=[1, 2])
    dataset = StructureAwareSpectralDataset.from_directory(tmp_path, signal_length=5)
    item = dataset[0]
    expected = {1: "CC", 2: "O"}[item["mol_id"]]
    assert item["graph"].canonical_smiles == expected


def test_duplicate_mol_id_fails_before_training(tmp_path):
    write_tiny_parquets(tmp_path, spectrum_ids=[1, 1], smiles_ids=[1, 2])
    with pytest.raises(ValueError, match="duplicate mol_id"):
        StructureAwareSpectralDataset.from_directory(tmp_path, signal_length=5)
```

同一测试文件定义具体 helper：`write_tiny_parquets(root, spectrum_ids, smiles_ids)` 写 `wavenumber.parquet`（5 个点）、`spectra.parquet`（列为 `mol_id,w_0,w_1,w_2,w_3,w_4`）和 `smiles.parquet`；SMILES 映射固定为 `{1: "CC", 2: "O"}`，强度行固定为 `[id, id+1, id+2, id+3, id+4]`，不依赖仓库数据。

另测试缺失 ID、非法 SMILES、NaN、signal length mismatch、collate 输出 shape，以及同 scaffold 不跨 split。

同时测试 feature cache round-trip：缓存前后 node/edge/fingerprint/descriptor 完全一致；修改 `morgan_radius` 后加载旧缓存必须抛出包含 `feature metadata mismatch` 的 `ValueError`。

- [ ] **Step 2: 写 JSONL 插值和验证测试**

用临时 JSONL 构造一条 500–4000/3500 点和一条 1–4000/4000 点记录。调用：

```python
prepare_raman_jsonl(
    inputs=[qme_path, fg_path],
    output_dir=tmp_path / "prepared",
    start=500.0,
    stop=4000.0,
    points=3501,
)
```

断言输出 2 行、3501 个 intensity columns、共同波数轴端点准确、SMILES 唯一，并在非单调 x、x/y 不等长、NaN 时失败。

- [ ] **Step 3: 运行数据测试确认新接口缺失**

Run: `pytest tests/data/test_structure_dataset.py tests/scripts/test_prepare_raman_jsonl.py -v`

- [ ] **Step 4: 实现 dataset、collate 和 split**

Dataset 初始化时用 Pandas `merge(validate="one_to_one")` 按 `mol_id` 对齐；提供 cache 时逐行恢复 `MoleculeGraph`，否则只为当前 split 预先 featurize SMILES。`__getitem__` 返回：

```python
{
    "spectrum_view1": torch.FloatTensor,
    "spectrum_view2": torch.FloatTensor,
    "graph": MoleculeGraph,
    "descriptor_targets": torch.FloatTensor,
    "element_presence": torch.FloatTensor,
    "fingerprint": torch.BoolTensor,
    "identity_id": torch.LongTensor,
    "mol_id": int,
}
```

`structure_aware_collate` 把 graph 列交给 `collate_graphs`。`identity_id` 由排序后的唯一 canonical SMILES 词表确定，保证不同 worker/rank 一致。`scaffold_split(smiles, fractions=(0.8,0.1,0.1), seed=0)` 按 Bemis–Murcko scaffold group 分配；无环分子的 scaffold 为空时改用其 canonical SMILES 作为 group key，避免把所有无环分子错误放入同一 split。canonical identity 绝不跨 split。split manifest 是 JSON，包含 `schema_version="scaffold_split_v1"`、seed、源 SMILES SHA-256 以及 train/validation/test 三组 `mol_id`。

- [ ] **Step 5: 实现版本化 feature cache CLI**

`scripts/cache_molecular_features.py` 要求调用者显式提供 `--smiles-path` 和 `--output`，不扫描父目录，并以 batch 方式构造 Parquet。列固定为 `mol_id:int64`、`smiles_canonical:string`、`node_features:list<list<int16>>`、`edge_src:list<int32>`、`edge_dst:list<int32>`、`edge_features:list<list<int8>>`、`fingerprint:binary`（2048 bits pack 为 256 bytes）、`descriptors:list<float32>`。Parquet schema metadata 保存完整 `StructureFeatureConfig.to_metadata()`；加载时逐字段比较。

- [ ] **Step 6: 实现 streaming JSONL 转换 CLI**

CLI 只逐行读取显式传入文件，使用 `np.interp(np.linspace(500,4000,3501), x, y)`；`mol_id` 按输入顺序从 0 递增，输出 `smiles_canonical`。`--split-seed 0` 在输出目录同时写 `split_manifest.json`，供全部消融复用。写入前校验全部 400 条授权数据预期不作为硬编码，仅在运行报告中记录实际数量。

- [ ] **Step 7: 运行测试并提交**

Run: `pytest tests/data tests/scripts/test_prepare_raman_jsonl.py tests/scripts/test_cache_molecular_features.py -v`

```bash
git add spectraexpert/data scripts/prepare_raman_jsonl.py scripts/cache_molecular_features.py tests/data tests/scripts
git commit -m "feat: add aligned spectrum-graph dataset pipeline"
```

### Task 7: 实现训练、验证、checkpoint 与消融入口

**Files:**
- Create: `spectraexpert/train_structure_aware.py`
- Create: `spectraexpert/training/checkpoints.py`
- Create: `spectraexpert/training/validation.py`
- Create: `spectraexpert/training/__init__.py`
- Create: `tests/training/test_checkpoint_schema.py`
- Create: `tests/training/test_validation.py`
- Create: `tests/training/test_train_smoke.py`

**Interfaces:**
- Consumes: Task 4 模型、Task 5 loss、Task 6 dataset。
- Produces: schema `spectrum_graph_v2` checkpoint；`validate_retrieval()`；可运行 CLI `python -m spectraexpert.train_structure_aware`。

- [ ] **Step 1: 写 checkpoint schema 与 one-step training 失败测试**

```python
def test_checkpoint_records_all_compatibility_metadata(tmp_path):
    model = StructureAwareRetriever(
        spectrum_encoder=SpectrumTransformer(signal_length=65, patch_size=8, stride=4,
                                             embed_dim=32, depth=1, num_heads=4,
                                             projection_dim=16),
        graph_encoder=GraphEncoder(hidden_dim=32, projection_dim=16, num_layers=2),
        descriptor_dim=15,
        predict_fingerprint=False,
    )
    stats = DescriptorNormalizer(mean=torch.zeros(15), std=torch.ones(15))
    path = save_checkpoint(tmp_path / "model.pt", model, optimizer=None, epoch=2,
                           validation={"top1_tanimoto": 0.4}, descriptor_stats=stats)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["schema_version"] == "spectrum_graph_v2"
    assert payload["feature_config"]["morgan_radius"] == 2
    assert payload["feature_config"]["use_chirality"] is False
    assert payload["selection_metric"] == "top1_tanimoto"


def test_one_epoch_smoke_writes_best_checkpoint(tiny_dataset_dir, tmp_path):
    result = train(TrainConfig.tiny(data_dir=tiny_dataset_dir, output_dir=tmp_path))
    assert (tmp_path / "best.pt").is_file()
    assert result["global_steps"] > 0
```

`tiny_dataset_dir` fixture 在 `tmp_path` 写三件套 Parquet：SMILES 固定为 `C, CC, CCC, CCO, CCN, COC, CO, CN, C=C, C#N, O, N`，`mol_id=0..11`；波数轴为 `np.linspace(500, 4000, 65)`；每条光谱由 `np.random.default_rng(mol_id).normal(size=65).astype(np.float32)` 生成。split manifest 固定 train IDs `0..7`、validation IDs `8,9`、test IDs `10,11`，并包含与文件内容一致的 SHA-256。fixture 只位于 `tests/training/test_train_smoke.py`，不进入生产包。

- [ ] **Step 2: 运行训练测试确认模块缺失**

Run: `pytest tests/training -v`

- [ ] **Step 3: 实现 checkpoint schema 和兼容性 loader**

checkpoint 必须保存 `model.to_config()`、feature config、descriptor train statistics、model/optimizer/scheduler state、epoch/global step、validation metrics、selection metric、seed。loader 对 schema、projection dim、feature config mismatch 给出字段级错误；旧 checkpoint 只由 Task 1 legacy wrapper 加载。

- [ ] **Step 4: 实现结构检索验证**

```python
@torch.no_grad()
def validate_retrieval(model, dataloader, device, exclude_identity: bool = False) -> dict[str, float]:
    """用 spectrum query 对同一验证集合的 graph embeddings 排序，返回 exact recall 与 Tanimoto 指标。"""
```

返回键至少含 `top1_tanimoto`、`best5_tanimoto`、`exact_recall_at_1`、`exact_recall_at_5`、`mean_heavy_atom_gap`。当 `exclude_identity=True` 时屏蔽所有相同 canonical identity，用于评估结构近邻而非自匹配。

- [ ] **Step 5: 实现训练 CLI 与消融开关**

CLI 必须支持：`--training-mode spectral_only_v1|spectrum_graph_v2`、`--data-dir`、`--output-dir`、`--split-manifest`、`--epochs`、`--patience`、`--batch-size`、`--embed-dim`、`--depth`、`--num-heads`、`--projection-dim`、`--graph-hidden-dim`、`--lambda-aug`、`--lambda-struct`、`--lambda-aux`、`--lambda-fingerprint`、`--temperature`、`--seed`、`--ddp`、`--num-workers`、`--resume-from`、`--max-samples`。best checkpoint 只根据 validation `top1_tanimoto` 保存，训练 loss 不参与 best 判定。A/B/C/D/E 均由该入口读取同一 split manifest 和相同模型规模参数：A 的 candidate modality 为 spectrum 且只优化 dynamic NT-Xent；B–E 的 candidate modality 为 graph，并按 loss 权重启用结构组件。

DDP 使用 DistributedSampler；各 rank 当前 batch size 必须一致，embedding/fingerprint/identity 经 Task 5 gather 后计算全局 loss。日志记录各 loss 分量、验证指标、GPU memory 和 feature schema。

- [ ] **Step 6: 运行训练 smoke 与全部相关测试后提交**

Run: `pytest tests/training tests/models tests/utils tests/data -v`

```bash
git add spectraexpert/train_structure_aware.py spectraexpert/training tests/training
git commit -m "feat: train structure-aware spectral retriever"
```

### Task 8: 实现版本化图索引与 Stage 1 纯光谱查询

**Files:**
- Create: `spectraexpert/retrieval/__init__.py`
- Create: `spectraexpert/retrieval/graph_index.py`
- Create: `scripts/build_graph_index.py`
- Create: `tests/retrieval/test_graph_index.py`
- Create: `tests/pipeline/test_stage1_structure_retrieval.py`
- Modify: `spectraexpert/inference.py`
- Modify: `spectraexpert/pipeline/config.py`
- Modify: `spectraexpert/pipeline/stage1_retrieval.py`
- Modify: `spectraexpert/pipeline/data_types.py`

**Interfaces:**
- Consumes: v2 checkpoint、候选 SMILES Parquet、query spectrum。
- Produces: `GraphIndex`、`StructureAwareSpectralEncoder`、Stage 1 `retrieval_mode="spectrum_graph_v2"`。

- [ ] **Step 1: 写索引 round-trip、schema mismatch 和无结构查询测试**

```python
def test_graph_index_round_trip_and_checkpoint_validation(tmp_path):
    embeddings = np.eye(2, dtype=np.float32)
    mol_ids = np.array([1, 2], dtype=np.int64)
    smiles = np.array(["CC", "CO"])
    fingerprints = np.zeros((2, 2048), dtype=np.uint8)
    fingerprints[0, [0, 2]] = 1
    fingerprints[1, [0, 1]] = 1
    descriptors = np.zeros((2, 15), dtype=np.float32)
    descriptors[0, :2] = [2.0, 0.0]
    descriptors[1, :2] = [1.0, 1.0]
    metadata = StructureFeatureConfig().to_metadata()
    index = GraphIndex.from_arrays(embeddings, mol_ids, smiles, fingerprints, descriptors,
                                   checkpoint_sha256="abc", feature_metadata=metadata)
    index.save(tmp_path / "index.npz")
    loaded = GraphIndex.load(tmp_path / "index.npz", expected_checkpoint_sha256="abc",
                             expected_feature_metadata=metadata)
    assert loaded.smiles.tolist() == smiles.tolist()
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        GraphIndex.load(tmp_path / "index.npz", expected_checkpoint_sha256="wrong",
                        expected_feature_metadata=metadata)


def test_optional_reranking_uses_only_predicted_query_features():
    fingerprints = np.zeros((2, 2048), dtype=np.uint8)
    fingerprints[0, [0, 2]] = 1
    fingerprints[1, [0, 1]] = 1
    descriptors = np.zeros((2, 15), dtype=np.float32)
    descriptors[0, :2] = [0.0, 1.0]
    descriptors[1, :2] = [1.0, 0.0]
    predicted_fingerprint = np.zeros(2048, dtype=np.float32)
    predicted_fingerprint[[0, 2]] = [0.9, 0.8]
    predicted_descriptors = np.zeros(15, dtype=np.float32)
    predicted_descriptors[:2] = [0.0, 1.0]
    index = GraphIndex.from_arrays(
        embeddings=np.array([[1.0, 0.0], [0.8, 0.2]], dtype=np.float32),
        mol_ids=np.array([1, 2], dtype=np.int64),
        smiles=np.array(["CC", "CO"]),
        fingerprints=fingerprints,
        descriptors=descriptors,
        checkpoint_sha256="abc",
        feature_metadata=StructureFeatureConfig().to_metadata(),
    )
    result = index.search(
        query_embedding=np.array([1.0, 0.0], dtype=np.float32),
        topk=1,
        rerank=RerankConfig(top_m=2, fingerprint_weight=0.2, descriptor_weight=0.1),
        predicted_fingerprint=predicted_fingerprint,
        predicted_descriptors=predicted_descriptors,
    )
    assert result.indices.shape == (1,)
    assert result.metadata["query_structure_source"] == "predicted_from_spectrum"


def test_stage1_v2_run_accepts_only_spectrum(v2_config, random_spectrum):
    result = Stage1Retrieval(v2_config).run(random_spectrum)
    assert len(result.candidates) == v2_config["pipeline"]["top_k"]
    assert all(c.retrieval_mode == "spectrum_graph_v2" for c in result.candidates)
```

`v2_config` fixture 使用 Task 4 的 65 点 tiny model、Task 7 `save_checkpoint` 和本任务 `GraphIndex.from_arrays` 在 `tmp_path` 生成真实 checkpoint/index；`random_spectrum` 固定为 `np.random.default_rng(0).normal(size=65).astype(np.float32)`。测试不得 mock encoder 或 index loader。

- [ ] **Step 2: 运行索引/Stage 1 测试确认新模式缺失**

Run: `pytest tests/retrieval/test_graph_index.py tests/pipeline/test_stage1_structure_retrieval.py -v`

- [ ] **Step 3: 实现 GraphIndex 及 metadata 校验**

索引保存 embeddings、mol IDs、canonical SMILES、packed fingerprint、descriptors 和 JSON metadata。metadata 包含 `schema_version="spectrum_graph_index_v2"`、checkpoint SHA-256、feature config、embedding dim、normalized flag 和 canonical SMILES 清单 SHA-256。加载时逐字段校验；query/embedding 非有限或维度不符时 fail fast。

`RerankConfig` 默认 disabled。启用时先取 cosine Top-M，再用 spectrum auxiliary head 的预测 fingerprint 与候选真实 fingerprint 计算 soft Tanimoto，并用标准化 descriptor mean absolute error 作为惩罚：`final_score = cosine + fingerprint_weight * soft_tanimoto - descriptor_weight * descriptor_mae`。缺少预测 head、维度不符或非有限时 fail fast，禁止读取 query ground truth。

- [ ] **Step 4: 扩展 inference wrapper 和索引构建 CLI**

```python
class StructureAwareSpectralEncoder:
    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: str | None = None):
        payload = load_checkpoint(checkpoint_path, expected_schema="spectrum_graph_v2")
        model = StructureAwareRetriever.from_config(payload["model_config"])
        model.load_state_dict(payload["model_state_dict"])
        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        return cls(model=model.eval().to(resolved_device), payload=payload,
                   device=resolved_device)

    def encode_spectrum(self, spectrum: np.ndarray) -> np.ndarray:
        tensor = self._prepare_spectrum(spectrum).to(self.device)
        with torch.no_grad():
            return self.model.encode_spectrum(tensor).squeeze(0).cpu().numpy()


def build_graph_index(model, dataloader, checkpoint_path, output_path) -> GraphIndex:
    embedding_batches = []
    mol_id_batches = []
    smiles = []
    with torch.no_grad():
        for graph_batch in dataloader:
            encoded = model.encode_graph(graph_batch.to(next(model.parameters()).device))
            embedding_batches.append(encoded.cpu().numpy())
            mol_id_batches.append(graph_batch.mol_ids.numpy())
            smiles.extend(graph_batch.canonical_smiles)
    index = GraphIndex.from_encoded_batches(
        embedding_batches=embedding_batches,
        mol_id_batches=mol_id_batches,
        smiles=smiles,
        checkpoint_path=checkpoint_path,
        dataset=dataloader.dataset,
    )
    index.save(output_path)
    return index
```

索引 CLI 只读取显式 `--smiles-path`，不扫描目录；分 batch 编码并写单一 `.npz`。

- [ ] **Step 5: 更新 Stage 1 config 和 Candidate provenance**

`stage1.retrieval_mode` 只接受 `spectral_only_v1` 或 `spectrum_graph_v2`。v2 加载图 index，legacy 加载旧 embeddings；默认设置为 v2，但缺文件时给出明确错误而非静默 fallback。`stage1.rerank` 保存 enabled、top_m、fingerprint_weight 和 descriptor_weight，默认 disabled。`Candidate` 新增 `retrieval_mode: str = "spectral_only_v1"` 与 `embedding_similarity: float | None = None`，保持旧调用兼容；启用重排序时现有 `similarity` 保存 final score。本任务在 Task 1 的可运行 legacy import 基础上，按 retrieval mode 选择 `SpectralEncoder` 或 `StructureAwareSpectralEncoder`。

- [ ] **Step 6: 运行检索和 pipeline 回归测试后提交**

Run: `pytest tests/retrieval tests/pipeline tests/test_package_imports.py -v`

```bash
git add spectraexpert/retrieval spectraexpert/inference.py spectraexpert/pipeline scripts/build_graph_index.py tests/retrieval tests/pipeline
git commit -m "feat: retrieve molecular graphs from spectrum queries"
```

### Task 9: 实现统一消融评测与统计验收报告

**Files:**
- Create: `spectraexpert/eval/evaluate_structure_retrieval.py`
- Create: `tests/eval/test_retrieval_evaluation.py`
- Create: `scripts/run_tiny_pilot.sh`
- Create: `tests/scripts/test_run_tiny_pilot.py`
- Modify: `spectraexpert/eval/metrics.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: 冻结 query dataset、candidate index、baseline 或 v2 encoder。
- Produces: JSON 评测结果、paired bootstrap confidence interval、可审计的 baseline/v2 汇总。

- [ ] **Step 1: 写检索指标与 paired bootstrap 失败测试**

```python
def test_retrieval_metrics_report_identity_excluded_structure_quality():
    metrics = rank_structure_candidates(
        query_smiles="CCO",
        candidate_smiles=["CCO", "CCCO", "c1ccccc1"],
        scores=[1.0, 0.8, 0.2],
        exclude_identity=True,
    )
    assert metrics["top1_smiles"] == "CCCO"
    assert "chemical_ged_v2" in metrics
    assert "legacy_ged_v1" in metrics


def test_paired_bootstrap_is_reproducible():
    result1 = paired_bootstrap(np.array([.3,.4,.5]), np.array([.1,.2,.2]), seed=0, n_resamples=2000)
    result2 = paired_bootstrap(np.array([.3,.4,.5]), np.array([.1,.2,.2]), seed=0, n_resamples=2000)
    assert result1 == result2
    assert result1["mean_delta"] > 0
```

- [ ] **Step 2: 运行评测测试确认新 API 缺失**

Run: `pytest tests/eval/test_retrieval_evaluation.py -v`

- [ ] **Step 3: 实现全指标与 bootstrap**

每个 query 保存 Top-K SMILES、cosine score、Tanimoto、两个 GED 版本、nGED、heavy-atom gap、元素 Jaccard、分子量误差和 scaffold match。aggregate 保存 Top-1/mean Top-5/best@5/best@10 Tanimoto、exact/scaffold/structure Recall@1/5/10、embedding–Tanimoto Spearman、候选中训练未覆盖元素的比例和 latency。paired bootstrap 固定 10,000 resamples、seed=0，并报告 mean delta 与 2.5/97.5 percentile。

- [ ] **Step 4: 实现 ablation CLI 与防泄漏审计字段**

CLI 接受 `--mode knn|spectral_only_v1|spectrum_graph_v2`、`--query-data-dir`、`--candidate-data-dir`、`--candidate-index`、`--checkpoint`、`--top-k`、`--exclude-identity`、`--output`。KNN 与 spectral-only 从显式 candidate data directory 构建光谱候选，v2 从 graph index 构建图候选。输出 config 显式记录 `query_inputs=["spectrum"]`，并拒绝 query 表中由调用者指定为模型输入的 SMILES/descriptor flags；ground-truth SMILES 只在 ranking 完成后交给 metrics。

- [ ] **Step 5: 实现固定 pilot launcher 并测试命令矩阵**

`scripts/run_tiny_pilot.sh` 接受 `--data-dir`、`--output-root`、`--gpu` 和可选 `--dry-run`。脚本固定 seed 0/1/2、epochs 50、patience 8、batch 32、`embed_dim=128`、depth 3、heads 4、projection 64、graph hidden 128，并运行下列矩阵：

| Variant | training mode | lambda aug | lambda struct | lambda aux | lambda fingerprint |
|---|---|---:|---:|---:|---:|
| A | `spectral_only_v1` | 1.0 | 0.0 | 0.0 | 0.0 |
| B | `spectrum_graph_v2` | 0.25 | 0.0 | 0.0 | 0.0 |
| C | `spectrum_graph_v2` | 0.25 | 0.5 | 0.0 | 0.0 |
| D | `spectrum_graph_v2` | 0.25 | 0.5 | 0.1 | 0.0 |

每条命令都显式传 `--split-manifest "$DATA_DIR/split_manifest.json"`，保证不同 seed 和 variant 只改变训练随机性与 loss，不改变数据划分。`--dry-run` 只打印 12 条完整 Python 命令。测试断言恰有 12 条、三个 seed、同一 split manifest 和四组 loss 参数，且 `bash -n scripts/run_tiny_pilot.sh` 通过。

- [ ] **Step 6: 更新 README 的训练/索引/评测命令并运行测试**

README 给出从 Raman JSONL 准备数据、训练 v2、构建 graph index、运行 identity-included/excluded 评测的命令，并注明 tiny pilot 不能作为最终论文结论。

Run: `pytest tests/eval tests/retrieval tests/pipeline tests/scripts/test_run_tiny_pilot.py -v && bash -n scripts/run_tiny_pilot.sh`

- [ ] **Step 7: 提交评测入口**

```bash
git add spectraexpert/eval tests/eval scripts/run_tiny_pilot.sh tests/scripts/test_run_tiny_pilot.py README.md
git commit -m "feat: evaluate structure-aware retrieval"
```

### Task 10: 在 JJ 服务器执行真实数据 pilot 与消融

**Files:**
- Create: `docs/experiments/2026-09-05-theory-data-tiny-pilot.md`
- No local data files or remote outputs are committed.

**Interfaces:**
- Consumes: 本分支代码；服务器授权的两个 Raman JSONL；当时空闲 GPU。
- Produces: 服务器端 prepared Parquet、checkpoints、graph index、JSON metrics 和日志；仓库内只提交不含大文件的实验报告。

- [ ] **Step 1: 运行本地完整测试并记录基线**

Run:

```bash
source /Users/andrewluan/.venvs/spectrexpert-local/bin/activate
pytest -v
```

Expected: 全部 PASS；任何失败必须在同步前修复。

- [ ] **Step 2: 执行同步 dry-run 并审计清单**

Run from repository root:

```bash
rsync -an --max-size=20m \
  --exclude='.git/' --exclude='.worktrees/' --exclude='.superpowers/' \
  --exclude='data/' --exclude='outputs/' --exclude='runs/' --exclude='checkpoints/' \
  --exclude='*.csv' --exclude='*.parquet' --exclude='*.data' --exclude='*.pdf' \
  --exclude='*.png' --exclude='*.jpg' --exclude='*.zip' --exclude='*.rar' \
  ./ user@114.214.255.164:/home/user/IRfingerprints/SpectrExpert/
```

确认清单只有源代码、测试、轻量配置和 Markdown 后，以相同参数去掉 `-n` 正式同步；禁止添加 `--delete`。

- [ ] **Step 3: 校验源文件 hash、GPU 和依赖**

本地与远端计算 `spectraexpert/models/multimodal_retriever.py` SHA-256 并确认一致。远端运行 `nvidia-smi`，选择空闲 GPU；不得沿用本计划编写时 GPU 1–4 空闲的快照。激活 `/home/user/.venvs/spectrexpert`，从 USTC pip/uv mirror 安装 `-e '.[dev]'`，记录 torch、CUDA、RDKit、SciPy 版本。

- [ ] **Step 4: 在服务器准备授权的 400 条 pilot 数据**

```bash
python scripts/prepare_raman_jsonl.py \
  --input /data/SpecGPT/data/theory_data_tiny/Raman_QME14S.jsonl \
  --input /data/SpecGPT/data/theory_data_tiny/Raman_fg26.jsonl \
  --output-dir /home/user/IRfingerprints/SpectrExpert/outputs/theory_data_tiny_prepared \
  --start 500 --stop 4000 --points 3501 --split-seed 0
```

验证输出 400 行、400 个唯一 canonical SMILES、统一 3501 点、无 NaN/Inf，并记录因 RDKit 非法而排除的数量；不读取同目录其他文件。

- [ ] **Step 5: 运行 CPU 单测和单 GPU synthetic/real-data smoke**

先运行 `pytest -v`。随后以当时选定 GPU 执行 1 epoch、小模型、最多 32 条数据：

```bash
GPU_INDEX=$(nvidia-smi --query-gpu=index,memory.free,utilization.gpu \
  --format=csv,noheader,nounits | awk -F, '$3 + 0 < 10 {gsub(/ /,"",$1); gsub(/ /,"",$2); print $1, $2}' \
  | sort -k2,2nr | head -n1 | awk '{print $1}')
test -n "$GPU_INDEX"
nvidia-smi -i "$GPU_INDEX"
CUDA_VISIBLE_DEVICES="$GPU_INDEX" python -m spectraexpert.train_structure_aware \
  --data-dir /home/user/IRfingerprints/SpectrExpert/outputs/theory_data_tiny_prepared \
  --output-dir /home/user/IRfingerprints/SpectrExpert/outputs/structure_smoke \
  --epochs 1 --batch-size 8 --embed-dim 64 --depth 2 --num-heads 4 \
  --projection-dim 32 --graph-hidden-dim 64 --max-samples 32 --num-workers 0
```

在启动 Python 前检查上述 `nvidia-smi -i "$GPU_INDEX"` 输出，确认没有新任务占用；确认 checkpoint schema、各 loss 有限、首个 validation 和 GPU memory 正常。

- [ ] **Step 6: 在 tmux 运行四种 pilot 消融**

固定 scaffold split 与 seed 0/1/2，对 400 条运行：A 为 spectral-only baseline；B 为 cross-modal 且 `lambda_struct=0, lambda_aux=0, lambda_fingerprint=0`；C 为 cross-modal + structure；D 为 C + descriptor。统一小模型 `embed_dim=128, depth=3, heads=4, projection_dim=64, graph_hidden_dim=128`、batch 32、最多 50 epochs、patience 8。每个 run 保存独立目录与日志；tmux 名称为 `structure_retrieval_pilot_20260905`，启动后检查进程、日志和首轮输出。

```bash
tmux new-session -d -s structure_retrieval_pilot_20260905 \
  "cd /home/user/IRfingerprints/SpectrExpert && \
   source /home/user/.venvs/spectrexpert/bin/activate && \
   bash scripts/run_tiny_pilot.sh \
     --data-dir /home/user/IRfingerprints/SpectrExpert/outputs/theory_data_tiny_prepared \
     --output-root /home/user/IRfingerprints/SpectrExpert/outputs/structure_pilot \
     --gpu $GPU_INDEX \
     > /home/user/IRfingerprints/SpectrExpert/outputs/structure_pilot.log 2>&1"
tmux has-session -t structure_retrieval_pilot_20260905
tmux capture-pane -pt structure_retrieval_pilot_20260905 -S -80
```

- [ ] **Step 7: 构建候选图索引并运行两类评测**

对每个 seed 的最佳 checkpoint 构建包含全部 400 个 canonical SMILES 的 graph index。分别运行：

- identity-included：检验 held-out spectrum 能否跨模态找到同一分子图；
- identity-excluded：屏蔽相同 canonical SMILES，检验 Top-K 结构近邻质量。

比较 A/B/C/D 的 Top-1 Tanimoto、best@5/10、chemical GED v2、heavy-atom gap、exact recall 和 paired bootstrap CI。KNN baseline 只在相同 query/candidate split 和相同 identity mask 下比较。

- [ ] **Step 8: 写实验报告、提交轻量结果摘要**

报告必须包含数据 schema、插值规则、split、依赖/GPU、命令、seed、每个配置的指标、失败/超时、日志与远端 output 路径，并明确标注“400 条 pilot，不能作为最终论文结论”。不得提交 checkpoint、Parquet、JSON 全量 per-sample 输出或日志。

```bash
git add docs/experiments/2026-09-05-theory-data-tiny-pilot.md
git commit -m "docs: report structure retrieval pilot"
```

## 最终整体复核

执行完所有任务后，完整检查：

```bash
pytest -v
rg -n 'from (proj\.)?spectra_expert|import (proj\.)?spectra_expert' spectraexpert
git diff --check $(git merge-base main HEAD)..HEAD
git status --short
```

最终 reviewer 必须重点确认：查询路径无结构标签泄漏；Morgan/GED/图 schema 一致；aromatic bond 未被截断；相同 canonical identity 不作为负样本；DDP 全局负样本保留梯度；index/checkpoint hash 校验有效；tiny pilot 没有被过度解读。
