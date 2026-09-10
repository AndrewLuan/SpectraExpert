# 同期竞争方法在统一 Raman Test-200 协议下的复测报告

## 结论先行

本次复测将 Vib2Mol 与 VibraCLIP 接入 SpectraExpert 最新冻结的 200 条 Raman 测试集，并统一了查询样本、10,000 分子候选库、结构身份判定和相似度指标。全部可运行实验均已完成，评分进程退出码为 0。

在不向模型泄露目标分子式或目标质量的主协议下，Vib2Mol 与 VibraCLIP 的 Group Top-1 和 Canonical Top-1 都是 0。VibraCLIP 在 Top-10 命中 1/200、Top-25 命中 2/200；Vib2Mol 检索重排在 Top-25 命中 1/200。Vib2Mol 的生成分支有 162/200 个首选 SMILES 可被 RDKit 解析，Valid rate 为 81.0%，但 Top-1、Top-5、Top-10 均未命中。

这些结果应解释为“实验 Raman 查询对理论 QM9S 候选库”的跨域压力测试，不应与竞争方法论文中各自数据分布、划分和输入条件不同的原始数字直接横向比较。

MolSpectLLM 没有被填入一个虚假的复测数字。服务器 `/home/user/repro` 中没有 MolSpectLLM 模型目录、权重或适配器；其官方仓库也未提供可执行推理代码或模型权重，并且公开任务覆盖 NMR、IR 和 MS，而不是 Raman。因此，在当前统一 Raman 协议下该方法记为 N/A。论文报告值也没有混入本表，因为它们不来自相同的 200 条查询、候选库和评分器。

## 冻结数据与统一输入

本次使用的数据版本为 `raman_closed_10k_test200_seed0_exact_v2`：

- 查询集为 200 条 SDBS 实验 Raman 光谱；
- 候选库为 10,000 个 QM9S 理论分子；
- 波数范围为 500–4000 cm⁻¹，统一网格长度为 3501；
- 200 个查询的精确目标结构均存在于候选库；
- 数据清单 SHA-256 为 `422e6729cad919b76789608db93faaad89276bb2c2397917d446d8acad41ef4c`；
- 导出协议 SHA-256 为 `77d3beb13134998056e4b35a77d9a54f53f13f431a428ef66ce91d6cf730d5ec`；
- 查询文件 SHA-256 为 `ce52aa96fa82a29190b5f29cd1c5e941c7e2e2ced7187ef3b85eaf263b8373fd`；
- 候选库文件 SHA-256 为 `730ba80df25a12170514825a48404a0c0007a67e046efb44e046eb670c12b0e7`。

为避免标签泄漏，统一导出只给模型 Raman 曲线和候选分子，不提供目标分子式或由目标结构计算的分子质量。不同模型仍使用其官方实现要求的输入变换：Vib2Mol 将 400–4000 cm⁻¹ 重采样到 1024 点并采用 clip/max normalization；VibraCLIP 将光谱重采样到 1750 点并采用 min–max normalization。这里统一的是样本语义、候选集合和评分规则，而不是强行把模型专属的前处理改成同一个张量形状。

## 统一指标

所有模型输出都由同一评分脚本处理：

- **Group hit@K**：使用冻结的 `group_hash` 判定结构身份；只保留最大分子片段，并忽略立体化学和同位素；
- **Canonical hit@K**：比较 RDKit canonical SMILES；
- **Morgan Tanimoto**：半径 2、2048 bit、包含键类型、不使用手性；数值越高越好；
- **chemical GED v2**：节点使用原子序数，边使用离散键类型；数值越低越好；每个样本最多计算 10 秒，超时或无效值记为 null，不进入均值，同时报告有效样本数 `n`；
- **Valid**：首选预测能被 RDKit 解析的比例。

统一评分器 SHA-256 为 `c4a1f408314f653156d22c9582202ef210910fdd93da251697275a8b64889454`，其指标实现依赖的源文件 SHA-256 为 `b0cef345e81617b3c05a4b4b45d751b4d521dea6f36c757a4717d43eeb87ac81`。

## 主结果

下表只保留不使用目标质量的主协议。Top-K 是闭集检索命中率；Vib2Mol generation 没有统一的 10,000 库排名，其 Top-K 表示 beam 前 K 个生成结构是否命中。

| 方法 | 分支 | Valid | Group@1 | Group@5 | Group@10 | Group@25 | Morgan Tanimoto ↑ | chemical GED ↓（n） |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Vib2Mol | retrieval + Raman rerank | 100.0% | 0.0% | 0.0% | 0.0% | 0.5% | 0.0644 | 12.5667（150） |
| Vib2Mol | generation + Raman rerank | 81.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0582 | 9.1698（159） |
| VibraCLIP | retrieval, no-mass | 100.0% | 0.0% | 0.0% | 0.5% | 1.0% | 0.0490 | 9.9457（184） |
| MolSpectLLM | Raman | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

三组可执行主实验的 Canonical hit@K 与 Group hit@K 相同。Vib2Mol 检索重排在 Top-25 的唯一命中也是 canonical exact match；VibraCLIP no-mass 在 Top-10 和 Top-25 的命中同样满足 canonical exact match。

## 敏感性分析不进入主表

为了判断实现选择是否改变结论，额外保留了两个对照：

| 设置 | Valid | Group@1 | Group@10 | Group@25 | Morgan Tanimoto ↑ | chemical GED ↓（n） |
|---|---:|---:|---:|---:|---:|---:|
| Vib2Mol retrieval，不做 Raman rerank | 100.0% | 0.0% | 0.0% | 0.5% | 0.0554 | 11.7134（164） |
| VibraCLIP with-mass，查询质量固定为训练均值锚点 | 100.0% | 0.0% | 0.0% | 0.0% | 0.0430 | 10.2653（196） |

VibraCLIP 的 with-mass 权重不能在主实验中使用真实目标质量，否则会把由目标结构得到的信息泄漏给模型。这里将标准化质量固定为 0，即训练均值位置，仅用于敏感性分析；它没有改善命中率。

## 模型与运行配置

Vib2Mol 使用复现目录中的 QM9S Raman–Kekulé SMILES 权重。检索与重排权重为：

`/home/user/repro/vib2mol/checkpoints/qm9s/raman-kekule_smiles/vib2mol/2025-07-06-12-15-bbe117/epoch999.pth`

生成权重为：

`/home/user/repro/vib2mol/checkpoints/qm9s/raman-kekule_smiles/vib2mol/2025-07-07-07-51-af1552/epoch999.pth`

生成使用 beam size 10，并对前 5 个候选进行 Raman 重排。检索输出候选库前 100 名。

VibraCLIP 主实验使用 no-mass 权重：

`/home/user/repro/vibraclip/pre_trained/no_mass/checkpoint_vibraclip_graph_ir_raman_nomass_allpairs.ckpt`

敏感性分析使用 with-mass 权重：

`/home/user/repro/vibraclip/pre_trained/with_mass/checkpoint_vibraclip_graph_ir_raman_mass_allpairs.ckpt`

VibraCLIP 为候选分子生成三维构象时，10,000 个分子中有 85 个 ETKDG 失败，最终可检索 9,915 个候选。经核对，这 85 个失败分子不包含 200 条查询中的任何目标结构，因此不会降低本次 Top-K 命中率的理论上限。

## 复现与审计路径

统一数据导出脚本位于 `scripts/export_competitor_benchmark.py`，统一评分脚本位于 `scripts/score_competitor_benchmark.py`，对应的最小测试位于 `tests/scripts/test_competitor_benchmark.py`。

服务器上的审计产物如下：

- 协议数据：`/home/user/repro/unified/benchmark/raman_exact_v2_test200`；
- 模型原始输出：`/home/user/repro/unified/results/raman_exact_v2_test200_20260909`；
- 运行日志：`/home/user/repro/unified/logs/raman_exact_v2_test200_20260909`；
- 统一评分：`/home/user/repro/unified/scored/raman_exact_v2_test200_20260909_all5`；
- 统一摘要 SHA-256：`d3d4eefcf089d7f85a68538dec724809dd7d93b648e2851f16406d4c64862e5d`；
- 统一表格 SHA-256：`2526f64451bfe3a7a76cef4c4d44764787f8500f502f120ea50bd90bfbfa236f`。

本地最小测试结果为 `3 passed`。远程五组统一评分进程的退出码为 0；每组均包含 200 个按 `q0000` 至 `q0199` 排序的结果，没有跳样本。

## MolSpectLLM 的可比性边界

MolSpectLLM 官方仓库为 <https://github.com/Eurekashen/SpecMol>，论文为 <https://arxiv.org/abs/2509.21861>。截至本次复测，官方仓库未提供模型权重或可执行推理入口，公开模态也没有 Raman。要补齐真正的同协议结果，至少需要作者提供可本地推理的 checkpoint、Raman 输入定义以及相应前处理；在这些条件缺失时，任何数字都不能被称为本次 Test-200 上的模型实测值。
