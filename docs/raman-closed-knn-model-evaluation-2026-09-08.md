# Raman 10k/200 闭库检索评测

用户关注的与纯 LLM 同口径的 Tanimoto/GED 比较已补算，见
[结构指标对照报告](raman-closed-structural-comparison-2026-09-08.md)。下文保留检索排名指标作审计。

两种方法均已完成全部 200 条实验 Raman 查询，候选库固定为 10,000 个理论分子组。
模型在 Top-1/5/10 上与纯 kNN 接近，本次结果不足以支持模型优于 kNN 的结论。

| 方法 | Top-1 | Top-5 | Top-10 | MRR | 中位排名 |
|---|---:|---:|---:|---:|---:|
| 纯 kNN，原始强度余弦 | 1.0%（2/200） | 6.0%（12/200） | 11.0%（22/200） | 0.04108 | 364 |
| spectrum_graph_v2，光谱→分子图余弦 | 1.0%（2/200） | 6.5%（13/200） | 10.5%（21/200） | 0.04149 | 136.5 |

相对 kNN，模型使 127 个查询的正确候选排名改善，1 个持平，72 个变差。
这种改善主要体现在完整排名，不意味着头部命中率显著提升；此处没有做显著性检验。

## 协议

查询顺序遵循冻结 sample_manifest 的 test_groups，候选顺序遵循 candidate_groups。
通过 mol_id 连接表格，不把实验查询行号当作理论候选 ID。两种方法均保留正确身份，
不使用 exclude-identity。每组各一条谱，按分数降序稳定排序，同分遵循冻结候选顺序。
MRR 根据完整 10,000 候选排名计算，不以 Top-10 截断。

主指标按 group_hash 命中：最大分子片段连接关系、忽略立体化学。
额外计算完整 canonical SMILES Top-1，kNN 为 0.5%，模型为 1.0%，与分组命中口径分列。
这是已见分子身份、未见实验谱的跨域闭库任务。200 个实验测试组的理论谱均参与过训练；
实验谱没有进入训练、早停或 checkpoint 选择。候选库包含 9,500 个训练身份及 500 个理论验证身份。

两种方法均使用冻结的 500–4000 cm⁻¹、3501 点 Parquet 谱。kNN 对原始强度做 L2
归一化和余弦检索，不做逐谱中心化、平滑或覆盖区间交集遮罩。模型查询按训练契约逐谱
标准化后编码；全部候选结构由同一最佳 checkpoint 的 graph encoder 编码。
模型不使用 LLM、辅助头重排或测试结构作为查询输入；真实结构只在排序之后计分。

使用第 23 轮最佳 checkpoint（内部 epoch=22），SHA-256：
`0a2f9f16e46925e9ba79ca61ec0fcc530b78b83cffeea325825999f75752fedb`。
校验了 checkpoint 与训练 split、理论三件套及规范化候选身份的哈希绑定，
并确认它不是 max-samples smoke 结果。

## 耗时与产物

| 方法 | 平均单查询 | 中位单查询 | p95 | 200 查询合计 |
|---|---:|---:|---:|---:|
| kNN | 85.65 ms | 86.57 ms | 90.04 ms | 17.13 s |
| 模型 | 4.10 ms | 4.05 ms | 4.29 ms | 0.82 s |

候选分子图构建和编码另耗时 6.58 秒，整项评测约 31.63 秒。
模型编码使用 GPU 0 A800，余弦检索和完整排序均使用 CPU；OMP/OpenBLAS 线程数为 4。
计时包含在线输入准备、模型查询编码、候选范数计算、评分和完整排序，不含文件读取、身份
计分与结果写盘；候选图构建单列。两者复用现有 retrieve 实现，未做 kNN 候选范数缓存等
专项性能优化，因此这些延迟是当前实现测量，不是算法速度上限。

服务器结果目录：
`/home/user/IRfingerprints/SpectrExpert/outputs/raman_closed_knn_model_eval_run01/`。

- `summary.json`：聚合指标与耗时。
- `protocol.json`：查询/候选 ID、数据哈希、checkpoint、归一化、同分和计时协议。
- `knn_cosine.jsonl`、`spectrum_graph_v2.jsonl`：逐查询目标排名及 Top-10 候选。
- `*_rankings.npy`：每种方法的完整 200×10000 候选 mol_id 排名。
- `candidate_graph_embeddings.npz`：图嵌入、候选 ID 与 checkpoint 哈希绑定。

复现入口为 `python -m scripts.evaluate_closed_raman`，参数为 `--data-root`、`--checkpoint`、
`--output-dir` 和 `--device cuda`；必须指定新的输出目录。
本地测试覆盖 MRR/Top-K 边界、非连续 ID 对齐和哈希拒绝。正式结果另经独立审计：每行排名
完整且无重复、两方法查询顺序一致、按 mol_id 重算的目标排名与 JSONL 一致，命中计数和 MRR 一致。
