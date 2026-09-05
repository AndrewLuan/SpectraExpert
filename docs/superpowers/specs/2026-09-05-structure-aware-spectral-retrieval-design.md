# 结构感知的光谱—分子图对比检索设计

## 目标与问题定义

当前 Stage 1 使用同一条 Raman 光谱的两个增强视图训练单编码器 SimCLR。这个目标能够学习光谱增强不变性，却没有告诉模型哪些分子在结构上相似。训练时，同一 batch 中除自身以外的所有分子都会被当作等价负样本；SMILES 文件虽然被数据集读取，但 SMILES、原子组成和分子图均未进入模型。结果是 encoder 检索仍主要服从谱形相似度，而不是分子结构相似度。

历史 ChEMBL 评测与这一诊断一致：训练后 encoder 的 Top-1 Morgan Tanimoto 均值为 0.0576，原始光谱 KNN 为 0.0556，二者几乎没有差别；encoder 的 Top-1 GED 均值为 20.55，KNN 为 20.93。另一方面，QM9S 候选库只包含不超过 9 个重原子的分子，而测试分子可达到 21 个重原子，因此候选库覆盖范围本身也限制了可达到的结构相似度。

本设计的目标是让未知光谱的 embedding 显式映射到分子图结构空间，使 Top-K 候选不仅具有相似光谱，还在元素组成、局部子结构、键连接和整体拓扑上更接近真实分子。推理阶段只允许输入一维 Raman/IR 光谱；分子式、分子量、原子数、元素种类和真实结构一律不可作为查询输入。

“结构相似”不能被表述为数学上的绝对保证。光谱到结构是多解逆问题，且检索质量受候选库覆盖影响。本阶段可提供的保证是：结构信息直接进入训练目标；模型选择和验收以结构指标为准；候选库缺少相近结构时明确报告覆盖限制，而不把高 cosine similarity 误称为高结构相似度。

## 设计边界

本阶段包含：

- 从已有 canonical SMILES 构造二维重原子分子图和结构监督标签；
- 新增分子图 encoder，并与 SpectrumTransformer 建立共享 embedding 空间；
- 保留光谱增强学习，同时增加结构关系和分子描述符监督；
- 将 Stage 1 从“光谱 embedding 检索光谱 embedding”改为“光谱 embedding 检索分子图 embedding”；
- 增加结构检索指标、消融实验入口、索引版本校验和最小测试；
- 修复阻碍该流程运行的包导入和缺失推理包装器问题。

本阶段不包含：

- 从未知光谱直接生成完整 SMILES；
- 使用查询分子的真实分子式、分子量或原子统计进行过滤；
- 本地遍历或复制 QM9S、ChEMBL 等大型数据目录；
- 未经用户许可上传大型数据、模型权重或实验输出；
- 默认引入依赖外部下载的预训练权重或需要编译扩展的图学习框架；
- 修改 Stage 2 和 Stage 3 的推理逻辑，除适配新版 Candidate/Stage 1 输出所必需的兼容改动外。

## 总体架构

系统采用光谱—分子图双编码器。训练样本为配对三元组：

\[
(x_i^{(1)}, x_i^{(2)}, G_i),
\]

其中 \(x_i^{(1)}\) 和 \(x_i^{(2)}\) 是同一分子的两个身份保持型光谱增强视图，\(G_i\) 是由 canonical SMILES 构造的二维分子图。

光谱分支沿用 SpectrumTransformer 主干：

\[
h_i^s=E_s(x_i),\qquad z_i^s=\operatorname{norm}(P_s(h_i^s)).
\]

图分支采用支持键特征的轻量 GIN：

\[
h_i^g=E_g(G_i),\qquad z_i^g=\operatorname{norm}(P_g(h_i^g)).
\]

两个 projection head 输出相同维度的 L2-normalized embedding。训练时同时使用光谱和分子图；推理时查询侧只调用 \(E_s\)，候选库则提前用 \(E_g\) 计算图 embedding。查询和候选通过 cosine similarity 排序。

采用纯 PyTorch 的图 batching 与 `index_add_` 聚合实现 GIN，不把 PyTorch Geometric 作为首版必需依赖。图级 readout 拼接 mean pooling 与 sum pooling：前者稳定表示局部组成，后者保留分子规模。这样可以避免服务器安装 `torch-scatter`、`torch-sparse` 等编译扩展，同时允许候选图包含任意数量的重原子。若后续实验证明图 encoder 是瓶颈，再单独比较 PyG 或三维等变网络。

## 分子图与辅助标签

### 由 GED 与 Tanimoto 对齐的图输入

节点和边特征不采用任意的属性堆叠，而是分成两层语义。第一层严格对应 GED 的编辑对象，第二层对应 radius-2 Morgan fingerprint 用于区分局部化学环境的信息。

GED 核心节点特征为 `atomic_number`。当前项目的 GED 在 `_node_match` 中只比较 atomic number，因此元素身份必须拥有独立 embedding，不能被压缩为“金属/非金属”或简单 one-hot 统计。atomic number 使用覆盖 1–118 的固定索引和独立 UNK token，避免因候选库新增元素而发生数组越界。

GED 核心边特征为离散 `bond_type`：`SINGLE`、`DOUBLE`、`TRIPLE`、`AROMATIC` 和 `OTHER`。当前 `metrics.py` 使用 `int(bond.GetBondTypeAsDouble())`，会把数值 1.5 的 aromatic bond 截断为 1，从而与 single bond 混淆。新版图特征和新版 GED 必须使用稳定的类别编码，禁止经过浮点值转整数。

为了与默认的 radius-2 Morgan Tanimoto 对齐，节点进一步包含：

- degree；
- total valence；
- formal charge；
- total hydrogen count；
- aromatic flag；
- ring membership。

边进一步包含：

- conjugation flag；
- ring membership。

这些属性和两跳连接关系共同描述 Morgan fingerprint 所编码的局部原子环境。首版默认不加入 chirality tag 和 bond stereo：现有 `GetMorganFingerprintAsBitVect(mol, 2, 2048)` 没有启用 chirality，当前 GED 也忽略立体信息。若后续实验启用 `useChirality=True`，必须同时升级 fingerprint、图特征、结构指标和 featurizer version，不能只改模型输入。

每条化学键构成双向边。氢原子不显式建图，使用节点的 total hydrogen count 表示，以控制图规模并与现有 heavy-atom GED/Tanimoto 评测保持一致。无法解析的 SMILES 不得静默生成空图；预处理应记录并排除该样本。

QM9S 分子组成只覆盖 H/C/N/O/F，其中重原子图节点只包含 C/N/O/F，H 仅通过 total hydrogen count 表示；没有配对训练数据的重元素 embedding 仍属于分布外表示。建索引和评测必须报告未知元素比例与训练元素词表，不能把“能够编码”误解为“已经学会该元素的谱图对应关系”。

### 与 Morgan radius 对齐的消息传递

图 encoder 使用 edge-conditioned GIN。第 \(k\) 层先根据邻接原子和键特征聚合消息，再更新中心原子：

\[
m_v^{(k)}=\sum_{u\in\mathcal N(v)}
\phi_k\!\left(h_u^{(k-1)},e_{uv}\right),
\qquad
h_v^{(k)}=\operatorname{MLP}_k\!\left((1+\epsilon_k)h_v^{(k-1)}+m_v^{(k)}\right).
\]

主配置使用两轮消息传递，并对 \(h^{(0)}\)、\(h^{(1)}\)、\(h^{(2)}\) 做 multi-scale readout，使表示显式覆盖中心原子、一步环境和两步环境，与 ECFP4/Morgan radius=2 的结构监督尺度一致。每一尺度分别计算 mean pooling 与 sum pooling，随后拼接并投影：mean 保留局部环境分布，sum 保留原子数和子结构计数。更深 GIN 只作为消融配置，不在首版默认启用，以免表示尺度与监督指标失配。

### 结构关系标签

为每个分子离线计算 radius=2、2048-bit Morgan fingerprint。实现必须显式固定 `useBondTypes=True`、`useChirality=False`，并把这些参数写入缓存与索引 metadata，避免 RDKit 版本或调用入口改变默认语义。batch 内结构相似度定义为：

\[
S_{ij}=\operatorname{Tanimoto}(f_i,f_j),\qquad S_{ij}\in[0,1].
\]

Morgan fingerprint 不作为查询时的已知输入，而是训练阶段的关系监督及可选辅助预测目标。候选库的 fingerprint 来自候选 SMILES，允许离线计算。

GED 不用于在线 batch loss。其最优匹配搜索开销远高于 fingerprint Tanimoto，当前实现还可能超时，并且离散 GED 对神经网络不可微。训练时用可批量计算的 Morgan Tanimoto 提供连续结构软标签；GED 用于验证集、测试集和少量 hard-pair 审计，检查模型是否只学会局部 fingerprint 相似而忽略整体拓扑。

### 分子描述符标签

从 SMILES 离线计算并在训练集统计量下标准化：

- C、N、O、F、H 数量及元素 presence；
- heavy-atom count；
- molecular weight；
- ring count 和 aromatic ring count；
- aromatic atom count；
- single、double、triple、aromatic bond count；
- formal charge。

这些标签仅监督光谱 backbone 学习组成和规模信息。验证/测试必须复用训练集统计量，不得重新拟合归一化参数。

## 损失函数

总损失为：

\[
\mathcal L =
\mathcal L_{\mathrm{cross}}
+\lambda_{\mathrm{aug}}\mathcal L_{\mathrm{aug}}
+\lambda_{\mathrm{struct}}\mathcal L_{\mathrm{struct}}
+\lambda_{\mathrm{aux}}\mathcal L_{\mathrm{aux}}.
\]

权重不在设计阶段伪定为最优值。实现提供可配置默认值，并通过消融和验证集选择；不得根据最终测试集调参。

### 跨模态身份对齐

对每个光谱增强视图分别计算对称 CLIP/InfoNCE：

\[
\mathcal L_{\mathrm{cross}}
=\frac{1}{4}\sum_{v\in\{1,2\}}
\left[
\operatorname{CE}(z^{s(v)}z^{g\top}/\tau,I)
+\operatorname{CE}(z^g z^{s(v)\top}/\tau,I)
\right].
\]

单位矩阵 \(I\) 表示光谱与同一分子图是精确配对。DDP 训练必须聚合各 rank 的 embedding 和标签形成全局负样本，不能继续让每张 GPU 独立计算小 batch 的负样本集合。实现应使用保留本 rank 梯度的 gather，并正确处理全局样本索引。

### 光谱增强一致性

\(\mathcal L_{\mathrm{aug}}\) 保留现有两种光谱视图之间的 NT-Xent 约束，使实验噪声、基线变化、峰位微扰和强度变化不会破坏同一分子的表示。新版实现不再依赖构造时固定的 batch size，最后一个不完整 batch 和 DDP 全局 batch 都应被正确支持。

### 结构关系保持

普通 one-hot InfoNCE 会把结构相近分子也视为强负样本。为此，使用 Morgan Tanimoto 构造结构软分布：

\[
P_{ij}^{\mathrm{struct}}
=\frac{\exp(S_{ij}/\tau_s)}{\sum_k\exp(S_{ik}/\tau_s)}.
\]

模型分布由跨模态相似度和同模态相似度产生。结构关系损失用 KL divergence 或 soft cross-entropy，使高 Tanimoto 分子在共享空间中比低 Tanimoto 分子更近，同时保留 \(\mathcal L_{\mathrm{cross}}\) 对精确配对的辨别能力：

\[
\mathcal L_{\mathrm{struct}}
=D_{KL}(P^{\mathrm{struct}}\|Q^{sg})
+\frac{1}{2}D_{KL}(P^{\mathrm{struct}}\|Q^{ss})
+\frac{1}{2}D_{KL}(P^{\mathrm{struct}}\|Q^{gg}).
\]

对角线、重复结构和可能的重复 SMILES 必须显式处理。相同 canonical SMILES 不得互相充当负样本；同分异构体保留各自 identity，但根据 fingerprint 相似度获得连续结构权重。

### 光谱侧辅助预测

从投影前的光谱特征 \(h^s\) 接出辅助头：

- 标准化连续/计数描述符使用 Smooth L1；
- 元素 presence 使用 BCEWithLogits；
- 可选 Morgan bit 预测使用带正类权重的 BCEWithLogits，以处理 2048 位 fingerprint 的高度稀疏性。

辅助头仅在训练中参与 loss。检索的主 embedding 不拼接真实描述符，也不要求推理时提供任何结构元数据。fingerprint 预测头默认作为可消融组件，只有验证表明其提升结构检索时才进入最终配置。

## 数据流与防泄漏规则

当前数据集分别按行读取 spectra 和 SMILES，隐含假设二者顺序完全一致。新版数据管线必须以 `mol_id` 做一对一 join，并验证：

- 两张表的 `mol_id` 唯一；
- join 后没有丢失或重复配对；
- 每条 canonical SMILES 可由 RDKit 解析；
- 光谱长度与 wavenumber 轴一致；
- fingerprint、descriptor 和 graph 全部来自同一个 join 后样本。

数据划分在增强前完成。训练、验证和测试按 canonical identity 去重；用于评估泛化时增加 Bemis–Murcko scaffold split，避免同一 scaffold 泄漏到不同 split。ChEMBL 保持完全外部测试集，不参与模型选择、loss 权重选择或标准化统计。

本地测试只使用代码内构造的少量 SMILES 和小型张量，不读取或遍历大型数据目录。正式特征缓存和训练在远端执行前，必须先确认数据实际存在、重新检查 GPU 状态，并遵循仓库的 dry-run 同步和禁止 `--delete` 规则。

## 检索与候选索引

候选索引由 canonical SMILES 而不是候选光谱构建，至少保存：

- graph embeddings；
- `mol_id`；
- canonical SMILES；
- Morgan fingerprint；
- 候选描述符；
- checkpoint hash；
- graph featurizer version；
- embedding dimension 和 normalization 标记；
- 候选 SMILES 清单 hash。

加载索引时必须校验 checkpoint、featurizer 和 embedding schema；不匹配时 fail fast，禁止用新模型查询旧 embedding 文件。

默认排序分数为共享空间 cosine similarity。另提供一个有明确开关的 Top-M 重排序实验：使用光谱辅助头预测的 fingerprint 概率和描述符，与候选的真实 fingerprint/descriptor 计算附加分数。该分数不得使用查询真实结构信息，也不得默认设置不可恢复的硬过滤条件，因为辅助预测错误可能把正确候选提前删除。

双编码器带来一个重要的覆盖优势：候选库只需 SMILES，不再要求每个候选都具有 Raman 光谱。因此后续可以将比 QM9S 更大、重原子分布更接近 ChEMBL 的结构库编码为候选索引。扩大候选库属于独立数据决策；在用户授权具体数据来源和规模前，本阶段只实现通用索引接口，不下载或传输新数据库。

## 与现有代码的衔接

实现预计涉及以下边界：

- `spectraexpert/data/`：按 `mol_id` 对齐数据、SMILES 图化、descriptor/fingerprint 缓存和 graph collate；
- `spectraexpert/models/spectrum_encoder.py`：区分 projection 前 backbone feature 与检索 embedding；
- `spectraexpert/models/graph_encoder.py`：新增纯 PyTorch 图 encoder；
- `spectraexpert/models/multimodal_encoder.py`：封装训练期双编码器和辅助头；
- `spectraexpert/utils/`：新增跨模态、结构软标签和动态 batch loss；
- `spectraexpert/train_contrastive.py`：训练循环、DDP 全局负样本、验证指标和 checkpoint schema；
- `spectraexpert/pipeline/stage1_retrieval.py`：加载光谱 encoder 和图索引；
- `spectraexpert/eval/`：新增结构相似性、组成误差和 embedding–structure correlation 指标；
- `scripts/`：新增候选图索引构建入口；
- `tests/`：覆盖数据对齐、图特征、loss、模型 shape、索引兼容性和端到端小样本检索。

仓库目前存在运行性不一致：实际包目录是 `spectraexpert`，训练代码却导入 `spectra_expert`，pipeline 多处导入不存在的 `proj.spectra_expert`，Stage 1 还依赖仓库中缺失的 `analyze_spectra.SpectralEncoder`。这不是本设计提出的新架构，但会阻碍新旧基线复现。实现时先统一为 `spectraexpert` 包路径，并在包内建立有版本校验的推理包装器；不重构无关的 Stage 2/3 行为。

旧的 spectral-only checkpoint 和 `embeddings.npz` 不与新版索引混用。若保留 legacy 模式，必须由显式配置选择，并在日志中标记 `spectral_only_v1`；新模式标记为 `spectrum_graph_v2`。

## 训练、验证与模型选择

第一轮实验采用逐步消融，不直接把所有组件一次性视为有效：

1. A：现有 spectral-only SimCLR；
2. B：仅加入精确配对的 spectrum–graph cross-modal loss；
3. C：B + structure relation loss；
4. D：C + descriptor auxiliary loss；
5. E：D + fingerprint auxiliary loss；
6. F：最佳训练配置 + 可选 Top-M 结构重排序。

模型选择的主指标从训练 loss 改为验证集结构检索指标。训练 loss 降低但结构指标不提升时不得保存为“最佳结构检索模型”。推荐主指标为 mean Top-1 Tanimoto，辅以 normalized GED、Top-K scaffold/structure recall 和精确匹配率。

由于 raw GED 随分子大小增长，报告中同时保留 raw GED 和 normalized GED。对结构距离耗时过高的全库比较，先使用 Morgan Tanimoto 做全量评估，再只对 Top-K 候选计算 GED。

为保持历史结果可追溯，评测同时区分两个版本：

- `legacy_ged_v1`：完全复现已有 `atomic_num` node match 及截断后的整数 bond type，只用于与旧报告比较；
- `chemical_ged_v2`：node match 仍以 atomic number 为核心，edge match 使用不会混淆芳香键的离散 bond type，作为新模型的主 GED 指标。

两种 GED 不得混在同一结果列中。论文引用旧数值时标记 legacy；所有新消融在同一候选集上同时报告两个版本，最终结构结论以 `chemical_ged_v2` 和固定配置的 Morgan Tanimoto 为准。

本项目的 normalized GED 明确定义为：

\[
\operatorname{nGED}(G_q,G_c)=
\frac{\operatorname{GED}(G_q,G_c)}
{|V_q|+|V_c|+|E_q|+|E_c|},
\]

其中主报告的 GED 使用 `chemical_ged_v2`：节点替换比较 atomic number，边替换比较离散 bond type；节点/边插入、删除及不匹配的默认代价均为 1。若分母为 0 或 GED 超时，该样本记为不可计算，不以 0 填充。描述局部环境的 degree、valence、H count、aromatic 和 ring 属性进入 encoder 与 Tanimoto 关系监督，但不额外叠加到 GED edit cost，避免对同一拓扑差异重复计费。

## 评估指标与验收标准

每个模型至少报告：

- Top-1、mean Top-5、best@5、best@10 Morgan Tanimoto；
- Top-1 和 best@K raw/normalized GED；
- exact match Recall@1/5/10；
- Bemis–Murcko scaffold Recall@1/5/10；
- structure Recall@K，即 Top-K 中至少存在一个 Morgan Tanimoto 不低于 0.3/0.5 的候选；
- 查询与候选的 heavy-atom count gap、元素集合 Jaccard 和分子量绝对误差；
- sampled molecule pairs 上 embedding cosine 与 Morgan Tanimoto 的 Spearman correlation；
- 光谱侧 descriptor 的 MAE/F1；
- 查询延迟、索引构建时间、索引大小和训练显存。

结构改造通过验收必须同时满足：

- 在同一个冻结测试集和同一个候选库上，Top-1 Tanimoto 相比 spectral-only baseline 提升，paired bootstrap 95% confidence interval 的下界大于 0；
- normalized Top-1 GED 不劣于 baseline，并报告 paired confidence interval；
- 在包含真实分子的同域验证集上，exact-match Recall@10 的变化不低于 -1 个百分点；若 paired bootstrap 95% confidence interval 无法排除更大退化，则不得把该配置作为最终模型；
- embedding–Tanimoto Spearman correlation 为正，且显著高于 spectral-only baseline；
- 不读取查询的真实 SMILES、分子式、原子数或分子量；
- 所有索引/模型 schema 校验、数据 join 校验和最小端到端测试通过。

ChEMBL 相对 QM9S 的候选覆盖不足必须单独报告。若扩大结构候选库后指标提升，应分别给出“模型改进”和“候选库覆盖改进”的对照，不能把二者混为同一个贡献。

## 测试与错误处理

单元测试首先覆盖可由小型合成数据验证的行为：

- SMILES 转图后的节点、边、元素计数和键类型正确；
- aromatic bond 与 single bond 在图特征和 `chemical_ged_v2` 中具有不同类别，而 `legacy_ged_v1` 仍可复现旧行为；
- 两轮消息传递后的 multi-scale readout 分别包含 0-hop、1-hop 和 2-hop 表示；
- fingerprint metadata 明确固定 radius、bit 数、bond type 和 chirality 配置；
- `mol_id` 顺序打乱时仍能正确 join，重复或缺失 ID 会失败；
- graph collate 在不同节点数、单原子和无键分子上 shape 正确；
- SpectrumEncoder、GraphEncoder 和 projection head 输出 shape、有限值和单位范数正确；
- 相同 canonical SMILES 不会在 loss 中被当作负样本；
- 提高结构相似样本的模型相似度会降低 structure loss；
- DDP gather 的目标索引与全局 batch 对齐；
- checkpoint 或 featurizer version 不匹配时索引加载失败；
- 推理 API 只传光谱也能完成 Top-K 图检索。

集成测试使用数个手工 SMILES 和随机/规则生成的小光谱验证训练一步、保存、加载、建索引和检索闭环。正式数据只在服务器环境运行 smoke test 和训练，不在本机加载。

错误处理遵循 fail-fast：非法 SMILES、非有限光谱、长度不符、空数据集、descriptor 统计缺失、索引与 checkpoint 不兼容均给出可定位错误。被排除样本的数量和原因写入预处理摘要，但不得静默改变 spectra 与 SMILES 的对应关系。

## 预期研究结论与限制

若实验通过验收，论文可以主张：与只学习谱图增强不变性的 SimCLR 相比，跨模态图对齐和连续结构关系监督显著提高了候选的结构相似性；模型在推理时仍只需要光谱，并可检索没有配套 Raman 光谱的结构数据库。

论文不能主张从单一振动光谱唯一恢复分子结构，也不能把预测描述符当作真实已知条件。对于不同结构产生近似光谱、立体异构难区分、实验域偏移以及候选库缺少相似分子的情况，应作为明确限制报告。

## 方法依据

- [VibraCLIP](https://doi.org/10.1039/D5DD00269A) 将分子图、IR 和 Raman 编码到共享空间，并在检索时用光谱查询分子图，是本设计跨模态主干的直接依据。
- [SMEN](https://doi.org/10.1039/D4DD00135D) 使用分子图 encoder 与光谱 Transformer 进行 IR–结构对比对齐，支持以光谱对分子库排序。
- [iMolCLR](https://arxiv.org/abs/2202.09346) 指出普通分子对比学习中的 faulty negatives，并使用 cheminformatics similarity 对负样本加权，是连续结构关系监督的依据。
- [MolCLR](https://arxiv.org/abs/2102.10056) 证明 GNN 对比表示能够保留分子拓扑和功能团相似性，为二维 GIN 图分支提供基础。

## 完成判据

- 新模型能够用配对光谱和 SMILES 完成双编码器训练，并在推理时只接收光谱；
- 候选索引完全由分子图构建，带有可验证的 schema/version metadata；
- 训练目标包含精确跨模态对齐、光谱增强一致性、结构关系保持和描述符辅助监督；
- 数据按 `mol_id` 严格对齐，不存在结构标签泄漏；
- 新旧基线在相同 split、候选库和指标实现上完成消融对比；
- 结构指标达到前述统计验收门槛；
- 本地单元/集成测试与服务器 smoke test 通过，正式训练日志和 checkpoint 可复现；
- 未经额外授权，不读取、上传、删除或覆盖大型数据与既有服务器实验产物。
