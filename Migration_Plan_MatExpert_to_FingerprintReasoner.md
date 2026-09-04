# 将 MatExpert 方法迁移到 Fingerprint Reasoner 的详细方案

## 适配目标：KDD Applied Track

---

## 一、两篇论文的核心对比

### MatExpert (ICLR 2025)
- **领域**: 材料发现 (Materials Discovery)
- **核心方法**: 三阶段框架 —— Retrieval（对比学习检索）→ Transition（思维链 CoT 推理改造路径）→ Generation（LLM 生成晶体结构）
- **关键技术**:
  - T5 编码器 + SimCLR 对比学习 → 属性描述与结构描述对齐
  - GPT-4 生成 ground-truth transition pathway → 微调 LLaMA 模型蒸馏
  - pyMatgen 库将 ALX 表示转为 CIF 格式
- **数据集**: NOMAD 数据库 (2,886,120 材料)
- **评测指标**: Validity, Coverage, Distribution, Stability, Novelty

### Fingerprint Reasoner (当前论文)
- **领域**: 拉曼光谱有机化合物识别
- **核心方法**: PCA+kNN 候选筛选 → LLM 知识增强推理 (CoT)
- **关键技术**:
  - PCA 降维 (3501→400 维) + L1-norm + kNN 候选选择
  - 200+ 条光谱-结构关联规则嵌入 prompt
  - Checklist 双重校验机制
- **数据集**: QM9S (50+ 聚合物光谱)
- **评测指标**: 准确率 (Accuracy %)

---

## 二、迁移的核心思路：为什么适合 Applied Track

Applied Track 关注：**(1) 部署系统 / 工业应用；(2) 跨学科新颖应用；(3) 新数据集/基准；(4) 全面实验分析**。

迁移后的论文定位：

> **"SpectraExpert: Decomposing Spectral Analysis by Mimicking Human Experts — A Retrieval-Transition-Reasoning Framework for Molecular Identification via IR/Raman Spectroscopy"**

核心卖点：
| Applied Track 维度 | 对应内容 |
|---|---|
| **跨学科新颖应用** | 将材料科学中的对比学习+LLM 三阶段框架首次引入光谱分析领域 |
| **新数据集/基准** | 构建大规模 HITRAN/QM9S 融合光谱指纹基准数据集 |
| **部署系统** | 可现场部署的光谱识别系统，替代昂贵实验室设备 |
| **全面实验分析** | 对比 PCA-kNN、纯 CoT、纯 kNN 与 SpectraExpert 在多项指标上的表现 |

---

## 三、详细迁移方案

### 3.1 总体架构映射

```
MatExpert (材料发现)                Fingerprint Reasoner (光谱识别)            迁移后架构
─────────────────────────        ──────────────────────────                ───────────────
Stage 1: Retrieval               PCA+kNN Candidate Selection                Stage 1: Contrastive Retrieval
  (对比学习找相似材料)             (PCA降维+kNN找候选)                         (对比学习找相似光谱)

Stage 2: Transition              (无对应阶段 — 直接跳到推理)                   Stage 2: Spectral Transition
  (CoT推理改造路径)               (无中间改造步骤)                             (CoT推理光谱→结构差异)

Stage 3: Generation              LLM Knowledge-Augmented CoT                Stage 3: Guided Reasoning
  (LLM生成CIF结构)                (知识增强推理识别)                           (LLM推理分子结构)
```

### 3.2 Stage 1: 对比学习检索替代 PCA-kNN

#### 当前 Fingerprint Reasoner 的做法
- PCA 降维到 400 维 → L1 距离 → kNN 选 Top-K 候选
- 公式: `S = argtopS { ||(x_ref/max(x_ref))W - (x_j/max(x_j))W||_1 }`

#### 迁移 MatExpert 对比学习方法

**目标**: 将光谱数据和分子结构描述嵌入同一空间，通过对比学习对齐。

**具体实现**:

```python
# 参考 MatExpert/retrieval/utils.py 中的 SimCLR 架构
class SpectralSimCLR(nn.Module):
    """
    对比学习框架: 将光谱指纹与分子描述嵌入共享空间
    类比 MatExpert 中 property embedding ↔ structure embedding 的对齐
    """
    def __init__(self, tokenizer, projection_dim=256):
        super().__init__()
        # Encoder 1: 处理光谱数据 (将峰位+强度转为文本描述或直接用光谱向量)
        self.encoder1 = T5Encoder(tokenizer)  # 光谱描述编码器
        # Encoder 2: 处理分子结构描述 (SMILES + 官能团信息)
        self.encoder2 = T5Encoder(tokenizer)  # 分子描述编码器

        self.projector1 = nn.Sequential(
            nn.Linear(768, 768, bias=False),
            nn.ReLU(),
            nn.Linear(768, projection_dim, bias=False),
        )
        self.projector2 = nn.Sequential(
            nn.Linear(768, 768, bias=False),
            nn.ReLU(),
            nn.Linear(768, projection_dim, bias=False),
        )

    def forward(self, spectrum_text, molecule_text, mask_s, mask_m):
        h_s = self.encoder1(spectrum_text, mask_s)
        h_m = self.encoder2(molecule_text, mask_m)
        # Mask-pooling, 同 MatExpert
        z_s = self.projector1(h_s)
        z_m = self.projector2(h_m)
        z_s = (z_s * mask_s.unsqueeze(-1)).sum(dim=-2) / mask_s.sum(dim=-1, keepdim=True)
        z_m = (z_m * mask_m.unsqueeze(-1)).sum(dim=-2) / mask_m.sum(dim=-1, keepdim=True)
        return z_s, z_m
```

**数据准备**:
```
Property Description (类比 MatExpert 的材料属性):
  "Below is a Raman spectrum of an organic compound.
   Peak positions: 483, 515, 600, 627, 654, 674, 779, 813, 869, 934, 1014,
                   1040, 1115, 1135, 1224, 1350, 1381, 1524, 1568, 1600 cm⁻¹.
   Strong peaks at: 1350, 1524, 1568, 1600 cm⁻¹.
   The compound contains functional groups in the carbonyl region."

Structure Description (类比 MatExpert 的晶体结构描述):
  "Citric acid (C6H8O7) is a tricarboxylic acid with three carboxyl groups
   and one hydroxyl group attached to a central carbon. The molecular weight
   is 192.12 g/mol. Key vibrational modes include O-H bending, C=O stretching,
   and C-C stretching."
```

**对比损失函数** (同 MatExpert 公式 1):
```
L = -log [ exp(sim(q_i, m_i)/τ) / Σ exp(sim(q_i, m_k)/τ) ]
```
其中 `q_i` = 光谱描述嵌入, `m_i` = 对应分子结构嵌入, `τ` = 温度参数

**训练流程**:
1. 从 HITRAN/QM9S 数据集提取光谱数据
2. 用 RoboCrystallographer 类似的方法（或 RDKit + 自定义模板）生成分子描述文本
3. 构建正负样本对：同一分子的光谱描述和结构描述为正对，其他为负对
4. 用 NT-Xent Loss 训练 T5 编码器
5. 推理时：将查询光谱编码后，在数据库中搜索最近邻

**优势对比 PCA-kNN**:
- PCA-kNN: 仅依赖峰位数值匹配，丢失峰形、强度比等信息
- 对比学习: 可将峰的物理含义（官能团、化学键）编码进嵌入空间，语义层面更准确

### 3.3 Stage 2: 新增 "Spectral Transition" 阶段

这是 MatExpert 独有的核心创新，当前 Fingerprint Reasoner **没有对应阶段**。

#### MatExpert 的做法
- 用 GPT-4 生成 transition pathway（从材料 A 改造成材料 B 的步骤）
- 微调小模型 (LLaMA-2/7B) 蒸馏 GPT-4 的知识
- Prompt 模板:
  ```
  I have two materials: <formula_source> and <formula_target>.
  Based on the descriptions and properties, can you summarize the main
  reasons for the differences in properties when transitioning from
  <formula_source> to <formula_target>?
  ```

#### 迁移到光谱分析

**核心思想**: 给定查询光谱和候选分子，让 LLM 推理"光谱特征如何对应到分子结构差异"。

```
Stage 2 Prompt Template:
  "I have a Raman spectrum with peaks at: [peak positions].
   The closest matching molecules in the database are:
   - Candidate 1: [SMILES] [官能团描述]
   - Candidate 2: [SMILES] [官能团描述]
   - Candidate 3: [SMILES] [官能团描述]

   Based on spectroscopic principles, analyze:
   1. Which functional groups are indicated by the peak patterns?
   2. How do the differences between candidates explain the spectral features?
   3. What structural modifications would be needed to transform each
      candidate to match the observed spectrum?

   Provide a transition analysis for each candidate."
```

**训练数据构建** (参考 MatExpert 的 `data_second/generate_data_llm.py`):
```python
# 伪代码: 用 GPT-4 生成 ground-truth spectral transition pathway
import openai

def generate_transition_pathway(query_spectrum, candidates):
    """
    类比 MatExpert 中 GPT-4 生成材料改造路径
    这里生成 '光谱 → 结构' 的推理路径
    """
    prompt = f"""
    You are a spectroscopy expert. Given:
    - Query spectrum peaks: {query_spectrum.peaks}
    - Candidate molecules: {candidates.smiles}

    Analyze the spectral-to-structural transition:
    1. Identify functional groups from peaks
    2. Match peaks to vibrational modes of each candidate
    3. Score each candidate based on peak coverage
    4. Explain discrepancies and propose structural adjustments
    """
    response = openai.ChatCompletion.create(model="gpt-4", messages=[...])
    return parse_transition(response)
```

**微调策略** (参考 MatExpert 的 `llama_stage/mp_train.yaml`):
- 用 LLaMA-Factory 框架微调 LLaMA-2/7B 或 Qwen-7B
- 训练数据: GPT-4 生成的 transition pathway (约 10k-50k 条)
- Loss: 标准语言建模 loss
- LoRA 参数高效微调

### 3.4 Stage 3: 增强推理生成

#### 当前 Fingerprint Reasoner 的做法
- 5 步 CoT prompt: Knowledge Retrieval → Checklist 1 → Correlation Analysis → Checklist 2 → Answer
- 直接匹配峰位和已知振动模式

#### 迁移后增强版

```
Stage 3 Prompt Template (融合 MatExpert 的 ALX Generation 思想):

  "Based on the transition analysis, generate a detailed structural
   interpretation for the molecule matching this spectrum:

   Step 1: List all identified functional groups and their peak assignments
   Step 2: Generate the SMILES representation based on structural constraints
   Step 3: Verify the SMILES by computing expected Raman peaks using
           group frequency correlation tables
   Step 4: Compare computed peaks with observed peaks
   Step 5: Final identification with confidence score

   Observed spectrum: [peaks]
   Transition analysis: [Stage 2 output]
   Empirical rules: [knowledge base]
   Checklist: [verification criteria]
   "
```

**关键增强**:
- MatExpert 中的 ALX Representation → 转为 SMILES 生成 + 振动模式验证
- 引入 RDKit 计算分子指纹，与光谱特征交叉验证
- 输出不仅是分子名称，还包括置信度和结构不确定性估计

---

## 四、工程实现架构

### 4.1 目录结构

```
IRfingerprints/
├── proj/
│   ├── fingerprint_recognition.py      # 原有代码
│   └── spectra_expert/                 # 新增模块
│       ├── __init__.py
│       ├── stage1_retrieval.py         # 对比学习检索
│       │   ├── spectral_dataset.py     # Dataset 类
│       │   ├── spectral_simclr.py      # SimCLR 模型
│       │   ├── train_retrieval.py      # 训练脚本
│       │   └── inference_retrieval.py  # 检索推理
│       ├── stage2_transition.py        # 光谱→结构过渡推理
│       │   ├── generate_transition.py  # GPT-4 数据生成
│       │   ├── fine_tune_transition.py # LLaMA 微调
│       │   └── inference_transition.py # 推理
│       ├── stage3_reasoning.py         # 引导推理生成
│       │   ├── knowledge_base.py       # 光谱-结构规则库
│       │   ├── prompt_templates.py     # Prompt 模板
│       │   └── inference_reasoning.py  # 推理
│       ├── data_processing/
│       │   ├── hitran_parser.py        # HITRAN .data 文件解析
│       │   ├── qm9s_loader.py          # QM9S 数据加载
│       │   ├── molecule_describer.py   # RDKit 分子描述生成
│       │   └── spectrum_formatter.py   # 光谱格式化
│       └── evaluation/
│           ├── metrics.py              # 评估指标
│           └── benchmark.py            # 基准测试
├── data/
│   ├── hitran/                         # HITRAN 原始数据
│   ├── qm9s/                           # QM9S 光谱数据
│   ├── processed/                      # 处理后的数据
│   │   ├── spectral_pairs.csv          # 光谱-分子配对
│   │   ├── molecule_descriptions.csv   # 分子描述文本
│   │   └── transition_data.json        # Transition pathway 数据
│   └── baselines/                      # 基线模型数据
├── llama_stage/                        # LLaMA 微调配置 (参考 MatExpert)
│   ├── se_train.yaml                   # 训练配置
│   ├── se_inference.yaml               # 推理配置
│   └── se_prediction_test.yaml         # 测试配置
├── requirements.txt
└── README.md
```

### 4.2 核心代码映射

| MatExpert 文件 | 对应新文件 | 映射关系 |
|---|---|---|
| `retrieval/utils.py` (SimCLR) | `spectra_expert/stage1_retrieval/spectral_simclr.py` | 替换 T5 输入为光谱描述 |
| `retrieval/main.py` | `spectra_expert/stage1_retrieval/train_retrieval.py` | 训练循环复用 |
| `retrieval/add_description.py` | `spectra_expert/data_processing/molecule_describer.py` | CIF→描述 改为 SMILES→描述 |
| `basic_eval.py` | `spectra_expert/evaluation/metrics.py` | 晶体有效性→光谱匹配准确率 |
| `llama_stage/mp_train.yaml` | `llama_stage/se_train.yaml` | LLaMA-Factory 配置复用 |
| `data_second/generate_data_llm.py` | `spectra_expert/stage2_transition/generate_transition.py` | GPT-4 数据生成 |

### 4.3 数据处理管道

```python
# 基于 fingerprint_recognition.py 中已有的 HITRAN 数据解析
# 扩展为完整的对比学习数据对

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

def build_spectral_pairs(hitran_folder, output_csv):
    """
    从 HITRAN .data 文件构建 光谱-分子描述 配对数据
    类比 MatExpert 中的 train_desc.csv 构建流程
    """
    column_names = [
        'molecule_id', 'nu', 'sw', 'A', 'gamma_air', 'gamma_self',
        'elower', 'n_air', 'delta_air', 'global_upper_quanta',
        'global_lower_quanta', 'local_upper_quanta', 'local_lower_quanta',
        'ierr1', 'ierr2', 'ierr3', 'ierr4', 'ierr5', 'ierr6',
        'iref1', 'iref2', 'iref3', 'iref4', 'iref5', 'iref6'
    ]

    pairs = []
    for filename in os.listdir(hitran_folder):
        if filename.endswith('.data'):
            data = pd.read_csv(
                os.path.join(hitran_folder, filename),
                sep=r'\s+', comment='#', names=column_names
            )

            # 提取波数和强度
            wavenumber = pd.to_numeric(data['nu'], errors='coerce').dropna()
            intensity = pd.to_numeric(data['sw'], errors='coerce').dropna()

            # 峰位提取 (类比 MatExpert 的属性提取)
            peak_positions = extract_peaks(wavenumber, intensity)

            # 构建光谱描述 (类比 MatExpert 的 property description)
            spectrum_desc = build_spectrum_description(peak_positions, intensity)

            # 从文件名或 molecule_id 构建分子描述
            molecule_name = extract_molecule_name(filename)
            molecule_desc = build_molecule_description(molecule_name)

            pairs.append({
                'spectrum_desc': spectrum_desc,
                'molecule_desc': molecule_desc,
                'molecule_id': molecule_name,
                'peak_positions': peak_positions.tolist(),
                'raw_wavenumber': wavenumber.tolist(),
                'raw_intensity': intensity.tolist(),
            })

    pd.DataFrame(pairs).to_csv(output_csv, index=False)

def extract_peaks(wavenumber, intensity, threshold=0.1):
    """提取显著峰位"""
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(intensity, height=np.max(intensity) * threshold)
    return wavenumber[peaks]

def build_spectrum_description(peaks, intensities):
    """构建光谱文本描述，类比 MatExpert 中的属性描述"""
    strong_peaks = peaks[intensities > np.mean(intensities) * 2]
    desc = f"Raman spectrum with {len(peaks)} peaks. "
    desc += f"Peak positions: {', '.join(map(str, np.round(peaks, 1)))} cm⁻¹. "
    desc += f"Strong peaks at: {', '.join(map(str, np.round(strong_peaks, 1)))} cm⁻¹. "
    return desc
```

---

## 五、评估体系设计

### 5.1 借用 MatExpert 的评估框架

| MatExpert 指标 | 光谱分析对应指标 | 说明 |
|---|---|---|
| **Struct Validity** | **Spectral Validity** | 识别结果的光谱是否与真实光谱匹配 (峰位误差 < 10 cm⁻¹) |
| **Comp Validity** | **Chemical Validity** | 识别的分子是否是化学合理的 SMILES |
| **Coverage Recall** | **Spectral Coverage** | 测试集中多少光谱被正确覆盖 |
| **Coverage Precision** | **Identification Precision** | 预测结果中多少是正确匹配 |
| **Wdist (density)** | **Wdist (peak intensity)** | Wasserstein 距离衡量峰强度分布 |
| **Wdist (num_elems)** | **Wdist (num_peaks)** | Wasserstein 距离衡量峰数量分布 |
| **Stability (M3GNet)** | **Confidence Calibration** | 预测置信度与准确率的校准 |
| **Novelty** | **Generalization** | 对未见分子类别的泛化能力 |

### 5.2 实验设计

```
实验 1: 无条件检索 (Unconditional Retrieval)
  类比 MatExpert 的无条件生成
  - 随机采样光谱，评估检索阶段的 Top-K 准确率
  - 对比: PCA-kNN vs 对比学习检索

实验 2: 条件检索 (Conditional Retrieval)
  类比 MatExpert 的条件生成
  - 给定目标官能团/化学键约束，评估检索能力
  - 对比: 有无 Transition 阶段的效果差异

实验 3: 消融实验 (Ablation Studies)
  类比 MatExpert 表 3 和图 6
  - w/o Contrastive Learning (纯 kNN)
  - w/o Transition Stage (直接推理)
  - w/o Checklist (去掉验证步骤)
  - 不同编码器 (BERT vs T5)

实验 4: 条件满足率 (Condition Satisfaction)
  类比 MatExpert 图 4
  - 给定峰位范围约束 → 满足率
  - 给定官能团约束 → 满足率
  - 给定分子大小约束 → 满足率

实验 5: 多样性和新颖性 (Diversity & Novelty)
  类比 MatExpert 图 5
  - 检索结果的多样性 (不同化学空间覆盖)
  - 对未见光谱的泛化能力
```

### 5.3 基线方法

| 基线 | 来源 | 说明 |
|---|---|---|
| PCA-kNN | Fingerprint Reasoner | 当前方法 |
| CoT (system) | Fingerprint Reasoner | 纯思维链推理 |
| CDVAE | MatExpert | 改为光谱 VAE |
| LM-CH / LM-AC | MatExpert | 字符级/原子级 Tokenizer 的 LLM |
| Crystal-LLM | MatExpert | 微调 LLM 基线 |

---

## 六、论文结构建议

```
Title: SpectraExpert: Decomposing Spectral Analysis by Mimicking Human Experts
       for Molecular Identification via Retrieval-Transition-Reasoning

1. Introduction
   - 光谱分析的痛点: 环境依赖性强、缺乏大规模标注数据
   - 现有方法的局限: PCA-kNN 丢失语义, LLM 幻觉
   - 我们的方法: 三阶段专家工作流
   - 贡献列表 (类比 MatExpert 的 3 点贡献)

2. Related Work
   - AI in Spectroscopy
   - LLMs in Chemistry
   - Contrastive Learning for Scientific Data
   - 对比分析 MatExpert、Crystal-LLM、Fingerprint Reasoner

3. Methodology
   3.1 First Stage: Contrastive Retrieval
       - 光谱描述和分子描述的 T5 编码
       - SimCLR 对比学习框架
       - NT-Xent Loss 训练
   3.2 Second Stage: Spectral Transition
       - GPT-4 生成 ground-truth 推理路径
       - 小模型蒸馏
       - Prompt 模板设计
   3.3 Third Stage: Guided Reasoning
       - ALX→SMILES 映射
       - 知识增强推理
       - 双重 Checklist 验证

4. Data Collection & Pre-processing
   4.1 HITRAN 数据库处理
   4.2 QM9S 数据集扩展
   4.3 分子描述生成 (RDKit + 模板)
   4.4 光谱-分子配对构建

5. Experiments
   5.1 Datasets and Baselines
   5.2 Unconditional Retrieval
   5.3 Conditional Identification
   5.4 Ablation Studies
   5.5 Case Study

6. Analysis
   6.1 Diversity and Generalization
   6.2 Ablation: Impact of Each Stage
   6.3 Encoder Comparison (BERT vs T5)

7. Conclusion
```

---

## 七、从现有代码出发的具体实现步骤

### Step 1: 数据准备 (1-2 周)
```bash
# 1. 解析 HITRAN .data 文件 → 结构化光谱数据
python proj/spectra_expert/data_processing/hitran_parser.py \
    --input ./data/hitran/ --output ./data/processed/hitran_parsed.csv

# 2. 加载 QM9S 数据集 → 光谱-分子配对
python proj/spectra_expert/data_processing/qm9s_loader.py \
    --input ./data/qm9s/ --output ./data/processed/qm9s_pairs.csv

# 3. 用 RDKit 生成分子描述
python proj/spectra_expert/data_processing/molecule_describer.py \
    --input ./data/processed/spectral_pairs.csv \
    --output ./data/processed/spectral_pairs_desc.csv
```

### Step 2: 对比学习训练 (2-3 周)
```bash
# 训练光谱对比学习模型
python proj/spectra_expert/stage1_retrieval/train_retrieval.py \
    --base_model t5-base \
    --data_dir ./data/processed/ \
    --epochs 50 \
    --batch_size 32 \
    --temperature 0.1 \
    --name spectral_simclr
```

### Step 3: Transition 数据生成 (1-2 周)
```bash
# 用 GPT-4 生成 transition pathway
python proj/spectra_expert/stage2_transition/generate_transition.py \
    --input ./data/processed/spectral_pairs_desc.csv \
    --output ./data/processed/transition_data.json
```

### Step 4: LLM 微调 (2-3 周)
```bash
# 用 LLaMA-Factory 微调
llamafactory-cli train llama_stage/se_train.yaml
```

### Step 5: 端到端推理 + 评估 (1-2 周)
```bash
# 端到端推理
python proj/spectra_expert/inference_pipeline.py \
    --query_spectra ./data/test/ \
    --retrieval_model ./saved_models/spectral_simclr_best.pth \
    --llm_model ./saved_models/se_llama2 \
    --output ./results/

# 评估
python proj/spectra_expert/evaluation/benchmark.py \
    --results ./results/ \
    --ground_truth ./data/test/ground_truth.csv
```

---

## 八、关键创新点和差异化

### vs. Fingerprint Reasoner (原方法)
| 维度 | 原方法 | 迁移后 |
|---|---|---|
| 检索 | PCA-kNN (数值匹配) | 对比学习 (语义匹配) |
| 推理 | 一次性 CoT | 三阶段迭代精炼 |
| 知识利用 | 静态规则库 | 动态 Transition 推理 |
| 验证 | Checklist | Checklist + 交叉验证 |

### vs. MatExpert (原方法)
| 维度 | MatExpert | 迁移后 |
|---|---|---|
| 领域 | 材料发现 | 光谱分析 |
| 输入 | 材料属性描述 | 光谱峰位数据 |
| 输出 | CIF 晶体结构 | SMILES + 分子结构 |
| 结构描述 | RoboCrystallographer | RDKit + 振动模式描述 |
| 生成阶段 | ALX→CIF | 推理路径→SMILES |

### 独特贡献
1. **首个将对比学习检索引入光谱分析的框架**
2. **将"Transition"概念从材料改造扩展到光谱-结构映射**
3. **HITRAN + QM9S 融合基准数据集**
4. **可部署的现场光谱识别系统**

---

## 九、风险与缓解

| 风险 | 缓解措施 |
|---|---|
| HITRAN 数据缺乏分子描述 | 用 RDKit + InChI/SMILES 反向推导，或从 NIST 数据库补充 |
| 光谱数据噪音大 | 加入数据增强: 高斯噪音、峰偏移、基线漂移 |
| 对比学习需要大量配对数据 | 先用 QM9S (小规模) 验证方法，再扩展到 HITRAN |
| LLM 幻觉问题 | 双重 Checklist + 规则库验证 + 置信度阈值过滤 |
| 计算资源限制 | LoRA 微调 + 7B 模型 + 梯度累积 (MatExpert 已验证此方案) |

---

## 十、时间线

| 阶段 | 时间 | 交付物 |
|---|---|---|
| 数据准备和处理 | 第 1-2 周 | 结构化数据集 |
| 对比学习检索实现 | 第 3-5 周 | 训练好的检索模型 |
| Transition 数据生成 | 第 6-7 周 | GPT-4 标注的 transition 数据 |
| LLM 微调 | 第 8-10 周 | 微调好的推理模型 |
| 端到端集成 | 第 11-12 周 | 完整推理管道 |
| 实验和评估 | 第 13-14 周 | 实验结果图表 |
| 论文写作 | 第 15-16 周 | 初稿 |

---

## 附录 A: MatExpert 关键代码片段引用

### A.1 SimCLR 模型 (retrieval/utils.py)
```python
class SimCLR(nn.Module):
    def __init__(self, tokenizer, projection_dim=256):
        # 两个 T5 编码器分别处理 property 和 structure
        self.encoder1 = T5Encoder(tokenizer)
        self.encoder2 = T5Encoder(tokenizer)
        # 投影头
        self.projector1 = nn.Sequential(...)
        self.projector2 = nn.Sequential(...)
    # Mask-pooling 聚合
```

### A.2 NT-Xent Loss (retrieval/utils.py)
```python
class NTXentLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        # InfoNCE 损失，正样本对角线，负样本其余位置
```

### A.3 Transition Prompt (论文第 5 页)
```
I have two materials: <formula_source> and <formula_target>.
Based on the descriptions and properties of the two materials below,
can you summarize the main reasons for the differences in properties
when transitioning from <formula_source> to <formula_target>?
```

### A.4 Generation Prompt (论文第 5 页)
```
Based on the information, for the new material, could you generate
a description of the lengths and angles of the lattice vectors and
then the element type and coordinates for each atom within the lattice?
```

---

## 附录 B: Fingerprint Reasoner Prompt 模板映射

| 原 Prompt 阶段 | 映射到三阶段 |
|---|---|
| Knowledge Retrieval (步骤 1-2) | Stage 1 检索 + Stage 2 知识增强 |
| Checklist 1 (步骤 2) | Stage 2 末尾验证 |
| Correlation Analysis (步骤 3) | Stage 2 核心推理 |
| Find candidate (步骤 4) | Stage 3 初选 |
| Checklist 2 (步骤 5) | Stage 3 最终验证 |