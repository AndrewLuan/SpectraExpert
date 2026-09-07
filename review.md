
        --========  Review Reports  ========--

The review report from reviewer #1:

*1: Is the paper relevant to Appliied Track?
  [_] No
  [X] Yes
  
*2: How innovative is the paper?
  [_] 6 (Very innovative)
  [_] 3 (Innovative)
  [X] -2 (Marginally)
  [_] -4 (Not very much)
  [_] -6 (Not at all)
  
*3: How would you rate the technical quality of the paper?
  [_] 6 (Very high)
  [_] 3 (High)
  [X] -2 (Marginal)
  [_] -4 (Low)
  [_] -6 (Very low)
  
*4: How is the presentation?
  [_] 6 (Excellent)
  [X] 3 (Good)
  [_] -2 (Marginal)
  [_] -4 (Below average)
  [_] -6 (Poor)
  
*5: Is the paper of interest to the ICDM community?
  [_] 3 (Yes)
  [X] 2 (May be)
  [_] 1 (No)
  [_] 0 (Not applicable)
  
*6: What is your confidence in your review of this paper?
  [_] 2 (High) -- I have substantial expertise in this area, e.g., I have published on closely related topics or regularly read papers in this area.
  [X] 1 (Medium) -- I am familiar with this area and related work, but I am not an expert on all aspects of the paper.
  [_] 0 (Low) -- I am only vaguely familiar with this area or important parts of the paper are outside my expertise.
  
*7: Overall recommendation
  [_] 6: must accept (in top 25% of ICDM accepted papers)
  [_] 3: should accept (in top 80% of ICDM accepted papers)
  [_] -2: marginal (in bottom 20% of ICDM accepted papers)
  [X] -4: should reject (below acceptance bar)
  [_] -6: must reject (unacceptable: too weak, incomplete, or wrong)
  
*8: Summary of the paper's main contribution and impact
  The paper introduces SpectraExpert, a pipeline for molecular identification from Raman spectra that mirrors human spectroscopists: a contrastively trained encoder for spectral retrieval on a 129k-sample QM9S database, a curated knowledge base of 181 Nakamoto-derived group-frequency rules for peak-to-functional-group analysis, and an LLM-based reasoning stage that selects or modifies candidate structures given retrieved spectra and rule-based evidence. The authors also construct a zero-shot cross-domain benchmark that trains retrieval on simulated small molecules (QM9S, ≤9 heavy atoms) and evaluates on an out-of-domain ChEMBL Raman subset with up to 21 heavy atoms, stressing domain shift. They report aggregate graph edit distance (GED), Tanimoto similarity, exact match, and SMILES validity across retrieval baselines and five LLMs in direct, rule-augmented (“Knowledge”), and full agentic modes.
The impact claim is that SpectraExpert offers a reproducible applied workflow and benchmark for spectrum-based molecular identification under domain shift and that integrating retrieval and spectroscopic rules improves interpretability and structured-output reliability relative to direct LLM generation. However, the system does not achieve any exact matches on the cross-domain ChEMBL benchmark, and the evidence for robust identification performance remains limited; the primary value is in problem framing and tooling, not in decisive or comparable performance gains.
本文介绍了 SpectraExpert，这是一个模拟人类光谱学家工作流程的拉曼光谱分子识别管线。该系统包含三个核心组件：一个基于 12.9 万个样本的 QM9S 数据库进行对比学习训练的编码器，用于执行“峰-官能团”分析的精选知识库（包含 181 条源自 Nakamoto 的基团频率规则），以及一个基于大语言模型（LLM）的推理阶段，该阶段根据检索到的光谱和基于规则的证据来筛选或修改候选分子结构。作者还构建了一个零样本跨域基准测试：利用模拟小分子（QM9S，重原子数 ≤9）进行检索训练，并在包含多达 21 个重原子的 ChEMBL 拉曼子集（属于域外数据）上进行评估，以此突出域偏移（domain shift）带来的挑战。研究报告了在直接生成、规则增强（“Knowledge”）和全智能体（agentic）模式下，针对各种检索基线和五种 LLM 的综合图编辑距离（GED）、Tanimoto 相似度、精确匹配率及 SMILES 有效性指标。
该研究的核心贡献在于，SpectraExpert 为存在域偏移情况下的光谱分子识别提供了一种可复现的应用工作流与基准测试方法；此外，与直接利用 LLM 生成相比，将检索机制与光谱学规则相结合，提升了系统的可解释性及结构化输出的可靠性。然而，该系统在跨域 ChEMBL 基准测试中未能实现任何精确匹配，且关于其识别性能稳健性的证据尚显不足；其主要价值在于问题建模与工具构建，而非在性能上取得了决定性或显著优于现有方法的提升。

*9: Justification of your recommendation
  The recommendation of should reject (-4) is grounded in the following concerns:
Impact: The paper does not convincingly show that SpectraExpert advances practical molecular identification beyond strong retrieval baselines. There is no comparison against other existing methods in the literature. This is the most outstanding concern.
Novelty: The main novelty lies in composing existing ingredients. Retrieval uses a relatively standard 1D ViT encoder, and the LLM orchestration follows established tools (MRKL, Toolformer, ReAct), while MatExpert-like retrieval-plus-reasoning systems already exist in materials discovery. The curated Nakamoto rules and their integration into prompts are useful engineering contributions, but largely incremental.
Soundness and rigor: The methodology is carefully motivated, but the experimental evidence falls short of demonstrating that SpectraExpert is an effective solution for molecular identification. On the core benchmark, all methods, including SpectraExpert’s agentic pipeline, fail to achieve any exact matches, and mean GED and Tanimoto are only modestly improved in some settings.
Reproducibility requires access to proprietary LLMs, several critical aspects of the LLM orchestration, such as exact prompts, are not reported.
建议拒稿（评分 -4）主要基于以下考量：
影响（Impact）：该论文未能令人信服地证明 SpectraExpert 在分子识别方面超越了现有的强检索基线（strong retrieval baselines）。文中未与文献中的其他现有方法进行对比，这是最主要的问题。
创新性（Novelty）：其主要创新点在于对现有组件的组合。检索部分使用了相对标准的 1D ViT 编码器，LLM 编排（orchestration）遵循了既有工具（如 MRKL、Toolformer、ReAct）的模式，且材料发现领域已存在类似 MatExpert 的“检索+推理”系统。虽然精心整理的 Nakamoto 规则及其在提示词（prompt）中的整合属于有益的工程贡献，但总体而言仅具增量式改进的性质。
合理性与严谨性（Soundness and rigor）：尽管方法论的构建动机充分，但实验证据不足以证明 SpectraExpert 是分子识别的有效解决方案。在核心基准测试中，包括 SpectraExpert 智能体（agentic）流程在内的所有方法均未能实现任何精确匹配；在某些设置下，平均 GED 和 Tanimoto 指标也仅有微弱改善。
可复现性（Reproducibility）：复现该方法需要使用专有 LLM，且 LLM 编排的若干关键细节（如具体的提示词）未在文中披露。

*10: Three strong points of this paper (please number each point)
S1 - The paper tackles a genuinely high-impact, underexplored applied data mining problem: molecular structure identification from Raman spectra under realistic domain shift.

S2 - The paper presents a clear three-stage architecture that mirrors expert spectroscopist workflows.

S3 - The paper introduces a cross-domain benchmark with well-defined metrics (GED, Tanimoto, SMILES validity) and systematically reports results across multiple retrieval baselines and several modern LLMs.
S1 - 本文探讨了一个极具影响力但尚未得到充分研究的应用数据挖掘问题：在存在实际领域偏移（domain shift）的情况下，基于拉曼光谱进行分子结构识别。

S2 - 本文提出了一种清晰的三阶段架构，该架构模拟了光谱学专家的工作流程。

S3 - 本文引入了一个跨领域基准测试，包含明确定义的评估指标（GED、Tanimoto 系数、SMILES 有效性），并系统地报告了基于多种检索基线及若干现代大语言模型（LLM）的实验结果。

*11: Three weak points of this paper (please number each point)
W1 - The empirical results are weak and inconclusive: the paper does not compare against at least three directly competing contemporaneous works: Vib2Mol (Lu et al., 2025), VibraCLIP (Rocabert-Oriols et al., 2025) and MolSpectLLM (Shen et al., 2025). No method in the paper achieves any exact matches on the main cross-domain benchmark, and observed GED/Tanimoto improvements for the agentic pipeline or rule-augmented prompting are modest, highly variable depending on the LLM, and evaluated on small, non-paired subsets (211 queries for Retrieval, 50 queries for LLMs), preventing strong performance claims.

W2 - The experimental design lacks statistical rigor and thorough ablations: there are no confidence intervals, significance tests, or multi-run analyses for key comparisons; ablations on the contributions of retrieval vs. rules vs. LLM reasoning are partial; and the agentic pipeline’s modification fallback generally degrades GED, suggesting the main innovation does not consistently help.

W3 - Reproducibility and methodological transparency are limited for the LLM components: prompts, decoding settings, and decision policies are only partially specified; reliance on proprietary frontier models hinders independent verification; and several important failures (e.g., Gemini’s zero Tanimoto, GPT-5’s very low parse rate, LLM knowledge better than Agentic metod on GED, but drops in % of valid molecules) are reported without systematic investigation of causes or mitigation strategies.

W1 - 实证结果薄弱且缺乏定论：该论文未与至少三项直接竞争的同期研究进行比较，包括 Vib2Mol (Lu et al., 2025)、VibraCLIP (Rocabert-Oriols et al., 2025) 和 MolSpectLLM (Shen et al., 2025)。论文中的任何方法在主要的跨领域基准测试中均未实现完全匹配；针对智能体（agentic）流水线或规则增强提示（rule-augmented prompting）所观察到的 GED/Tanimoto 指标改进幅度有限，且高度依赖于所使用的 LLM；此外，评估基于小规模、非配对的子集（检索任务 211 个查询，LLM 任务 50 个查询），这使得无法有力地宣称其性能优势。

W2 - 实验设计缺乏统计严谨性和详尽的消融分析：关键比较中未包含置信区间、显著性检验或多次运行分析；针对检索、规则与 LLM 推理各自贡献的消融分析不完整；且智能体流水线中的“修改回退”（modification fallback）机制通常会导致 GED 变差，这表明其主要创新点并未带来持续的性能提升。

W3 - LLM 组件的可复现性和方法透明度有限：提示词（prompts）、解码设置和决策策略仅有部分说明；对专有前沿模型的依赖阻碍了独立验证；此外，报告中提及了若干重大失败案例（例如 Gemini 的 Tanimoto 得分为零、GPT-5 的解析率极低、LLM 在 GED 指标上优于智能体方法但有效分子比例下降），却未对成因或缓解策略进行系统性探究。


*12: Is this submission among the best 10% of submissions that you reviewed for ICDM'26?
  [X] No
  [_] Yes
  
*13: Are the datasets used in the study correctly identified and referenced?
  [X] 3 Yes
  [_] 2 Partial
  [_] 1 No
  [_] 0 Not applicable
  
*14: If the authors use private data in the experiments, do they state that they will make the data publicly available in the camera-ready version?
  [_] 3 Yes
  [_] 2 Partial
  [_] 1 No
  [X] 0 Not applicable
  
*15: Are the competing methods used in the study correctly identified and referenced?
  [_] 3 Yes
  [X] 2 Partial
  [_] 1 No
  [_] 0 Not applicable
  
*16: Do the authors provide a link to their source code, or state that they will make the source code publicly available in the camera-ready version?
  [X] 3 Yes
  [_] 2 Partial
  [_] 1 No
  [_] 0 Not applicable
  
*17: Is the experimental design detailed enough to allow for reproducibility? (You can also include comments on reproducibility in the body of your review.)
  [_] 3 Yes
  [X] 2 Partial
  [_] 1 No
  [_] 0 Not applicable
  
*18: If the paper is accepted, which format would you suggest?
  [_] Regular Presentation
  [X] Short Presentation
  
*19: Detailed comments for the authors
  Please see weaknesses for more details.
Minor comments:
C1: Many reported gains (e.g., GED 13.52 for one LLM+knowledge batch vs ~16.9 for retrieval) are on small, non-paired batches and are difficult to interpret as strong, statistically supported improvements. The domain shift setup (QM9S to ChEMBL) is indeed challenging, but the paper does not adequately separate retrieval quality, LLM reasoning, and the inherent difficulty of the chemical space, nor does it provide robust significance testing, multiple seeds or thorough ablations on the agentic pipeline.
C2: Critical aspects of the LLM orchestration, such as exact prompts, decoding strategies, are only partially described, and the heavy dependence on proprietary frontier LLMs limits reproducibility and scientific transparency in a core part of the method. The benchmark is described as “reproducible” but in practice, replicating the main results would require access to the same commercial models and careful reimplementation of prompt templates and fallbacks, which goes beyond what is specified.
C3: The MODIFY, which activates when confidence < 0.6, is presented as a core agentic capability, yet it consistently degrades GED in all five tested models: Claude +4.9, DeepSeek +3.00, o3 +1.3 (Table IV). Overall, the method performance with different LLMs is not stable and not explained in the text.
请参阅“弱点”部分以了解更多详情。
次要意见：
C1：文中报告的许多性能提升（例如，某组“LLM+知识”方法的 GED 为 13.52，而检索方法的 GED 约为 16.9）是基于小规模、非配对样本得出的，难以被视为具有统计学支持的显著改进。尽管跨域设置（从 QM9S 到 ChEMBL）确实具有挑战性，但论文未能充分区分检索质量、LLM 推理能力以及化学空间本身的固有难度；此外，针对该智能体（agentic）流程，论文也缺乏稳健的显著性检验、多随机种子实验或详尽的消融实验。
C2：关于 LLM 编排的关键方面（如具体的提示词、解码策略等）仅有部分描述；且该方法的核心部分严重依赖专有的前沿 LLM，这限制了其可复现性和科学透明度。虽然该基准测试被称为“可复现”的，但在实际操作中，若要复现主要结果，不仅需要使用相同的商业模型，还需仔细重新实现提示词模板和回退（fallback）机制，而这些要求超出了文中明确说明的范围。
C3：当置信度低于 0.6 时触发的 MODIFY 操作被视为一项核心智能体能力，然而在所有五个测试模型中，它始终导致 GED 变差（即数值升高）：Claude +4.9、DeepSeek +3.00、o3 +1.3（见表 IV）。总体而言，该方法在使用不同 LLM 时的表现并不稳定，且文中未对此现象进行解释。
========================================================
The review report from reviewer #2:

*1: Is the paper relevant to Appliied Track?
  [_] No
  [X] Yes
  
*2: How innovative is the paper?
  [X] 6 (Very innovative)
  [_] 3 (Innovative)
  [_] -2 (Marginally)
  [_] -4 (Not very much)
  [_] -6 (Not at all)
  
*3: How would you rate the technical quality of the paper?
  [X] 6 (Very high)
  [_] 3 (High)
  [_] -2 (Marginal)
  [_] -4 (Low)
  [_] -6 (Very low)
  
*4: How is the presentation?
  [_] 6 (Excellent)
  [X] 3 (Good)
  [_] -2 (Marginal)
  [_] -4 (Below average)
  [_] -6 (Poor)
  
*5: Is the paper of interest to the ICDM community?
  [X] 3 (Yes)
  [_] 2 (May be)
  [_] 1 (No)
  [_] 0 (Not applicable)
  
*6: What is your confidence in your review of this paper?
  [X] 2 (High) -- I have substantial expertise in this area, e.g., I have published on closely related topics or regularly read papers in this area.
  [_] 1 (Medium) -- I am familiar with this area and related work, but I am not an expert on all aspects of the paper.
  [_] 0 (Low) -- I am only vaguely familiar with this area or important parts of the paper are outside my expertise.
  
*7: Overall recommendation
  [_] 6: must accept (in top 25% of ICDM accepted papers)
  [X] 3: should accept (in top 80% of ICDM accepted papers)
  [_] -2: marginal (in bottom 20% of ICDM accepted papers)
  [_] -4: should reject (below acceptance bar)
  [_] -6: must reject (unacceptable: too weak, incomplete, or wrong)
  
*8: Summary of the paper's main contribution and impact
  This paper presents SpectraExpert, a retrieval-augmented reasoning system for identifying molecular structures from Raman spectra. The framework emulates the workflow of human spectroscopists through a three-stage pipeline: (1) retrieval to narrow down candidate molecules; (2) structured spectral analysis matching; (3) LLM-guided reasoning that evaluates candidates, selects the best match, or proposes structural modifications.
This paper provides a well-engineered, end-to-end demonstration of how LLMs can be integrated into a scientific inverse problem workflow.
本文介绍了 SpectraExpert，这是一个利用检索增强推理技术，根据拉曼光谱识别分子结构的系统。该框架通过一个包含三个阶段的流程，模拟了人类光谱学家的工作方式：(1) 检索以缩小候选分子范围；(2) 结构化光谱分析与匹配；(3) 在大语言模型（LLM）引导下进行推理，评估候选对象、选出最佳匹配项或提出结构修改建议。
本文展示了一个设计精良的端到端案例，阐明了如何将大语言模型整合到科学反演问题的处理流程中。

*9: Justification of your recommendation
  This paper makes a significant contribution by providing a rigorously engineered, end-to-end demonstration of how LLMs can be integrated into a scientific inverse problem workflow. It is valuable not only to the spectroscopy community but also to the broader AI-for-science community working on LLM deployment in experimental sciences. Specify LLM inference parameters and random seeds for full reproducibility.
  本文做出了重要贡献，提供了一个经过严谨工程设计的端到端演示，展示了如何将大语言模型（LLM）集成到科学反问题的工作流程中。这不仅对光谱学界具有价值，对于致力于在实验科学领域部署大语言模型的更广泛的“AI 科学”（AI-for-Science）社群同样意义重大。请明确列出大语言模型的推理参数及随机种子，以确保结果完全可复现。

*10: Three strong points of this paper (please number each point)
  S1: The authors convincingly argue that direct LLM generation suffers from hallucination and lack of domain-specific reasoning, and that human spectroscopists follow a "retrieve-analyze-reason" workflow.
S2: The inclusion of five models from four major providers (OpenAI, Anthropic, Google, DeepSeek) lends generality to the conclusions.
S3: The authors provide an objective assessment of the paper's limitations and clearly define the scope and positioning of their work.
S1：作者有力地论证了直接利用大语言模型（LLM）生成内容存在“幻觉”及缺乏特定领域推理能力的问题，并指出人类光谱学家遵循的是“检索-分析-推理”的工作流程。
S2：该研究纳入了来自四大主流厂商（OpenAI、Anthropic、Google、DeepSeek）的五款模型，从而增强了结论的普适性。
S3：作者客观评估了论文的局限性，并明确界定了其研究的范围与定位。

*11: Three weak points of this paper (please number each point)
  W1: Inconsistent Evaluation Batches Hinder Fair Comparison Issue, for example, Retrieval is evaluated on all 211 queries, LLM Direct/Knowledge on randomly sampled 50-query subsets, and Agentic on a different 20-query subset. Because the query sets differ, direct GED comparisons between methods are statistically confounded. 
W2: As an applied data-mining work, practical deployment requires understanding cost-performance trade-offs. But the paper does not report Per-query inference time for each LLM, API costs (particularly relevant for commercial models), Total end-to-end latency for the complete Agentic pipeline.
W3: While the paper notes that MODIFY degrades GED, it does not analyze why modification fails.
W4: QM9S provides simulated spectra, while ChEMBL provides a mix of experimental and computed spectra. Although the cross-domain setup is valuable, the paper does not discuss the Systematic differences between simulated and real experimental spectra. Whether such differences bias retrieval.

W1：评估批次不一致导致无法进行公平比较。例如，检索（Retrieval）任务基于全部 211 个查询进行评估，LLM 直接生成/知识检索（LLM Direct/Knowledge）基于随机抽取的 50 个查询子集进行评估，而智能体（Agentic）方法则基于另一个包含 20 个查询的子集进行评估。由于查询集不同，各方法在 GED 指标上的直接比较在统计学上存在混淆因素。
W2：作为一项应用型数据挖掘研究，实际部署时需要权衡成本与性能。然而，论文未报告每种 LLM 的单次查询推理时间、API 成本（这对商业模型尤为重要）以及完整智能体工作流（pipeline）的总端到端延迟。
W3：尽管论文指出 MODIFY 操作会降低 GED 性能，但并未分析导致修改失败的原因。
W4：QM9S 提供的是模拟光谱，而 ChEMBL 提供的则是实验光谱与计算光谱的混合数据。虽然这种跨域设置很有价值，但论文未探讨模拟光谱与真实实验光谱之间的系统性差异，也未讨论这些差异是否会对检索结果产生偏差。

*12: Is this submission among the best 10% of submissions that you reviewed for ICDM'26?
  [_] No
  [X] Yes
  
*13: Are the datasets used in the study correctly identified and referenced?
  [X] 3 Yes
  [_] 2 Partial
  [_] 1 No
  [_] 0 Not applicable
  
*14: If the authors use private data in the experiments, do they state that they will make the data publicly available in the camera-ready version?
  [X] 3 Yes
  [_] 2 Partial
  [_] 1 No
  [_] 0 Not applicable
  
*15: Are the competing methods used in the study correctly identified and referenced?
  [X] 3 Yes
  [_] 2 Partial
  [_] 1 No
  [_] 0 Not applicable
  
*16: Do the authors provide a link to their source code, or state that they will make the source code publicly available in the camera-ready version?
  [X] 3 Yes
  [_] 2 Partial
  [_] 1 No
  [_] 0 Not applicable
  
*17: Is the experimental design detailed enough to allow for reproducibility? (You can also include comments on reproducibility in the body of your review.)
  [X] 3 Yes
  [_] 2 Partial
  [_] 1 No
  [_] 0 Not applicable
  
*18: If the paper is accepted, which format would you suggest?
  [X] Regular Presentation
  [_] Short Presentation
  
*19: Detailed comments for the authors
  see above.

========================================================
The review report from reviewer #3:

*1: Is the paper relevant to Appliied Track?
  [_] No
  [X] Yes
  
*2: How innovative is the paper?
  [X] 6 (Very innovative)
  [_] 3 (Innovative)
  [_] -2 (Marginally)
  [_] -4 (Not very much)
  [_] -6 (Not at all)
  
*3: How would you rate the technical quality of the paper?
  [X] 6 (Very high)
  [_] 3 (High)
  [_] -2 (Marginal)
  [_] -4 (Low)
  [_] -6 (Very low)
  
*4: How is the presentation?
  [X] 6 (Excellent)
  [_] 3 (Good)
  [_] -2 (Marginal)
  [_] -4 (Below average)
  [_] -6 (Poor)
  
*5: Is the paper of interest to the ICDM community?
  [X] 3 (Yes)
  [_] 2 (May be)
  [_] 1 (No)
  [_] 0 (Not applicable)
  
*6: What is your confidence in your review of this paper?
  [X] 2 (High) -- I have substantial expertise in this area, e.g., I have published on closely related topics or regularly read papers in this area.
  [_] 1 (Medium) -- I am familiar with this area and related work, but I am not an expert on all aspects of the paper.
  [_] 0 (Low) -- I am only vaguely familiar with this area or important parts of the paper are outside my expertise.
  
*7: Overall recommendation
  [X] 6: must accept (in top 25% of ICDM accepted papers)
  [_] 3: should accept (in top 80% of ICDM accepted papers)
  [_] -2: marginal (in bottom 20% of ICDM accepted papers)
  [_] -4: should reject (below acceptance bar)
  [_] -6: must reject (unacceptable: too weak, incomplete, or wrong)
  
*8: Summary of the paper's main contribution and impact
  This paper presents SpectraExpert, a Raman-based molecular identification framework that combines spectral retrieval, expert-rule matching, and LLM-guided candidate reasoning. Its main contribution is the adaptation of retrieval-augmented, rule-grounded LLM reasoning to a challenging scientific inverse problem. By narrowing the molecular search space and grounding decisions in spectroscopic evidence, the framework offers a practical and interpretable alternative to direct structure generation.
  本文提出了 SpectraExpert，这是一个基于拉曼光谱的分子识别框架，结合了光谱检索、专家规则匹配以及大语言模型（LLM）引导的候选分子推理。其主要贡献在于将“检索增强与规则支撑的 LLM 推理”这一范式应用于极具挑战性的科学反演问题。通过缩小分子搜索空间并基于光谱证据进行决策，该框架为直接生成分子结构提供了一种实用且具有可解释性的替代方案。

*9: Justification of your recommendation
  The paper combines spectral retrieval, expert rules, and LLM-based candidate reasoning. The proposed idea is novel and the experiment evaluation is comprehensive.
该论文结合了光谱检索、专家规则以及基于大语言模型（LLM）的候选推理。所提出的方法新颖，且实验评估全面。

*10: Three strong points of this paper (please number each point)
  1. **Novel applied idea.**
 The paper combines spectral retrieval, expert rules, and LLM-based candidate reasoning for Raman molecular identification.

 2. **Interpretable workflow.**
 The LLM decisions are grounded in explicit peak-level rules, making the reasoning more transparent than direct SMILES generation.

 3. **Challenging evaluation setting.**
 The cross-domain evaluation from QM9S to ChEMBL tests the method under realistic distribution shift.

 1. **新颖的应用构思。**
该论文结合了光谱检索、专家规则以及基于大语言模型（LLM）的候选推理，用于拉曼分子识别。

2. **可解释的工作流程。**
LLM 的决策基于明确的峰级规则，使得推理过程比直接生成 SMILES 序列更具透明度。

3. **具有挑战性的评估设置。**
从 QM9S 到 ChEMBL 的跨域评估，在真实的分布偏移场景下检验了该方法的性能。

*11: Three weak points of this paper (please number each point)
  1. **The comparisons are not fully controlled.**
 Different methods are evaluated on different query subsets, making direct comparison difficult.

 2. **The value of the LLM is not clearly isolated.**
 The paper lacks simple non-LLM reranking baselines and controlled ablations.

 3. **The performance gains are limited.**
 The learned retriever is close to KNN, the modification step often hurts performance, and no exact matches are achieved.

 1. **对比缺乏充分的控制条件。**
不同方法是在不同的查询子集上进行评估的，这使得直接比较变得困难。

2. **未能明确剥离出 LLM 的实际价值。**
论文缺乏简单的非 LLM 重排序基线（baseline）以及受控的消融实验。

3. **性能提升有限。**
学习型检索器的表现与 KNN 相当，修改步骤往往会损害性能，且未能实现精确匹配。

*12: Is this submission among the best 10% of submissions that you reviewed for ICDM'26?
  [X] No
  [_] Yes
  
*13: Are the datasets used in the study correctly identified and referenced?
  [X] 3 Yes
  [_] 2 Partial
  [_] 1 No
  [_] 0 Not applicable
  
*14: If the authors use private data in the experiments, do they state that they will make the data publicly available in the camera-ready version?
  [X] 3 Yes
  [_] 2 Partial
  [_] 1 No
  [_] 0 Not applicable
  
*15: Are the competing methods used in the study correctly identified and referenced?
  [X] 3 Yes
  [_] 2 Partial
  [_] 1 No
  [_] 0 Not applicable
  
*16: Do the authors provide a link to their source code, or state that they will make the source code publicly available in the camera-ready version?
  [X] 3 Yes
  [_] 2 Partial
  [_] 1 No
  [_] 0 Not applicable
  
*17: Is the experimental design detailed enough to allow for reproducibility? (You can also include comments on reproducibility in the body of your review.)
  [X] 3 Yes
  [_] 2 Partial
  [_] 1 No
  [_] 0 Not applicable
  
*18: If the paper is accepted, which format would you suggest?
  [X] Regular Presentation
  [_] Short Presentation
  
*19: Detailed comments for the authors
   1. **Report the effect of LLM candidate selection more directly.**
 It would be useful to show how often the LLM improves, preserves, or worsens the retrieval Top-1 result, and how often it selects the best candidate within the retrieved Top-(K).

 2. **Include simple non-LLM reranking baselines.**
 Comparisons with rule-coverage scoring or a lightweight learned reranker would help determine whether the benefit comes specifically from LLM reasoning or simply from adding rule-derived features.

 3. **Provide a controlled ablation study.**
 A comparison among retrieval only, retrieval + rules, retrieval + LLM, and retrieval + rules + LLM would better isolate the effect of each component.

1. **更直接地展示 LLM 候选结果筛选的效果。**
建议展示 LLM 在多大程度上改善、保持或恶化了检索结果中的 Top-1 结果，以及它在检索出的 Top-K 候选结果中选中最佳选项的频率。

2. **纳入简单的非 LLM 重排序基线（baseline）。**
通过与基于规则覆盖率的评分方法或轻量级学习型重排序器进行对比，有助于判断性能提升究竟源于 LLM 的推理能力，还是仅仅因为引入了基于规则的特征。

3. **提供受控的消融实验。**
通过对比“仅检索”、“检索+规则”、“检索+LLM”以及“检索+规则+LLM”这几种配置，可以更清晰地分离并评估各组件的作用。

========================================================
The review report from reviewer #4:

*1: Is the paper relevant to Appliied Track?
  [_] No
  [X] Yes
  
*2: How innovative is the paper?
  [X] 6 (Very innovative)
  [_] 3 (Innovative)
  [_] -2 (Marginally)
  [_] -4 (Not very much)
  [_] -6 (Not at all)
  
*3: How would you rate the technical quality of the paper?
  [X] 6 (Very high)
  [_] 3 (High)
  [_] -2 (Marginal)
  [_] -4 (Low)
  [_] -6 (Very low)
  
*4: How is the presentation?
  [X] 6 (Excellent)
  [_] 3 (Good)
  [_] -2 (Marginal)
  [_] -4 (Below average)
  [_] -6 (Poor)
  
*5: Is the paper of interest to the ICDM community?
  [X] 3 (Yes)
  [_] 2 (May be)
  [_] 1 (No)
  [_] 0 (Not applicable)
  
*6: What is your confidence in your review of this paper?
  [_] 2 (High) -- I have substantial expertise in this area, e.g., I have published on closely related topics or regularly read papers in this area.
  [X] 1 (Medium) -- I am familiar with this area and related work, but I am not an expert on all aspects of the paper.
  [_] 0 (Low) -- I am only vaguely familiar with this area or important parts of the paper are outside my expertise.
  
*7: Overall recommendation
  [X] 6: must accept (in top 25% of ICDM accepted papers)
  [_] 3: should accept (in top 80% of ICDM accepted papers)
  [_] -2: marginal (in bottom 20% of ICDM accepted papers)
  [_] -4: should reject (below acceptance bar)
  [_] -6: must reject (unacceptable: too weak, incomplete, or wrong)
  
*8: Summary of the paper's main contribution and impact
  This paper presents SpectraExpert, an applied data mining system for identifying molecular structures from Raman spectra. The framework integrates contrastive retrieval of candidate molecules from a spectral database, matching of detected peaks against 181 Nakamoto group-frequency rules, and LLM-guided reasoning for candidate evaluation and structural modification. The work constructs a reproducible cross-domain benchmark using QM9S for training and ChEMBL for testing, and evaluates five frontier LLMs across retrieval, direct prompting, knowledge-augmented prompting, and agentic settings. Results show that retrieval provides the most stable baseline, while rule-grounded LLM reasoning can produce strong hypotheses.

本文介绍了 SpectraExpert，这是一个用于根据拉曼光谱识别分子结构的应用型数据挖掘系统。该框架整合了多种机制：从光谱数据库中对候选分子进行对比检索；将检测到的峰值与 181 条 Nakamoto 基团频率规则进行匹配；以及利用大语言模型（LLM）引导的推理来进行候选评估与结构修正。本研究利用 QM9S 数据集进行训练、ChEMBL 数据集进行测试，构建了一个可复现的跨领域基准，并评估了五种前沿大语言模型在检索、直接提示、知识增强提示及智能体（agentic）模式下的表现。结果表明，检索方法提供了最稳定的基准，而基于规则的大语言模型推理则能生成高质量的结构假设。

*9: Justification of your recommendation
  I recommend acceptance because the SpectraExpert framework provides a well-structured end-to-end solution that mirrors the workflow of human spectroscopists. The cross-domain benchmark design is rigorous and reflects real-world deployment challenges. The experiments are comprehensive, covering five LLMs across multiple settings, and the analysis of failure modes is honest and insightful. The findings on structured-output reliability as a deployment bottleneck offer practical value beyond the specific application.

我建议予以录用，因为 SpectraExpert 框架提供了一种结构严谨的端到端解决方案，能够模拟人类光谱学家的工作流程。其跨领域基准测试设计严谨，反映了实际部署中面临的挑战。实验涵盖了多种设置下的五种大语言模型（LLM），内容详尽；对失效模式的分析既坦诚又深刻。关于“结构化输出可靠性是部署瓶颈”这一发现，除了在特定应用场景外，也具有实际应用价值。

*10: Three strong points of this paper (please number each point)
  S1. The paper presents a complete end-to-end applied workflow that integrates retrieval, rule-based analysis, and LLM reasoning for a real-world scientific inverse problem. The three-stage design closely mimics the decision-making process of human experts and provides interpretable, auditable outputs.

S2. The cross-domain benchmark setup is well-designed. Training on small molecules and testing on larger, application-oriented molecules provides a rigorous assessment of generalization under realistic domain shift, preventing the system from succeeding through memorized database overlap.

S3. The paper evaluates five frontier LLMs across multiple settings, reports multiple metrics , and provides detailed analysis of failure modes. The finding that structured-output reliability is the primary deployment bottleneck, offers actionable insights for future work.

S1. 本文针对现实世界中的科学反问题，提出了一套完整的端到端应用工作流，集成了检索、基于规则的分析以及大语言模型（LLM）推理。这种三阶段设计紧密模拟了人类专家的决策过程，并能提供可解释且可审计的输出结果。

S2. 跨领域基准测试的设置设计精良。通过在小分子上进行训练，并在更大、面向实际应用的分子上进行测试，该方案在真实的领域偏移（domain shift）条件下对模型的泛化能力进行了严格评估，从而避免了模型仅凭记忆数据库中的重叠数据而“成功”的情况。

S3. 本文在多种设置下评估了五种前沿大语言模型，报告了多项指标，并对失败模式进行了详细分析。研究发现，结构化输出的可靠性是部署面临的主要瓶颈，这一发现为未来的研究工作提供了具有实践指导意义的见解。

*11: Three weak points of this paper (please number each point)
  W1. Different experimental settings use different query batches, which makes direct comparisons across rows in Table I problematic.

W2. The modification fallback consistently degrades performance compared to the SELECT path across all models. The authors should analyze why it fails or propose concrete improvements. 

W3. The paper should clarify how much of the retrieval performance comes from the contrastive learning design versus the basic principle that similar spectra imply similar structures.

W1. 不同的实验设置采用了不同的查询批次（query batches），这使得表 I 中各行之间的数据无法直接进行比较。

W2. 在所有模型中，与 SELECT 路径相比，“修改回退”（modification fallback）机制均导致了性能下降。作者应分析其失效原因，或提出具体的改进方案。

W3. 论文应明确阐述：检索性能的提升在多大程度上归功于对比学习设计，又在多大程度上源于“相似光谱意味着相似结构”这一基本原理。

*12: Is this submission among the best 10% of submissions that you reviewed for ICDM'26?
  [_] No
  [X] Yes
  
*13: Are the datasets used in the study correctly identified and referenced?
  [X] 3 Yes
  [_] 2 Partial
  [_] 1 No
  [_] 0 Not applicable
  
*14: If the authors use private data in the experiments, do they state that they will make the data publicly available in the camera-ready version?
  [_] 3 Yes
  [_] 2 Partial
  [_] 1 No
  [X] 0 Not applicable
  
*15: Are the competing methods used in the study correctly identified and referenced?
  [X] 3 Yes
  [_] 2 Partial
  [_] 1 No
  [_] 0 Not applicable
  
*16: Do the authors provide a link to their source code, or state that they will make the source code publicly available in the camera-ready version?
  [X] 3 Yes
  [_] 2 Partial
  [_] 1 No
  [_] 0 Not applicable
  
*17: Is the experimental design detailed enough to allow for reproducibility? (You can also include comments on reproducibility in the body of your review.)
  [X] 3 Yes
  [_] 2 Partial
  [_] 1 No
  [_] 0 Not applicable
  
*18: If the paper is accepted, which format would you suggest?
  [X] Regular Presentation
  [_] Short Presentation
  
*19: Detailed comments for the authors
  D1. The reason that modification fallback shows consistent degradation across all models should be analyzed.

D2. How much of the retrieval performance comes from the contrastive learning design versus the basic principle that similar spectra imply similar structures should be clarified.

D3. I appreciate the honest discussion of limitations and failure modes. The parse rate analysis is particularly valuable and highlights a practical concern that is often overlooked in LLM research. I encourage the authors to expand this discussion slightly to include recommendations for practitioners.

D1. 应当分析为何“修改回退”（modification fallback）机制在所有模型中均表现出一致的性能下降。

D2. 应当明确检索性能的提升在多大程度上归功于对比学习设计，以及在多大程度上源于“相似光谱意味着相似结构”这一基本原理。

D3. 我很欣赏文中对局限性和失效模式的坦诚探讨。关于解析率（parse rate）的分析尤为有价值，它揭示了一个在大语言模型（LLM）研究中常被忽视的实际问题。建议作者适当扩展这部分讨论，并针对从业者提出相关建议。

========================================================