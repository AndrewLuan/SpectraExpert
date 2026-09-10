# 最新更新：五模型 Agent 新结果入稿

当前 PDF 为 **9 页、5 张编号表**。总表 III 已加入修复后的 GPT-5、o3、Claude Sonnet 4.6、Gemini 2.5 Flash，并保留已完成的 DeepSeek full-Agent 和竞品结果。摘要、方法输出契约、Results、复现说明及结论同步更新；完整数值与解释边界见 [AGENT_MODELS_AUDIT.md](AGENT_MODELS_AUDIT.md)。以下为此前版本的历史审计。

# 当前更新：竞争方法进入总表（2026-09-09）

camera-ready 仍为 **9 页（含参考文献）**，共 5 张编号表。总表 III 同时展示本方法和 Vib2Mol 检索/生成、VibraCLIP 的 Valid、top-1 Exact、Tanimoto 与 chemical GED；MolSpectLLM 标为 N/A。三个已完成竞争分支在 JJ 上从各 200 条逐查询评分重新汇总，全部与完成版汇总一致。原独立竞争表已移除，top-10/top-25 保留在一处正文中。

最新检查：`results/jj-20260909/competitor_main_table_verification.json`；远端重新统计及来源哈希：`results/jj-20260909/competitor_main_table_refresh.json`。以下内容保留此前版本记录。

# 当前版本：10 页上限下的内容收敛（2026-09-09）

最终 camera-ready 为 **9 页（含参考文献）**，由 main.tex、domain_method.tex、latest_results.tex 和 discussion_repro.tex 编译。6 张表、1 张当前流程图；历史 ChEMBL 与原生复现长表不再进入 PDF，但源文件和全部结果快照保留。当前证据核对见 [REVIEW_RESPONSE_AUDIT.md](REVIEW_RESPONSE_AUDIT.md)，最终检查见 `results/jj-20260909/ten_page_verification.json`。以下段落保留此前修订的历史记录，不代表当前仍包含附录或与原 PDF 内容等价。

# 2026-09-09 实质修订说明

用户已授权整理 JJ 最新结果并更新 camera-ready。以下 2026-09-04 记录仅描述当时的格式重建，不再表示当前稿件与旧 PDF 内容等价。当前数值来源、版本选择与文件哈希见 [RESULTS_AUDIT.md](RESULTS_AUDIT.md)。新的实验正文、摘要、结论与附录以本次 JJ 快照为准，原 ChEMBL 内容保留为历史附录。

# Paper Source Provenance

The manuscript uses two independent sources of truth.

- Content source: `/Users/andrewluan/Downloads/DM2092_5291 (2).pdf`
- Content SHA-256: `e7bad853c4cf11fc866680a2cc1a9e06a7c40d04dd34be9bb48984823212cf1f`
- Format source: `/Users/andrewluan/Downloads/conference-latex-template.zip`
- Format SHA-256: `516252d9ac6e974af3a1d9aa72f97ecc462b9a47556225019fb0adef69bba78a`
- Official class: `IEEEtran.cls 2015/08/26 V1.8b`
- Imported on: `2026-09-04`

The PDF supplies all visible manuscript content. The official template supplies layout only. `../IEEE_Conference_Template/main/conference_101719.tex` is an older structural reference and must not override the final PDF.
