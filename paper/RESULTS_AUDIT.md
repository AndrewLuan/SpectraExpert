# 最新更新：五模型 Agent 新结果入稿

当前 PDF 为 **9 页、5 张编号表**。总表 III 已加入修复后的 GPT-5、o3、Claude Sonnet 4.6、Gemini 2.5 Flash，并保留已完成的 DeepSeek full-Agent 和竞品结果。摘要、方法输出契约、Results、复现说明及结论同步更新；完整数值与解释边界见 [AGENT_MODELS_AUDIT.md](AGENT_MODELS_AUDIT.md)。以下为此前版本的历史审计。

# 当前更新：竞争方法进入总表（2026-09-09）

camera-ready 仍为 **9 页（含参考文献）**，共 5 张编号表。总表 III 同时展示本方法和 Vib2Mol 检索/生成、VibraCLIP 的 Valid、top-1 Exact、Tanimoto 与 chemical GED；MolSpectLLM 标为 N/A。三个已完成竞争分支在 JJ 上从各 200 条逐查询评分重新汇总，全部与完成版汇总一致。原独立竞争表已移除，top-10/top-25 保留在一处正文中。

最新检查：`results/jj-20260909/competitor_main_table_verification.json`；远端重新统计及来源哈希：`results/jj-20260909/competitor_main_table_refresh.json`。以下内容保留此前版本记录。

# 当前版本：10 页上限下的内容收敛（2026-09-09）

最终 camera-ready 为 **9 页（含参考文献）**，由 main.tex、domain_method.tex、latest_results.tex 和 discussion_repro.tex 编译。6 张表、1 张当前流程图；历史 ChEMBL 与原生复现长表不再进入 PDF，但源文件和全部结果快照保留。当前证据核对见 [REVIEW_RESPONSE_AUDIT.md](REVIEW_RESPONSE_AUDIT.md)，最终检查见 `results/jj-20260909/ten_page_verification.json`。以下段落保留此前修订的历史记录，不代表当前仍包含附录或与原 PDF 内容等价。

# JJ 结果与 camera-ready 更新审计

本轮按用户 2026-09-09 的要求实质更新论文内容，不再受 9 月 4 日“只改排版”的历史内容锁定约束。保留修订前 main.tex 和用户已有的作者、相关工作等修改；未改动训练代码，未启动训练或 API 请求，未向服务器写入、删除或覆盖结果。

## 本轮采用口径

正文以 corrected exact-v2 Test-200 为主，历史 QM9S–ChEMBL 移到独立附录。官方数据复现也使用单独附录。采集时间为 2026-09-09（Asia/Shanghai）；最后纳入的已完成实验是 17:39 的非 LLM GED 汇总。以下时间统一记录 UTC，换算北京时间加 8 小时。

“最新”按同协议完成状态、数据/评分修复与实验依赖选择，不按较大指标挑选。具体选择如下：

- 完整结构库采用 exact_v2，旧库 21 个完整结构缺口的原分数不进入新主表；Direct LLM 采用修复后的 192 条复用 + 8 条补跑结果和 strict 口径。
- 多种子采用 run02/summary_v2.json，替代早期 summary.json；报告五种子均值与样本标准差，不用最高测试分数替代均值。
- 候选 LLM 采用 ascend_run02/format_normalized 的最新格式兼容评分，保留实际模型路由说明；未把后来的多种子 checkpoint 冒充为已调用 LLM 的模型。
- 知识库同时纳入 matched legacy、15-mode 六组以及最新 static full KB。Test-200 的 44/200 仍属旧 KB，未归因为新 15-mode 库。
- 非 LLM 采用 with_ged 的完成结果，同时给出共同可算 GED 子集。旧 GED 第一上界与 chemical GED 完成搜索分开；超时不补零。
- 竞争方法采用 unified/scored 的 all5 最终汇总，包括两项敏感性分析，替代早期三臂/单臂汇总。无兼容完成记录的 MolSpectLLM 不填零分。
- 最新曲线拟合报告采用 fit_v2，但编码器实际依赖 fit_v1；正文明确区分。原 overlap pilot 与 A/B/C 域适配只作独立历史诊断。
- 官方复现使用 corrected generation_fix / fix2、最终 retrieval 日志及 9 月 6 日补齐的 VibraCLIP 五组，数值直接从日志提取；不使用报告中的未经核验论文对照数值。
- failed launcher、smoke、TOY、准备数据和重复旧评分均不视为独立性能实验。其相关旧结果保存在快照用于追溯。

## 论文中的组织

- `latest_results.tex`：最新闭库、多种子、LLM、知识库、非 LLM、竞争方法、曲线与历史诊断。
- `native_results.tex`：15 个官方检索配置、7 个生成配置及 5 个 VibraCLIP 配置。
- `historical_results.tex`：修订前 ChEMBL 实验正文、全部原表格和统计图；明确历史协议。
- `results/jj-20260909/main.before.tex`：修订前工作区稿件，不覆盖用户已有更改。
- `snapshot.json`：汇总/逐查询评分及相关协议的内容快照，包含远端路径、原文件 SHA-256、mtime；不包含模型、原始谱库或 API 请求凭据。
- `additional_sources.json`：补充诊断、官方复现聚合日志与历史报告，带原路径和哈希。

本轮没有遍历或传输禁止读取的大型原始数据目录；读取范围是已知项目 outputs、repro 评分目录和小型日志。指标从这些实验产物复核，不声称重新执行全部实验。当前验证与 Test-200 已被多次观察，正文不称盲测。五种子 test hard-baseline 置信区间跨零，知识库净收益有限，已如实写入。

## 文件清单

| 远端文件 | 更新时间 UTC | SHA-256 |
|---|---|---|
| `/home/user/IRfingerprints/SpectrExpert/outputs/raman_closed_10k_test200_seed0_exact_v2/repair_report.json` | 2026-09-08T14:51:44.888712+00:00 | `524bab3cda4d447975eb1fe65a7e76b260d41b73204ff0d058e7e051c660b8c1` |
| `/home/user/IRfingerprints/SpectrExpert/outputs/raman_closed_knn_model_eval_run01/protocol.json` | 2026-09-08T13:51:12.425980+00:00 | `ba9eae27a156a46c360c4a8ca1e09a736c5e3410b02090199398333ae438c9c5` |
| `/home/user/IRfingerprints/SpectrExpert/outputs/raman_closed_knn_model_eval_run01/summary.json` | 2026-09-08T13:51:38.005583+00:00 | `cba43445bb53460de0402d2adba496ee5e43e1f89683d51c766fb627441bdeed` |
| `/home/user/IRfingerprints/SpectrExpert/outputs/raman_closed_structural_metrics_run01/knn_cosine/results.json` | 2026-09-08T14:07:31.753226+00:00 | `f220fd5726dd89a706e89e7be292496ba124173063a997163bf344f6b490d2c8` |
| `/home/user/IRfingerprints/SpectrExpert/outputs/raman_closed_structural_metrics_run01/protocol.json` | 2026-09-08T14:00:47.440325+00:00 | `02cc85874133d28e20c5b5bc8a1b378691de2f035a0d7c6005d65f5bf0708ce9` |
| `/home/user/IRfingerprints/SpectrExpert/outputs/raman_closed_structural_metrics_run01/spectrum_graph_v2/results.json` | 2026-09-08T14:14:20.327199+00:00 | `41fc22f984a6624712f23aa6fb917cad6380a9a1a7d8491897bc996fa4592c2c` |
| `/home/user/IRfingerprints/SpectrExpert/outputs/raman_closed_structural_metrics_run01/summary.json` | 2026-09-08T14:14:20.330397+00:00 | `fd47dd64793366ebe6846f5bc530fdd130a6fa3bfe79e4bc0b01ef6c848c5e8a` |
| `/home/user/IRfingerprints/SpectrExpert/outputs/raman_curve_proxy_fit_20260909_v1/metrics.json` | 2026-09-08T17:19:41.232289+00:00 | `59e7fbc08a81bd4032330d4a256d2b7ef401260941c30a766364e241bb36cb48` |
| `/home/user/IRfingerprints/SpectrExpert/outputs/raman_curve_proxy_fit_20260909_v2/metrics.json` | 2026-09-08T17:24:35.810636+00:00 | `07318cff9dac66485b8fd1a55974659a62ffd6923c18d654a6bbd53caf4a99ab` |
| `/home/user/IRfingerprints/SpectrExpert/outputs/raman_curve_proxy_preview_20260909_v1/metrics.json` | 2026-09-08T17:18:16.006602+00:00 | `0b4ba1d3df392e63a0631be5b8a5a1eb5626095a621ce347990c91e1e0b7df6f` |
| `/home/user/IRfingerprints/SpectrExpert/outputs/raman_domain_exact_v2_run02/A/summary.json` | 2026-09-08T15:51:06.793688+00:00 | `7b0274b504ee7f32f82e7719116733b8f93f46507febbaac4943cc5ac62ef2c6` |
| `/home/user/IRfingerprints/SpectrExpert/outputs/raman_domain_exact_v2_run02/B/summary.json` | 2026-09-08T15:53:34.333718+00:00 | `f0ddc3c2cc23f606f5eb0a57774b44ebc95469780010ce11df0530712ae86a8d` |
| `/home/user/IRfingerprints/SpectrExpert/outputs/raman_domain_exact_v2_run02/C/summary.json` | 2026-09-08T15:53:23.432549+00:00 | `1c4fc60dd8c46b547bd557a807400f0b7e8183d598995676a2a1f3c2fcaf4883` |
| `/home/user/IRfingerprints/SpectrExpert/outputs/raman_domain_exact_v2_run02/final/model/results.json` | 2026-09-08T15:59:29.421998+00:00 | `a563782c57534081c2f98ffda7acea8f2d53cc7cee5df91a79eb71a4b5735de4` |
| `/home/user/IRfingerprints/SpectrExpert/outputs/raman_domain_exact_v2_run02/final/raw_knn/results.json` | 2026-09-08T16:05:35.709613+00:00 | `b725e26c9dabc6475903c8d2bfbd2ac0caafa3f12d8bdd8f8cee2a85ddd999ce` |
| `/home/user/IRfingerprints/SpectrExpert/outputs/raman_domain_exact_v2_run02/final/status.json` | 2026-09-08T16:05:35.844498+00:00 | `e85a4abb9644ceed6b8ac03a75bf0ddb720d25daca909706d4d27695d1eed8e7` |
| `/home/user/IRfingerprints/SpectrExpert/outputs/raman_domain_exact_v2_run02/smoke/C/summary.json` | 2026-09-08T15:47:18.835725+00:00 | `2135d48219302ea0de6289cbff69ff4a511265d7f928379f651eb01ee004044c` |
| `/home/user/IRfingerprints/SpectrExpert/outputs/raman_domain_exact_v2_run02/topk_exact/summary.json` | 2026-09-08T16:20:58.702065+00:00 | `eb912cdeaf8f3d963a203cb98f22e77175cb106b149c886c44f9fa6b21aa9574` |
| `/home/user/IRfingerprints/SpectrExpert/outputs/raman_encoder_kb_llm_run01/model/results.json` | 2026-09-09T02:31:04.155727+00:00 | `4ffaf6fbc6290603ad971c88de6c0a94511c2d884b02f3a575a7858434d4786c` |
| `/home/user/IRfingerprints/SpectrExpert/outputs/raman_encoder_kb_llm_run01/selection_summary.json` | 2026-09-09T02:24:32.619186+00:00 | `899be37b8e8442a7419efc8be56856f592cce70417cd578f6bc094daecb753b0` |
| `/home/user/IRfingerprints/SpectrExpert/outputs/raman_kb_ablation_val_run01/summary.json` | 2026-09-09T03:55:54.769861+00:00 | `bfc91e4b9b92b74755650fa348b921a557225b390c34eb5cee8e35ec970b2123` |
| `/home/user/IRfingerprints/SpectrExpert/outputs/raman_llm_direct_deepseek_v4_flash_test200_exact_v2_run01/replacement_metrics/results.json` | 2026-09-08T15:04:49.299545+00:00 | `3d3ff63eb8e2e56e73bffb2b6401b9a263b0375335b259851d99d276682807c8` |
| `/home/user/IRfingerprints/SpectrExpert/outputs/raman_llm_direct_deepseek_v4_flash_test200_exact_v2_run01/results.json` | 2026-09-08T15:04:49.350416+00:00 | `8f8960fa9b7f87fd8561b3ba3bd489c55c2c691691ae0d0794a2935b97ffcafe` |
| `/home/user/IRfingerprints/SpectrExpert/outputs/raman_llm_direct_deepseek_v4_flash_test200_exact_v2_run01/status.json` | 2026-09-08T15:04:49.350580+00:00 | `dd1e1514de44dd6118e41a567436399041af5185ef2897f94f69bfcdc0c60ee9` |
| `/home/user/IRfingerprints/SpectrExpert/outputs/raman_llm_direct_deepseek_v4_flash_test200_run01/results.json` | 2026-09-08T13:25:34.092344+00:00 | `dafba8b29140fbe74c9160f7ad68ec5fb2a4835120ad8e89445e395eee6a65be` |
| `/home/user/IRfingerprints/SpectrExpert/outputs/raman_nonllm_rerank_run01/common_ged_tables.json` | 2026-09-09T09:39:24.596577+00:00 | `bc996ae4863ffe72c7fba779a4fe976b9fdf5c1107e56e7ab59e62425df2dd63` |
| `/home/user/IRfingerprints/SpectrExpert/outputs/raman_nonllm_rerank_run01/completion.json` | 2026-09-09T09:39:24.597056+00:00 | `169362fb0cab455aa86cde470e20d475a0b38682b43b4438d2a1687b4b042e20` |
| `/home/user/IRfingerprints/SpectrExpert/outputs/raman_nonllm_rerank_run01/frozen.json` | 2026-09-09T09:22:47.964269+00:00 | `1a90036b0a7f71a940188c3b6693eaf8882fa7c0e78b5610a0cc6fba3b6d7ae2` |
| `/home/user/IRfingerprints/SpectrExpert/outputs/raman_nonllm_rerank_run01/protocol.json` | 2026-09-09T09:22:41.226675+00:00 | `3e5f1593ec9d08e2eed5fe00cae9b442e62ff90eb1144a59696db3eef9632006` |
| `/home/user/IRfingerprints/SpectrExpert/outputs/raman_nonllm_rerank_run01/test/summary.json` | 2026-09-09T09:23:11.655048+00:00 | `663c5c542aaff8f7654fb780de4e6bd59cc48e25eb128392cfd346f62deca347` |
| `/home/user/IRfingerprints/SpectrExpert/outputs/raman_nonllm_rerank_run01/test/with_ged/protocol.json` | 2026-09-09T09:38:09.779214+00:00 | `4f6184dc361b334e9e07f597372d6b558bc1946df62a226ad5340d068aee7d21` |
| `/home/user/IRfingerprints/SpectrExpert/outputs/raman_nonllm_rerank_run01/test/with_ged/summary.json` | 2026-09-09T09:38:09.562651+00:00 | `f1c81f10f891a6c3b9641073ec868160f4265eb9f0b8f7f2696197d9097cb024` |
| `/home/user/IRfingerprints/SpectrExpert/outputs/raman_nonllm_rerank_run01/validation/summary.json` | 2026-09-09T09:22:48.357678+00:00 | `a6286945feed1e645f30f004e45996ad8c87b1a2c6d0479bd0503dd331b1e474` |
| `/home/user/IRfingerprints/SpectrExpert/outputs/raman_nonllm_rerank_run01/validation/with_ged/protocol.json` | 2026-09-09T09:37:42.191474+00:00 | `b136f9ae2a3744b025e92c3857cfd0420105561ea03f80602bd810527b070849` |
| `/home/user/IRfingerprints/SpectrExpert/outputs/raman_nonllm_rerank_run01/validation/with_ged/summary.json` | 2026-09-09T09:37:41.903226+00:00 | `d342ec39a6e1618493a2d723b7ec6b8576035b066bc15016264c38891155f985` |
| `/home/user/IRfingerprints/SpectrExpert/outputs/raman_overlap_knn_20260908_01/protocol.json` | 2026-09-08T11:30:12.622000+00:00 | `333b9f123dc287cf5731400e094314dc815313d957b249b9d17345be2dbc3135` |
| `/home/user/IRfingerprints/SpectrExpert/outputs/raman_overlap_knn_20260908_01/summary.json` | 2026-09-08T11:32:06.166072+00:00 | `2690a8e59e0b63f265344c61d64601cc415b23c106fc3656b2e08c568d85dea9` |
| `/home/user/IRfingerprints/SpectrExpert/outputs/raman_static_kb_val_run01/summary.json` | 2026-09-09T08:43:22.362552+00:00 | `749127fd10a4d57a4db7f0bb889ad0020a0f762c1366ad138ccb8ed0687f0858` |
| `/home/user/IRfingerprints/SpectrExpert/outputs/raman_top10_llm_selection_ascend_run02/analysis.json` | 2026-09-08T19:05:55.019110+00:00 | `95c26bdbafb883d0092e1376e56f6e067391b2d63db52a7cbd7b3c65500e55f0` |
| `/home/user/IRfingerprints/SpectrExpert/outputs/raman_top10_llm_selection_ascend_run02/format_normalized/analysis.json` | 2026-09-08T19:06:17.172415+00:00 | `3d2c1c20bf272c49c0a4e2517290afa016c8a438e96f073b2c61b4f335d00c87` |
| `/home/user/IRfingerprints/SpectrExpert/outputs/raman_top10_llm_selection_ascend_run02/format_normalized/model/results.json` | 2026-09-08T19:05:55.131116+00:00 | `962e0142d9cb4bada6ebdfa38d4eea5afe74f44d8d227892136429c389a1a5e0` |
| `/home/user/IRfingerprints/SpectrExpert/outputs/raman_top10_llm_selection_ascend_run02/format_normalized/raw_knn/correction_0060/results.json` | 2026-09-08T19:06:02.718623+00:00 | `85ba89a87fa668ebbdeead4c2fa01089760a5482c1c5b204ba2ec693050fa99b` |
| `/home/user/IRfingerprints/SpectrExpert/outputs/raman_top10_llm_selection_ascend_run02/format_normalized/raw_knn/results.json` | 2026-09-08T19:06:02.730029+00:00 | `dab46553b0d576d5a7662b596e5467d98e3b004e6f1821d6aaa2c08444c13741` |
| `/home/user/IRfingerprints/SpectrExpert/outputs/raman_top10_llm_selection_ascend_run02/model/results.json` | 2026-09-08T19:04:46.320679+00:00 | `962e0142d9cb4bada6ebdfa38d4eea5afe74f44d8d227892136429c389a1a5e0` |
| `/home/user/IRfingerprints/SpectrExpert/outputs/raman_top10_llm_selection_ascend_run02/raw_knn/results.json` | 2026-09-08T19:05:18.622427+00:00 | `966b9f1d28f014de0b0287531d42b220317787ed67ffc0d61420be70b2576057` |
| `/home/user/IRfingerprints/SpectrExpert/outputs/raman_top10_llm_selection_ascend_run02/selection_summary.json` | 2026-09-08T18:59:04.401441+00:00 | `9f5a7c0c81df5c9b1130252459774989f705e3e8f2740e21251e327db43be091` |
| `/home/user/IRfingerprints/SpectrExpert/outputs/raman_top10_multiseed_run02/seed_1/baseline/summary.json` | 2026-09-09T02:07:28.990170+00:00 | `ce690e9c24128a8cbb1cba1a6033e60162c5699816c4943ba44e780b2b7ced96` |
| `/home/user/IRfingerprints/SpectrExpert/outputs/raman_top10_multiseed_run02/seed_1/hard/summary.json` | 2026-09-09T02:16:25.577097+00:00 | `472bcf50537e25e47a7d02961b27ad74d7f74b9eedc379f11fe3cf2c3c03ae1e` |
| `/home/user/IRfingerprints/SpectrExpert/outputs/raman_top10_multiseed_run02/seed_2/baseline/summary.json` | 2026-09-09T02:07:33.976679+00:00 | `a90aef0e0fa5ddcea80d3f1e80dc5c8846be03a7fbb278e6d618bd5589ddc625` |
| `/home/user/IRfingerprints/SpectrExpert/outputs/raman_top10_multiseed_run02/seed_2/hard/summary.json` | 2026-09-09T02:16:43.203249+00:00 | `79012b5e63387f768fe1407831c467305d22cfdaa42f01423dea7ab8656e1cd8` |
| `/home/user/IRfingerprints/SpectrExpert/outputs/raman_top10_multiseed_run02/seed_3/baseline/summary.json` | 2026-09-09T02:04:42.021183+00:00 | `64764c7a042ba6edcc0a5aa7e0197f718064d07fea7f5d9361bd8ec3b50a6291` |
| `/home/user/IRfingerprints/SpectrExpert/outputs/raman_top10_multiseed_run02/seed_3/hard/summary.json` | 2026-09-09T02:13:39.049158+00:00 | `3dbb5089d8ca78edd7b0085a2d76c5293134ee1742d007f219fec15efa6d41b7` |
| `/home/user/IRfingerprints/SpectrExpert/outputs/raman_top10_multiseed_run02/seed_4/baseline/summary.json` | 2026-09-09T02:07:34.668277+00:00 | `ef7afc5e55751565fe71247b267c0045ac5e058eb1ebbc422aa0d5032b0bb3ac` |
| `/home/user/IRfingerprints/SpectrExpert/outputs/raman_top10_multiseed_run02/seed_4/hard/summary.json` | 2026-09-09T02:16:31.483538+00:00 | `ce6cb43491f57adf0fd43984811a8c284ec7072f8c27453317f3cc85d25dfe29` |
| `/home/user/IRfingerprints/SpectrExpert/outputs/raman_top10_multiseed_run02/summary.json` | 2026-09-09T02:16:55.339834+00:00 | `4760cbd2a80cf7413d37e4977f2b01cf4f4cf5ef2444e23c3603325d606159a8` |
| `/home/user/IRfingerprints/SpectrExpert/outputs/raman_top10_multiseed_run02/summary_v2.json` | 2026-09-09T02:19:50.419232+00:00 | `2e6a4738d4be82c5df9edbce82ac82e812241bfbc21fdbfdc1a5499061ec3fa6` |
| `/home/user/IRfingerprints/SpectrExpert/outputs/raman_top10_surrogate_run01/baseline/summary.json` | 2026-09-08T18:16:32.571797+00:00 | `fea94d6403379b4286334f21c9a736d3a0ee249557b6a96557946f4080938d0d` |
| `/home/user/IRfingerprints/SpectrExpert/outputs/raman_top10_surrogate_run01/comparison.json` | 2026-09-08T18:17:57.819544+00:00 | `ff163d4b57c41debd5ad1962914391470c512caaa94537fba2dafb5036088307` |
| `/home/user/IRfingerprints/SpectrExpert/outputs/raman_top10_surrogate_run01/domain_diagnostics.json` | 2026-09-08T18:06:23.180570+00:00 | `4791ebe2c061d6959578cdbd0bb532a8b0e66fc607f3f490d8eef61b52a054c6` |
| `/home/user/IRfingerprints/SpectrExpert/outputs/raman_top10_surrogate_run01/final/selection.json` | 2026-09-08T18:16:39.662374+00:00 | `3903975e5749fb415c7b76529071dfd926dad522eb97aa440ab497f14e0b55c4` |
| `/home/user/IRfingerprints/SpectrExpert/outputs/raman_top10_surrogate_run01/final/summary.json` | 2026-09-08T18:16:43.527018+00:00 | `46f41e749c4303f22664be8c65e3a13345e7f2c39042e8782a0335163cea5ed6` |
| `/home/user/IRfingerprints/SpectrExpert/outputs/raman_top10_surrogate_run01/fixed/summary.json` | 2026-09-08T18:16:16.273799+00:00 | `df46155f6f326c7a3e8bb3335a0d8a05c103c06d44c50de667ccc61078e26f0d` |
| `/home/user/IRfingerprints/SpectrExpert/outputs/raman_top10_surrogate_run01/hard/summary.json` | 2026-09-08T18:16:35.013253+00:00 | `6b58bc2a6a88b1693720d4dc6ed7cb933d9edd517cad0df1905f7499dca1eb50` |
| `/home/user/IRfingerprints/SpectrExpert/outputs/raman_top10_surrogate_run01/protocol.json` | 2026-09-08T18:06:28.900012+00:00 | `c444492f4ed33168ef9dd7684078f2c22e98a8ffdecd8e1b7c70cb6f544cacc6` |
| `/home/user/IRfingerprints/SpectrExpert/outputs/raman_top10_surrogate_run01/proxy_retrieval_diagnostic.json` | 2026-09-08T18:11:17.644061+00:00 | `d4bfec9ded267a9b9117531c66463207bf3af4afe4636fa320fb194aaca8c14c` |
| `/home/user/IRfingerprints/SpectrExpert/outputs/raman_top10_surrogate_run01/random/summary.json` | 2026-09-08T18:16:03.957689+00:00 | `3f1bb27e43455842c92b7de2bbbfb1d813be2a99128154096e028ee573d90e11` |
| `/home/user/repro/REPORT_vib2mol.md` | 2026-09-04T07:22:28.597799+00:00 | `ef371a46df6cc4b8e125f842209ef526268c645701c926b94564cc946a52b1d2` |
| `/home/user/repro/REPORT_vib2mol_generation.md` | 2026-09-05T13:38:16+00:00 | `f823f0d40ee487208970545c5cab0d52d87613ce556af688634cb6bc30b30cbc` |
| `/home/user/repro/logs/vib2mol/generation_fix2_mols_ir-raman_07-14-07-07-e1c1f4_base_0905_1502.log` | 2026-09-05T11:01:32.613832+00:00 | `2330a08afea0f4494113dfe7c38138ab97676c01fb3384ebde3f253c6e5e1c36` |
| `/home/user/repro/logs/vib2mol/generation_fix2_mols_ir-raman_07-14-07-07-e1c1f4_rerank_0905_1502.log` | 2026-09-05T13:19:37.011751+00:00 | `0d743b872fc367cbde6eb3c607ccf71bdb294f61656ec4b4647622a695b8aad9` |
| `/home/user/repro/logs/vib2mol/generation_fix2_mols_raman_07-14-07-06-8a587e_base_0905_1502.log` | 2026-09-05T11:02:29.803831+00:00 | `204f7b6247e8d02429d8035985e508631ea9726fc121f62e494077ada460cbd7` |
| `/home/user/repro/logs/vib2mol/generation_fix2_mols_raman_07-14-07-06-8a587e_rerank_0905_1502.log` | 2026-09-05T13:18:49.460752+00:00 | `6b9c7b74a3dd7b855c00bc9e049a1a7fda64871515fce6957971c0a58c2e3e13` |
| `/home/user/repro/logs/vib2mol/generation_fix2_qm9s_raman_07-07-07-51-af1552_base_0905_1502.log` | 2026-09-05T09:05:30.205899+00:00 | `d32453a0630c4cf3bca534c38861bd2870c468285c019c36b6041482ce247387` |
| `/home/user/repro/logs/vib2mol/generation_fix2_qm9s_raman_07-07-07-51-af1552_rerank_0905_1502.log` | 2026-09-05T11:17:44.421822+00:00 | `11abe8b597c89f007d06345d84f7d9b83fdbbd255227e95b6bf7e320275ebf5c` |
| `/home/user/repro/logs/vib2mol/generation_fix2_qm9s_raman_07-19-03-49-370cd1_base_0905_1502.log` | 2026-09-05T09:11:29.629896+00:00 | `2baf011b8845941fda714bb7c6e9ca0872ccd41b390155055be37d2d81d08d18` |
| `/home/user/repro/logs/vib2mol/generation_fix2_qm9s_raman_07-19-03-49-370cd1_rerank_0905_1502.log` | 2026-09-05T11:20:17.339821+00:00 | `c2e7015a73acdd2eb6f6e543329c9a7aac6133bb2a9f1d895afa95ea3af3a300` |
| `/home/user/repro/logs/vib2mol/generation_fix_mols_ir_07-14-07-06-e44b51_base_0905_1234.log` | 2026-09-05T06:44:15.568982+00:00 | `59253901f6e648d62931d64ceaa1b8531922a1e7b886c8115ea85cdcbba55d1c` |
| `/home/user/repro/logs/vib2mol/generation_fix_mols_ir_07-14-07-06-e44b51_rerank_0905_1234.log` | 2026-09-05T09:03:50.690900+00:00 | `83aa0e70d920cb833a003038f8e35bc5cb02dff184ea5fb409ef77b7ac76adb8` |
| `/home/user/repro/logs/vib2mol/generation_fix_qm9s_ir_07-12-02-02-4afc55_base_0905_1234.log` | 2026-09-05T04:50:43.721048+00:00 | `943777140edb2d15ce344b010a6c7a0b28fa10a4a2dcd5cbe658c3155ab6ba99` |
| `/home/user/repro/logs/vib2mol/generation_fix_qm9s_ir_07-12-02-02-4afc55_rerank_0905_1234.log` | 2026-09-05T07:01:57.187971+00:00 | `733492c308f83ea19e4c9fee619f11e6e9d8b69293926dec297c7cf5c6979178` |
| `/home/user/repro/logs/vib2mol/generation_fix_qm9s_raman-ir_07-19-03-47-7c9f30_base_0905_1234.log` | 2026-09-05T04:50:39.639048+00:00 | `39b84db17cada2f30782c19f2bb478129517e7cb3507adaa037ace05c4dcaffb` |
| `/home/user/repro/logs/vib2mol/retrieval_mols_ir_base_0904_1412.log` | 2026-09-04T06:16:37.801838+00:00 | `c3bd7423c139b7e4ecc53432d5486d5e79eea0b12de421e5f53a3628f2b2d12e` |
| `/home/user/repro/logs/vib2mol/retrieval_mols_ir_formula_0904_1433.log` | 2026-09-04T06:43:43.968822+00:00 | `14fc8de2190243c4d7ae434ad74eb77175791f96a46ecb2aae78f756951409cd` |
| `/home/user/repro/logs/vib2mol/retrieval_mols_ir_rerank_0904_1412.log` | 2026-09-04T06:21:08.380835+00:00 | `adfbe1d6a6ccd43a26cec457f649f018d6e2453cf5b96d4d4df38010b3861273` |
| `/home/user/repro/logs/vib2mol/retrieval_mols_raman_base_0904_1412.log` | 2026-09-04T06:21:56.492835+00:00 | `b3765e787496851eadc0fb8e355ea06d2b8aa4ca729825bebf552720f7164cf5` |
| `/home/user/repro/logs/vib2mol/retrieval_mols_raman_formula_0904_1433.log` | 2026-09-04T06:50:04.651818+00:00 | `08ab54130516d77d6871c273b98d25bf3baebffc78e8f0b89f846aac24c6dce7` |
| `/home/user/repro/logs/vib2mol/retrieval_mols_raman_rerank_0904_1412.log` | 2026-09-04T06:26:26.276832+00:00 | `ff8985a83913521a2b261ebe607d4a64d369434060c546c6de6d898065d62f51` |
| `/home/user/repro/logs/vib2mol/retrieval_nist_ir_base_0904_1412.log` | 2026-09-04T06:26:33.415832+00:00 | `a7fb7254ba9cb5818fad69b41325daea211663a36b992133f4b1d45dd8a5789e` |
| `/home/user/repro/logs/vib2mol/retrieval_nist_ir_formula_0904_1433.log` | 2026-09-04T06:50:24.119818+00:00 | `e485b72f2d7a0fea93c0c45564deeeb9f984545cd1e9d69a6287f08bf718d039` |
| `/home/user/repro/logs/vib2mol/retrieval_nist_ir_rerank_0904_1412.log` | 2026-09-04T06:26:49.489832+00:00 | `ec30e35424752333f0ad68abbd290c65938cbec0463ea540578cf216dec02ccf` |
| `/home/user/repro/logs/vib2mol/retrieval_qm9s_ir_base_0904_1412.log` | 2026-09-04T06:12:32.971840+00:00 | `739f9f8daecb88639f22cf9fdb864d96cbe93c8a7d2cfa4addb62d6e323e6654` |
| `/home/user/repro/logs/vib2mol/retrieval_qm9s_ir_formula_0904_1433.log` | 2026-09-04T06:35:15.820827+00:00 | `e95ce9840e794d7290b7db98c65d7e0380212e4b3a0a15380d9fea4cb1262e79` |
| `/home/user/repro/logs/vib2mol/retrieval_qm9s_ir_rerank_0904_1412.log` | 2026-09-04T06:14:01.351839+00:00 | `442256e1425a903933bb480456eac26012c119de46f7e60073dbcf5a58e56bf0` |
| `/home/user/repro/logs/vib2mol/retrieval_qm9s_raman_base_0904_1412.log` | 2026-09-04T06:14:20.420839+00:00 | `e970d6038d9af6f6bfb68b1225e9c8fa16ae3bc32b3d2d244c6b4fb3e29cc0b9` |
| `/home/user/repro/logs/vib2mol/retrieval_qm9s_raman_formula_0904_1433.log` | 2026-09-04T06:37:22.797826+00:00 | `c351349e07be131bfd1d72567d485fc7d10a6578c19105fd120d2099625f5d4b` |
| `/home/user/repro/logs/vib2mol/retrieval_qm9s_raman_rerank_0904_1412.log` | 2026-09-04T06:15:50.089838+00:00 | `4411d8268e8dd19f966d045729912f327f3ccc2934e1a6a4fefdbd569cb2ed3f` |
| `/home/user/repro/logs/vibraclip/official/nomass_0906_1208_allpairs.json` | 2026-09-06T04:11:49.748231+00:00 | `9f5afb6c3a6b7b5aaa4048ee90685c11f26999c8450e6bd8b9cbb374809c62e7` |
| `/home/user/repro/logs/vibraclip/official/nomass_0906_1208_ir.json` | 2026-09-06T04:11:49.025231+00:00 | `4fd2502ee14bffd50f46c0cc73a5d25d53c0d133f3788dda147863029bf7654e` |
| `/home/user/repro/logs/vibraclip/official/nomass_0906_1208_ir_raman.json` | 2026-09-06T04:11:43.457231+00:00 | `32e5eb52c6ee923221f61ac35873e01ad869c5e9559dd37af8a8684ed4415bda` |
| `/home/user/repro/logs/vibraclip/official/official_0905_2333_allpairs_mass.json` | 2026-09-05T15:35:27.698383+00:00 | `10efd8a26f8109bfd8b65f39d04cc213e9426a11ccfa3ed7722644d512be51ab` |
| `/home/user/repro/logs/vibraclip/official/official_0905_2333_ir_mass.json` | 2026-09-05T15:37:22.737671+00:00 | `ebf58b7e21694aee24ccb2d9f88083b5d10313cc6ed8536fd88e4b325c7c6249` |
| `/home/user/repro/report.md` | 2026-09-06T04:35:41+00:00 | `a1706c940050ded5290282f9c190d2d44e6ec06e231919b72faf45507eefba0d` |
| `/home/user/repro/unified/scored/raman_exact_v2_test200_20260909/summary.json` | 2026-09-09T07:22:34.927990+00:00 | `6b1b858e62b9e47fda7dba22c7d7630a2d4067d691c1e6e7dbda29cf375adbca` |
| `/home/user/repro/unified/scored/raman_exact_v2_test200_20260909_/summary.json` | 2026-09-09T07:17:39.569395+00:00 | `98cc94a35b4d4024cac9acd887f5879a727c07b011b385edf364701d9c41f9b4` |
| `/home/user/repro/unified/scored/raman_exact_v2_test200_20260909_all5/summary.json` | 2026-09-09T07:20:04.946499+00:00 | `d3d4eefcf089d7f85a68538dec724809dd7d93b648e2851f16406d4c64862e5d` |

## 最终核验

105 个远端来源文件 SHA-256 复核一致。latexmk 编译成功，引用全部解析，无 Overfull。PDF 为 13 页 US Letter；逐页渲染检查通过，去除了中间版的浮动表格整页留白。12 张编号结果表分为最新主实验、官方协议与历史协议。验证记录见 `results/jj-20260909/verification.json`。

最终 PDF SHA-256：`b123299afd3a0be507f79b5516614aabd06b049c5306de7091a5b17f3b638029`。

## 2026-09-09 表格样式修订

按用户要求复用前版 IEEE 主表的深蓝表头、四色方法分组、浅色交替行与 1.38 行距。共美化 12 张编号表，91 行数据和全部数值单元格与修订前一致；仅调整行分组/顺序、标题、列宽与说明位置。排版后为 14 页，无越界和未解析引用。此版 PDF 的 SHA-256 为 `68568bb4dc8634c241972b1fd868cdf81909a68f9e464813bc3449a565a05a71`；此前 13 页核验记录对应样式修订前版本。

## 10 页上限修订的最终检查

本次按 review.md 逐条核对并重组内容，14 页压到 9 页（含参考文献）。正文与实际 3 层配对编码器一致，补入覆盖归一化、来源仿射/展宽、矩形多正样本对比损失和训练组困难负样本。保留的 31 个表格数据行经数值复核；历史记录仍可追溯，不再占用正文页数。

新增 `latency_audit.json` 仅聚合已保存的 400 条成功 HTTP 尝试，逐条记录来源文件哈希；没有新 API 请求。encoder+LLM 与 kNN+LLM 的成功尝试均值分别为 21.499486 和 21.640433 秒。正文保留时间边界、187 次无 usage 的 403 和未知在途消耗，不推算货币成本。

latexmk 编译通过，无 Overfull 或未解析引用；字体与版心未用于缩页。PDF 页面边界检查没有越界字符，逐页渲染核查图、表、公式、末页双栏和提示词。完整检查与文件 SHA-256 见 `ten_page_verification.json`。

当前 PDF SHA-256：`736fc72a782774bc1681f5355eeb054e7d19b503382eea7ff83832f713892bc7`。

本轮总表合并后的 PDF SHA-256：`24c22049e67e1328d0e2e39fe5efd548dd3aea5ff8f1d22fb379aebe6624a184`。
