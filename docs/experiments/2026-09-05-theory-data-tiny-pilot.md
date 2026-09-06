# theory_data_tiny 结构感知检索 pilot

状态：远程执行阻塞，不能据此作出论文结论。

本 pilot 计划只读取服务器授权的 `Raman_QME14S.jsonl` 与 `Raman_fg26.jsonl`，统一插值到 500--4000 cm⁻¹ 的 3501 点，并使用固定 scaffold split、A/B/C/D 四种 loss 配置和 seed 0/1/2。远端正式产物应保留在 `/home/user/IRfingerprints/SpectrExpert/outputs/`，不提交到仓库。

截至 2026-09-06，JJ SSH `user@114.214.255.164:22` 在 `ConnectTimeout=10` 时出现 `Connection timed out during banner exchange`，因此尚未完成同步、GPU 选择、数据准备、smoke 或 tmux pilot。恢复 SSH 后应按 `docs/superpowers/plans/2026-09-05-structure-aware-spectral-retrieval.md` Task 10 的命令顺序重跑，并记录 GPU、依赖、首轮日志、checkpoint/index 与 identity-included/excluded 评测。

本地代码在 Task 9 修复后已通过完整测试：429 passed，274 条既有 PyTorch nested-tensor warning（来源 `spectrum_encoder.py:134`）。本报告明确标注：授权 tiny 数据最多 400 条，仅用于工程 pilot，不能作为最终论文结论。
