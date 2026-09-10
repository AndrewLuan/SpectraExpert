# 使用 SpecGPT 理论 Raman 训练

`scripts/prepare_specgpt_theory.py` 读取 continuous_v1 处理清单中的理论 Raman 分片。
它不读取实验 IR/Raman，也不使用 GPT 对话格式。清单给出的 Raman 波数轴是
0–4000 cm⁻¹、3000 点；`coverage` 是数组覆盖信息，不能用作波数轴。
转换以真实坐标线性插值到 500–4000 cm⁻¹、3501 点，保持现有光谱编码器输入约定。
输入已做 robust min-max 处理，转换不重复归一化；训练器仍按样本标准化。

在服务器项目根目录、安装了项目依赖的 Python 环境中运行：

```bash
python -m scripts.prepare_specgpt_theory \
  --manifest /data/SpecGPT/dataset/processed_stage_2_v1/continuous/manifests/processed_manifest_theory_raman.json \
  --output-dir /home/user/IRfingerprints/specgpt_theory_raman_3501

python -m spectraexpert.train_structure_aware \
  --training-mode spectrum_graph_v2 \
  --data-dir /home/user/IRfingerprints/specgpt_theory_raman_3501 \
  --output-dir /home/user/IRfingerprints/runs/specgpt_theory_raman_v2
```

先用转换器的 `--max-records-per-file 16` 和新的输出目录验证小样本流程。
该参数逐源、逐 split 限制读取记录数，属于调试前缀样本，不能用于正式性能结论。
去掉参数才转换全量。全量转换记录各输入压缩文件 SHA-256；小样本模式不为计算 hash
额外扫描整份源数据。输出目录已存在时拒绝覆盖。

数据行使用唯一整数 `mol_id`；同一分子的多条谱保留，由训练器的 canonical identity
构造 multi-positive。原 train/valid/test 分组保持不变，valid 映射成训练器的 validation。
`split.json` 使用 `specgpt_frozen_v1`，不是重新生成的 scaffold 划分。转换检查 group_hash
和 canonical SMILES 的跨集泄漏，发现泄漏时失败；非法 SMILES、非有限/错误长度光谱、
目标频段内常数谱被过滤并计数。处理后的每个 split 必须非空。

`preparation_metadata.json` 保存原清单、转换参数、逐文件统计及来源 hash。
训练入口继续验证三件套 hash、identity 隔离和 split 覆盖，仅用训练数据拟合描述符统计，
使用理论 validation 选择 checkpoint。实验数据不参与训练或模型选择。
独立冻结集评测 CLI 目前仍要求本项目 scaffold manifest；不要把外部 split 改名冒充它。

转换分批写 Parquet，但现有训练 Dataset 仍将谱矩阵和所选分子图载入内存。
正式训练应在服务器进行，启动前检查 `nvidia-smi` 和主机内存；多卡时每个进程有自己的
Dataset 内存开销。使用空闲 GPU 和 tmux 运行长期训练。这些命令从头训练新模型，
`--resume-from` 仅适用于同一数据及配置的训练恢复。
