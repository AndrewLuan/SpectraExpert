这次在严格 canonical SMILES 配对数据上完成了 8 组原始理论—实验对比图，以及一个仅有 6 个拟合参数的确定性曲线代理。它按理论来源共享波数缩放、平移和额外 Gaussian 展宽，没有神经网络，也没有逐验证样本配准。训练后可以仅凭理论曲线、coverage 和理论来源产生代理曲线。

数据沿用原来的分子组划分，不重新分配角色。Train 为 576 个分子、565 个组、874 个谱对组合；validation 为 145 个分子、142 个组、207 个谱对组合。受保护测试集只检查身份和分组元数据，未索引其光谱数组，也没有用于绘图、拟合或评价。读取数组采用内存映射，只索引允许的 train/validation 记录。

原始数据集留在服务器。本次只取回新生成的图、指标、配对审计记录和小型模型文件，没有下载全量理论/实验谱数组。

| 文件 | 用途 |
|---|---|
| `scripts/fit_raman_curve_proxy.py` | 独立绘图、参数拟合与验证入口 |
| `tests/scripts/test_fit_raman_curve_proxy.py` | 单峰数值、coverage、分组权重和角色保护检查 |
| `output/figures/raman_curve_proxy_20260909/fit/model.json` | 最终共享参数和优化状态 |
| `output/figures/raman_curve_proxy_20260909/fit/metrics.json` | 按组汇总及来源/相态分层结果 |
| `output/figures/raman_curve_proxy_20260909/fit/ablations.json` | 冻结参数后的分项诊断 |
| `output/figures/raman_curve_proxy_20260909/fit/surrogate_train_pairs.json` | train 全部允许谱对及逐对指标 |
| `output/figures/raman_curve_proxy_20260909/fit/surrogate_validation_pairs.json` | validation 全部允许谱对及逐对指标 |

每个角色的原始图和代理图分别为 `surrogate_train_original.png`、`surrogate_train_proxy.png`、`surrogate_validation_original.png`、`surrogate_validation_proxy.png`，同目录还有 SVG。每张图按理论来源 × LIQUID/SOLID 分为四格。每个分层先对分子组去重，再选择原始余弦相似度最接近中位数的记录；没有按代理改善幅度挑选，也没有用最佳配准挑选。UNKNOWN 相态仍参加拟合/评价，但未纳入四格展示。

蓝色是原始理论曲线，橙色是实验曲线，绿色是 train 拟合参数产生的代理曲线。展示范围为 500–4000 cm⁻¹，各曲线仅为显示按覆盖内最大值缩放，不再做基线校正。灰色表示原始两域至少一方缺少覆盖。代理还会收缩边界，以避开未知区域对卷积的污染；这部分表现为绿色曲线停止。图中平顶峰来自已有归一化曲线，不能用这些图恢复未归一化峰强。

代理的映射为 x_exp = a × x_theory + b，先在理论坐标中进行额外 Gaussian 展宽，再重采样到实验坐标；除以 a 保持该数值变换的积分。这里的积分守恒只约束输入曲线的数值变换，不意味着已有 robust-minmax 曲线保留了真实散射积分强度。sigma 使用理论坐标的 cm⁻¹，输出坐标中的额外宽度约为 a × sigma。

| 来源 | a | b / cm⁻¹ | 额外 sigma / cm⁻¹ |
|---|---:|---:|---:|
| QME14S | 0.98225345 | 39.42416 | 12.00000 |
| fg26 | 0.94515225 | 39.99949 | 11.99999 |

预设搜索边界为 a∈[0.94,1.02]、b∈[−40,40]、sigma∈[0,12]。目标是 train 谱形的 masked cosine，相同来源内各组等权，同一组内的谱对组合等权。先粗搜 a、b，再从两个较好起点做有界 Powell 优化。为节省成本，拟合目标每 3 点取样一次，输出结果用全分辨率检查。参数选择、粗搜和局部优化都只使用 train。

第一轮每次优化最多 240 次函数计算，得到可用参数但尚未全部收敛。最终运行只提高优化计算预算，没有扩大参数范围或增加参数；四次优化均报告成功收敛。服务器最终输出目录为：

```text
/home/user/IRfingerprints/SpectrExpert/outputs/raman_curve_proxy_fit_20260909_v2
```

拟合、主指标和四张图的记录耗时约 109 秒，分项诊断另需少量时间。全部使用 CPU，未分配 GPU；服务器任务退出码为 0。

评价域在拟合前固定：取实验覆盖、原始理论覆盖和所有搜索边界变换下都有效的理论区域的交集，并限制 x≥500。展宽核要求其截断支持位于已知覆盖内。原始与代理严格在相同位置比较，不能通过改变参数而移除难匹配的谱点。因此下面的数值不是完整 500–4000 区间指标，也不应与采用不同 mask 的旧实验直接比较。Train 每对至少保留 1646 点，validation 至少保留 1829 点。

| 评价范围 | 原始理论 | 代理预测 | 改善 |
|---|---:|---:|---:|
| Train，组等权 | 0.5155 | 0.7491 | +0.2336 |
| Validation，组等权 | 0.5036 | 0.7665 | +0.2629 |
| Validation / QME14S | 0.6454 | 0.7792 | +0.1338 |
| Validation / fg26 | 0.1996 | 0.7265 | +0.5269 |
| Validation / LIQUID | 0.5336 | 0.8142 | +0.2806 |
| Validation / SOLID | 0.4086 | 0.6032 | +0.1946 |

这里全部是余弦相似度，不是 Pearson 相关系数或分子检索准确率。Validation 平均改善的分子组配对 bootstrap 95% 区间为 [0.2348, 0.2912]，使用固定随机种子和 2000 次重采样；这反映给定拟合模型后的验证组抽样不确定性，不包含重新拟合模型的参数不确定性。同一分子可能具有两个理论来源，因此来源分层的组数不能直接相加。

将完整模型的参数冻结，再关闭部分变换得到以下诊断；没有对各消融单独重拟合，也没有据此选择最终模型。

| Validation | 原始 | 仅峰位变换 | 仅额外展宽 | 完整代理 |
|---|---:|---:|---:|---:|
| 全部，组等权 | 0.5036 | 0.6910 | 0.5734 | 0.7665 |
| QME14S | 0.6454 | 0.7080 | 0.7296 | 0.7792 |
| fg26 | 0.1996 | 0.6411 | 0.2380 | 0.7265 |

fg26 的改善主要来自可迁移的峰位修正；QME14S 的额外展宽贡献较大。两种来源的 sigma 都达到搜索上界，fg26 的平移也几乎达到上界，说明当前余弦目标偏好较宽的峰或仍有未建模差异。它们不能解释为真实仪器分辨率或标定误差，也不应仅为继续提高验证相似度而扩大边界。

固相样本仍有明显的相对峰强、背景和理论多余峰差异。该基线没有模拟荧光、计数噪声、激光、温度、取向或晶格模，也没有生成随机视图或训练后续检索编码器。当前证据支持先用它作为曲线域修正基线；若用于检索，还需要检验展宽是否损失分子间的判别信息。受保护测试集本次完全未评测。

复现时输出目录必须不存在。服务器命令如下：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/user/.venvs/spectrexpert/bin/python \
  /home/user/IRfingerprints/SpectrExpert/scripts/fit_raman_curve_proxy.py \
  --dataset /home/user/IRfingerprints/SpectrExpert/outputs/raman_surrogate_exact_pairs_v1 \
  --output /home/user/IRfingerprints/SpectrExpert/outputs/raman_curve_proxy_new_run \
  --mode fit
```

只画原始图时改为 `--mode preview`。复用参数时，加载 `model.json` 中相应 `sources[source]['params']`，调用 `transform(axis, theory, coverage, params)`，返回二维代理曲线及其有效 mask，不需要实验谱。函数保留变换后的数值幅度；如用于现有检索器，应遵循该检索器既有的输入归一化约定。

本地 4 项单元检查通过：单峰测试验证 identity、已知峰移动与积分守恒；其余三项分别验证共同有效区域、分子组权重和拟合角色保护。关键脚本的本地/服务器 SHA-256 一致：`327971aef637caf69bbde237666be5468871e105406a46e5fca3285c8b955a57`。输入 split SHA-256 为 `b87928f53bbfc081a57601092da7f92d88e6a8365d601f27997a4b2abdf93377`。
