"""功能概述：组织结构感知训练的持久化与验证模块，不执行训练或导入数据。
主要职责：子模块 checkpoints 提供版本隔离的原子 checkpoint；validation 提供检索指标。
代码库关系：由 train_structure_aware CLI 使用；旧版训练入口保持独立。
输入输出与副作用：导入本包无文件、网络或 GPU 副作用。
"""
