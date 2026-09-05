# 功能概述：定义 SpectraExpert 的可安装包元数据与运行/开发依赖，是本仓库
# ``pip install -e .[dev]`` 可复现导入测试的唯一打包入口。
# 主要职责：声明 ``spectraexpert`` 子包、基础运行依赖和 pytest 开发 extra；不
# 承载训练或推理逻辑。
# 代码库关系：本文件由 pip/setuptools 读取；``spectraexpert`` 下的训练、管线、
# 评估模块依赖此处固定的 torch、NetworkX、YAML 与 TensorBoard 版本下限。
# 输入输出与副作用：setuptools 根据这些常量构建或以 editable 方式安装包，安装
# 会解析依赖；不读取实验数据，也不写入项目源码。

from setuptools import find_packages, setup

setup(
    name="spectraexpert",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.0",
        "scipy>=1.10",
        "pyarrow>=12.0",
        "matplotlib>=3.7",
        "rdkit>=2023.9",
        "scikit-learn>=1.3",
        "torch>=2.0",
        "transformers>=4.35",
        "networkx>=3.1",
        "pyyaml>=6.0",
        "tensorboard>=2.14",
    ],
    extras_require={"dev": ["pytest>=8.0"]},
)
