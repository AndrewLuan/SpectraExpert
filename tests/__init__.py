"""
功能概述：定义 SpectraExpert 的测试包边界，使 pytest 能稳定发现本目录下的可复现基线测试。
主要职责：本文件不导出运行时代码；它仅标识 ``tests`` 为 Python 包，以便测试模块可使用一致的
导入和收集规则。
代码库关系：由根目录 ``pytest.ini`` 的 ``testpaths`` 指向，当前承载
``test_package_imports.py`` 对公开包导入及 legacy retrieval 契约的验证。
输入输出与副作用：不接收输入、不写文件，也不在导入时配置测试环境；测试行为完全由各测试模块承担。
"""
