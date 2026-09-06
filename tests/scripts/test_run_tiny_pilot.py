"""
功能概述：验证 tiny pilot launcher 的固定 4×3 消融命令矩阵与 shell 语法。
主要职责：执行 ``--dry-run`` 并断言恰好十二条训练命令、三个 seed、同一 split manifest、
固定模型/训练超参数及 A–D 四组 loss；另以 ``bash -n`` 检查脚本可解析。
代码库关系：脚本调用 ``spectraexpert.train_structure_aware``，其输出仅供 Task 10 远端 pilot；
本测试不调用训练模块本身，也不把微型实验结果解释为论文结论。
输入输出与副作用：使用临时 data/output 路径，只捕获 dry-run 标准输出；不创建 checkpoint、
不占用 GPU、不读取真实 Raman 文件且不启动远程实验。
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "run_tiny_pilot.sh"


def test_tiny_pilot_dry_run_has_fixed_twelve_command_matrix(tmp_path):
    data_dir = tmp_path / "prepared data"
    output_root = tmp_path / "pilot output"

    completed = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--data-dir",
            str(data_dir),
            "--output-root",
            str(output_root),
            "--gpu",
            "2",
            "--dry-run",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    lines = completed.stdout.splitlines()
    assert len(lines) == 12
    commands = [shlex.split(line) for line in lines]
    assert all(command[0] == "env" and command[1] == "CUDA_VISIBLE_DEVICES=2" for command in commands)
    assert {command[command.index("--seed") + 1] for command in commands} == {"0", "1", "2"}
    assert {
        command[command.index("--split-manifest") + 1] for command in commands
    } == {str(data_dir / "split_manifest.json")}
    assert all(command[command.index("--epochs") + 1] == "50" for command in commands)
    assert all(command[command.index("--patience") + 1] == "8" for command in commands)
    assert all(command[command.index("--batch-size") + 1] == "32" for command in commands)
    assert all(command[command.index("--embed-dim") + 1] == "128" for command in commands)
    assert all(command[command.index("--depth") + 1] == "3" for command in commands)
    assert all(command[command.index("--num-heads") + 1] == "4" for command in commands)
    assert all(command[command.index("--projection-dim") + 1] == "64" for command in commands)
    assert all(command[command.index("--graph-hidden-dim") + 1] == "128" for command in commands)
    matrix = {
        (
            command[command.index("--training-mode") + 1],
            command[command.index("--lambda-aug") + 1],
            command[command.index("--lambda-struct") + 1],
            command[command.index("--lambda-aux") + 1],
            command[command.index("--lambda-fingerprint") + 1],
        )
        for command in commands
    }
    assert matrix == {
        ("spectral_only_v1", "1.0", "0.0", "0.0", "0.0"),
        ("spectrum_graph_v2", "0.25", "0.0", "0.0", "0.0"),
        ("spectrum_graph_v2", "0.25", "0.5", "0.0", "0.0"),
        ("spectrum_graph_v2", "0.25", "0.5", "0.1", "0.0"),
    }


def test_tiny_pilot_shell_syntax_is_valid():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
