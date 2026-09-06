#!/usr/bin/env bash
# 功能概述：启动固定的 4 个消融变体 × 3 个随机种子 tiny pilot，仅验证端到端可行性。
# 主要职责：解析 data/output/GPU 参数，固定训练轮数、早停、batch 与模型尺寸，为 A–D 变体生成
# 十二条完整 ``spectraexpert.train_structure_aware`` 命令；``--dry-run`` 只打印 shell 可复现命令。
# 代码库关系：所有命令复用同一 ``split_manifest.json``，输出供后续统一评测入口消费；本脚本不准备
# 数据、不构建索引，也不声称 tiny 结果可作为论文最终结论。
# 输入输出与副作用：正常模式创建 output root 并按 variant/seed 写训练产物、占用指定可见 GPU；
# dry-run 不访问数据、不创建目录、不启动 Python。缺少必需参数或未知选项时以非零状态退出。

set -euo pipefail

DATA_DIR=""
OUTPUT_ROOT=""
GPU=""
DRY_RUN=0

while (($#)); do
    case "$1" in
        --data-dir)
            DATA_DIR=${2:?"--data-dir requires a value"}
            shift 2
            ;;
        --output-root)
            OUTPUT_ROOT=${2:?"--output-root requires a value"}
            shift 2
            ;;
        --gpu)
            GPU=${2:?"--gpu requires a value"}
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        *)
            printf 'unknown argument: %s\n' "$1" >&2
            exit 2
            ;;
    esac
done

if [[ -z "$DATA_DIR" || -z "$OUTPUT_ROOT" || -z "$GPU" ]]; then
    printf 'required: --data-dir DIR --output-root DIR --gpu ID [--dry-run]\n' >&2
    exit 2
fi

variants=(A B C D)
modes=(spectral_only_v1 spectrum_graph_v2 spectrum_graph_v2 spectrum_graph_v2)
lambda_aug=(1.0 0.25 0.25 0.25)
lambda_struct=(0.0 0.0 0.5 0.5)
lambda_aux=(0.0 0.0 0.0 0.1)
lambda_fingerprint=(0.0 0.0 0.0 0.0)

if ((DRY_RUN == 0)); then
    mkdir -p "$OUTPUT_ROOT"
fi

for variant_index in "${!variants[@]}"; do
    for seed in 0 1 2; do
        command=(
            python -m spectraexpert.train_structure_aware
            --training-mode "${modes[$variant_index]}"
            --data-dir "$DATA_DIR"
            --output-dir "$OUTPUT_ROOT/${variants[$variant_index]}_seed${seed}"
            --split-manifest "$DATA_DIR/split_manifest.json"
            --seed "$seed"
            --epochs 50
            --patience 8
            --batch-size 32
            --embed-dim 128
            --depth 3
            --num-heads 4
            --projection-dim 64
            --graph-hidden-dim 128
            --lambda-aug "${lambda_aug[$variant_index]}"
            --lambda-struct "${lambda_struct[$variant_index]}"
            --lambda-aux "${lambda_aux[$variant_index]}"
            --lambda-fingerprint "${lambda_fingerprint[$variant_index]}"
        )
        if ((DRY_RUN)); then
            printf 'env %q' "CUDA_VISIBLE_DEVICES=$GPU"
            printf ' %q' "${command[@]}"
            printf '\n'
        else
            env "CUDA_VISIBLE_DEVICES=$GPU" "${command[@]}"
        fi
    done
done
