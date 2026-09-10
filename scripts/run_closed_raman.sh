#!/usr/bin/env bash
# 功能概述：持久启动冻结 Raman 9500/500 主方法，分别支持 smoke 和正式训练。
# 主要职责：参数为 mode/GPU/run_name；独占创建运行目录，保存命令、PID、开始/结束时间和
# exit_code，使用交接文档参数调用 train_structure_aware。smoke 仅限制训练为 64 条、一轮。
# 代码库关系：使用当前 spectrexpert 环境及项目 outputs 下已冻结的数据和 training_split；
# SPECTRAEXPERT_PYTHON 和 SPECTRAEXPERT_CLOSED_DATA 可分别覆盖 Python 解释器和数据目录。
# 输入输出与副作用：不改源数据；仅写新运行目录。调用者用 nohup 执行以跨 SSH 断线存活，
# 失败返回真实退出码；禁止复用任何已有运行目录，不自动恢复或触碰旧 checkpoint。
set -euo pipefail
mode=${1:?mode must be smoke or train}
gpu=${2:?choose a currently free GPU}
name=${3:?unique run name required}
[[ "$mode" == smoke || "$mode" == train ]]
[[ "$gpu" =~ ^[0-9]+$ ]]
[[ "$name" =~ ^raman_closed_[a-zA-Z0-9_]+$ ]]
project=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
run="$project/outputs/$name"
mkdir -- "$run"
exec >"$run/launcher.log" 2>&1
date -u +%FT%TZ >"$run/started_at"
trap 'rc=$?; date -u +%FT%TZ >"$run/finished_at"; printf "%s\n" "$rc" >"$run/exit_code"' EXIT
export CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=4
cd -- "$project"
python=${SPECTRAEXPERT_PYTHON:-python}
data=${SPECTRAEXPERT_CLOSED_DATA:-$project/outputs/raman_closed_10k_test200_seed0}
epochs=30
extra=()
if [[ "$mode" == smoke ]]; then
    epochs=1
    extra=(--max-samples 64)
fi
command=("$python" -u -m spectraexpert.train_structure_aware
    --training-mode spectrum_graph_v2
    --data-dir "$data/theory_train_candidate" --split-manifest "$data/training_split.json"
    --output-dir "$run" --seed 0 --epochs "$epochs" --patience 5 --batch-size 32 --num-workers 0
    --learning-rate 0.001 --embed-dim 128 --depth 3 --num-heads 4 --projection-dim 64
    --graph-hidden-dim 128 --lambda-aug 0.25 --lambda-struct 0.5 --lambda-aux 0.1
    --lambda-fingerprint 0.0 "${extra[@]}")
printf 'CUDA_VISIBLE_DEVICES=%q OMP_NUM_THREADS=4 ' "$gpu" >"$run/command.txt"
printf '%q ' "${command[@]}" >>"$run/command.txt"
printf '\n' >>"$run/command.txt"
"$python" -m pip freeze >"$run/environment.txt"
"${command[@]}" >"$run/train.log" 2>&1 &
child=$!
printf '%s\n' "$child" >"$run/train.pid"
wait "$child"
