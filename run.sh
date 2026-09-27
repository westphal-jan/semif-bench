#!/usr/bin/env bash
# Prepare, score, and evaluate a sample of PubMed RCT test abstracts.
# Usage: ./run.sh [ABSTRACTS=5] [SEED=0]
set -euo pipefail
cd "$(dirname "$0")"

abstracts="${1:-5}"
seed="${2:-0}"
model="Qwen/Qwen3.5-4B"
revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"

run_dir="results/$(date +%Y%m%d-%H%M%S)-n${abstracts}-s${seed}"
mkdir -p results
mkdir "$run_dir"

uv run python prepare.py --abstracts "$abstracts" --seed "$seed" --output "$run_dir/input.jsonl"

# Serial mode prefills each abstract once and reuses it for all of its sentences.
uv run semif-score --backend mlx --mode serial \
    --model "$model" --revision "$revision" \
    --input "$run_dir/input.jsonl" --output "$run_dir/predictions.jsonl"

uv run python evaluate.py --input "$run_dir/input.jsonl" --predictions "$run_dir/predictions.jsonl" \
    | tee "$run_dir/report.txt"
