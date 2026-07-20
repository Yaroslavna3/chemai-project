#!/bin/bash
#SBATCH --partition=aichem
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -euo pipefail

TARGET="${1:?Usage: train.sh <1kzn|3fqs> <llm_docking|all_properties|docking_metrics> [epochs]}"
TRAJECTORY="${2:?Usage: train.sh <1kzn|3fqs> <llm_docking|all_properties|docking_metrics> [epochs]}"
EPOCHS="${3:-50}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

PROJECT_ROOT="$(resolve_project_root)"
activate_runtime "$PROJECT_ROOT"
TARGET_ARGS="$(target_args "$PROJECT_ROOT" "$TARGET")"
OBJECTIVE_ARGS="$(objective_args "$TRAJECTORY")"
EXPERIMENT_NAME="${TARGET}_${TRAJECTORY}_${EPOCHS}_ep"

cd "$PROJECT_ROOT/freedpp"
python main.py \
    --exp_root "$PROJECT_ROOT/experiments" \
    --name "$EXPERIMENT_NAME" \
    --commands "train" \
    --epochs "$EPOCHS" \
    --save_freq 5 \
    --alert_collections "$PROJECT_ROOT/alert_collections.csv" \
    --fragments "$PROJECT_ROOT/zinc_crem.json" \
    --vina_program "$PROJECT_ROOT/freedpp/env/qvina02" \
    --starting_smile "c1([*:1])c([*:2])ccc([*:3])c1" \
    --fragmentation crem \
    --num_sub_proc 12 \
    --n_conf 1 \
    --exhaustiveness 1 \
    --reward_version soft \
    --seed 150 \
    $TARGET_ARGS \
    $OBJECTIVE_ARGS
