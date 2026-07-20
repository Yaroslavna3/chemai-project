#!/bin/bash
#SBATCH --partition=aichem
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00

set -euo pipefail

TARGET="${1:?Usage: sample_1000.sh <1kzn|3fqs> <llm_docking|all_properties|docking_metrics> <checkpoint> [sample_name]}"
TRAJECTORY="${2:?Usage: sample_1000.sh <1kzn|3fqs> <llm_docking|all_properties|docking_metrics> <checkpoint> [sample_name]}"
CHECKPOINT="${3:?Usage: sample_1000.sh <1kzn|3fqs> <llm_docking|all_properties|docking_metrics> <checkpoint> [sample_name]}"
SAMPLE_NAME="${4:-${TARGET}_${TRAJECTORY}_sample_1000}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

PROJECT_ROOT="$(resolve_project_root)"
activate_runtime "$PROJECT_ROOT"
TARGET_ARGS="$(target_args "$PROJECT_ROOT" "$TARGET")"
OBJECTIVE_ARGS="$(objective_args "$TRAJECTORY")"

cd "$PROJECT_ROOT/freedpp"
python main.py \
    --exp_root "$PROJECT_ROOT/experiments" \
    --name "$SAMPLE_NAME" \
    --commands "sample" \
    --num_mols 1000 \
    --checkpoint "$CHECKPOINT" \
    --alert_collections "$PROJECT_ROOT/alert_collections.csv" \
    --fragments "$PROJECT_ROOT/zinc_crem.json" \
    --vina_program "$PROJECT_ROOT/freedpp/env/qvina02" \
    --starting_smile "c1([*:1])c([*:2])ccc([*:3])c1" \
    --fragmentation crem \
    --reward_version soft \
    --seed 150 \
    $TARGET_ARGS \
    $OBJECTIVE_ARGS
