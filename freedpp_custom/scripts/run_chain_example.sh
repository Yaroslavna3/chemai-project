#!/bin/bash

set -euo pipefail

TARGET="${1:-1kzn}"
TRAJECTORY="${2:-llm_docking}"
EPOCHS="${3:-50}"
TRAIN_NAME="${TARGET}_${TRAJECTORY}_${EPOCHS}_ep"
SAMPLE_NAME="${TARGET}_${TRAJECTORY}_sample_1000"
CHECKPOINT="experiments/${TRAIN_NAME}/ckpt/model_$(printf '%03d' "$EPOCHS").pth"

train_job=$(sbatch --parsable scripts/train.sh "$TARGET" "$TRAJECTORY" "$EPOCHS")
sample_job=$(sbatch --parsable --dependency=afterok:"$train_job" scripts/sample_1000.sh "$TARGET" "$TRAJECTORY" "$CHECKPOINT" "$SAMPLE_NAME")
eval_job=$(sbatch --parsable --dependency=afterok:"$sample_job" scripts/evaluate.sh "$TARGET" "$SAMPLE_NAME" "$EPOCHS")

echo "Submitted train job: $train_job"
echo "Submitted sample job: $sample_job"
echo "Submitted evaluate job: $eval_job"
