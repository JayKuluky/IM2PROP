#!/usr/bin/env bash
set -euo pipefail

# Run all 8 combinations for:
# USE_PHASE_RATIOS, USE_PHASE_ATTENTION, USE_PHASE_FEAT
# Each combination is repeated 5 times.

REPEATS=5

combo_id=1
for ratios in false true; do
  for attention in false true; do
    for feat in false true; do
      printf -v combo_tag "combo%02d_r%s_a%s_f%s" \
        "$combo_id" \
        "$([[ "$ratios" == "true" ]] && echo 1 || echo 0)" \
        "$([[ "$attention" == "true" ]] && echo 1 || echo 0)" \
        "$([[ "$feat" == "true" ]] && echo 1 || echo 0)"
        wandb_group="${combo_tag}_30_epochs"

      for rep in $(seq 1 "$REPEATS"); do
        run_name="${combo_tag}_${rep}_30_epochs"
          echo "Running ${run_name} (WANDB_GROUP=${wandb_group})"

        uv run scripts/run_pipeline.py \
          --MODE train-test \
          --ENABLE_GRADCAM true \
          --WANDB_GROUP "${wandb_group}" \
          --NUM_EPOCHS 30 \
          --USE_OLD_MASKS true \
          --ENABLE_PATCHING false \
          --USE_PHASE_RATIOS "${ratios}" \
          --USE_PHASE_ATTENTION "${attention}" \
          --USE_PHASE_FEAT "${feat}" \
          --RUN_NAME "${run_name}"
      done

      combo_id=$((combo_id + 1))
    done
  done
done

printf "\nAll phase-combination runs completed successfully.\n"
