#!/usr/bin/env bash
set -euo pipefail

# Run 8 combinations for:
# USE_PHASE_RATIOS, USE_PHASE_ATTENTION, USE_PHASE_FEAT
# Mode is train-test, Grad-CAM is enabled, and each run name is unique.

uv run scripts/run_pipeline.py \
  --MODE train-test \
  --ENABLE_GRADCAM true \
  --USE_PHASE_RATIOS false \
  --USE_PHASE_ATTENTION false \
  --USE_PHASE_FEAT false \
  --RUN_NAME combo01_r0_a0_f0_1

uv run scripts/run_pipeline.py \
  --MODE train-test \
  --ENABLE_GRADCAM true \
  --USE_PHASE_RATIOS false \
  --USE_PHASE_ATTENTION false \
  --USE_PHASE_FEAT true \
  --RUN_NAME combo02_r0_a0_f1_1

uv run scripts/run_pipeline.py \
  --MODE train-test \
  --ENABLE_GRADCAM true \
  --USE_PHASE_RATIOS false \
  --USE_PHASE_ATTENTION true \
  --USE_PHASE_FEAT false \
  --RUN_NAME combo03_r0_a1_f0_1

uv run scripts/run_pipeline.py \
  --MODE train-test \
  --ENABLE_GRADCAM true \
  --USE_PHASE_RATIOS false \
  --USE_PHASE_ATTENTION true \
  --USE_PHASE_FEAT true \
  --RUN_NAME combo04_r0_a1_f1_1

uv run scripts/run_pipeline.py \
  --MODE train-test \
  --ENABLE_GRADCAM true \
  --USE_PHASE_RATIOS true \
  --USE_PHASE_ATTENTION false \
  --USE_PHASE_FEAT false \
  --RUN_NAME combo05_r1_a0_f0_1

uv run scripts/run_pipeline.py \
  --MODE train-test \
  --ENABLE_GRADCAM true \
  --USE_PHASE_RATIOS true \
  --USE_PHASE_ATTENTION false \
  --USE_PHASE_FEAT true \
  --RUN_NAME combo06_r1_a0_f1_1

uv run scripts/run_pipeline.py \
  --MODE train-test \
  --ENABLE_GRADCAM true \
  --USE_PHASE_RATIOS true \
  --USE_PHASE_ATTENTION true \
  --USE_PHASE_FEAT false \
  --RUN_NAME combo07_r1_a1_f0_1

uv run scripts/run_pipeline.py \
  --MODE train-test \
  --ENABLE_GRADCAM true \
  --USE_PHASE_RATIOS true \
  --USE_PHASE_ATTENTION true \
  --USE_PHASE_FEAT true \
  --RUN_NAME combo08_r1_a1_f1_1

printf "\nAll phase-combination runs completed successfully.\n"
