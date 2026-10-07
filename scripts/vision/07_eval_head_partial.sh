#!/bin/bash
# Partial eval, head contributions: head + head+whitening, per (model x pooling).
# Needs stages A, C, D. Heavy: run per model. Compare with 06 (frozen) for the head's contribution.
#
# Uso:
#   sbatch scripts/vision/07_eval_head_partial.sh dinov2    # recommended: one model per job
#   sbatch scripts/vision/07_eval_head_partial.sh           # all models (very long)
#
# $2 = single resolution (only for multi-resolution encoders, see resolutions_for in _common.sh): one job per resolution.

#SBATCH --job-name="vx_07_eval_head_partial"
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --partition=all_usr_prod
#SBATCH --account=cvcs2026
#SBATCH --constraint="gpu_RTX5000_16G|gpu_2080_11G|gpu_2080Ti_11G|gpu_P100_16G|gpu_RTX_A5000_24G|gpu_A40_45G|gpu_L40S_45G"

source ~/floorplan-env/bin/activate
source /work/cvcs2026/ai_interior_design/CVCS-AI-Assisted-ID/scripts/vision/_common.sh
cd "$PROJECT_DIR"
mkdir -p logs

# Query split, default `valid` (test selects nothing); to read the test explicitly:
#   EVAL_SPLIT=test sbatch <questo script> [modello] [risoluzione]
EVAL_SPLIT="${EVAL_SPLIT:-valid}"

echo "=== $(date) | EVAL PARTIAL head (head + head+whiten) | modelli: $(select_models "$1") | split: $EVAL_SPLIT ==="
nvidia-smi

for MODEL in $(select_models "$1"); do
  for POOL in $(poolings_for "$MODEL"); do
    for RES in $(select_resolutions "$MODEL" "${2:-}"); do
      VAR=$(variant_for "$POOL" "$RES")
      RFLAGS=$(res_flags "$RES")
      COMMON="model.name=$MODEL model.variant=$VAR model.kwargs.pooling=$POOL $RFLAGS partial.enabled=true head.enabled=true eval.split=$EVAL_SPLIT"
      echo ""
      echo "########  EVAL PARTIAL: $(label_for "$MODEL" "$POOL" "$RES") / head  ########"
      python -m src.vision.evaluation.evaluate $COMMON whitening.enabled=false
      echo ""
      echo "########  EVAL PARTIAL: $(label_for "$MODEL" "$POOL" "$RES") / head+whiten  ########"
      python -m src.vision.evaluation.evaluate $COMMON whitening.enabled=true
    done
  done
done

echo ""
echo "=== completato: $(date) ==="
