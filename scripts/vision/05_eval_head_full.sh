#!/bin/bash
# Full eval, head contributions: head + head+whitening, per (model x pooling).
# Needs stages A, C, D (head.pt). Queries = $EVAL_SPLIT split (default valid). Compare with 04 (frozen) for the head's contribution.
#
# Uso:
#   sbatch scripts/vision/05_eval_head_full.sh            # all models
#   sbatch scripts/vision/05_eval_head_full.sh dinov2     # one model
#
# $2 = single resolution (only for multi-resolution encoders, see resolutions_for in _common.sh): one job per resolution.

#SBATCH --job-name="vx_05_eval_head_full"
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=all_usr_prod
#SBATCH --account=cvcs2026
# Exclude Blackwell nodes (sm_120): floorplan-env PyTorch crashes on every forward. Same whitelist as 07_eval_head_partial.sh.
#SBATCH --constraint="gpu_RTX5000_16G|gpu_2080_11G|gpu_2080Ti_11G|gpu_P100_16G|gpu_RTX_A5000_24G|gpu_A40_45G|gpu_L40S_45G"

source ~/floorplan-env/bin/activate
source /work/cvcs2026/ai_interior_design/CVCS-AI-Assisted-ID/scripts/vision/_common.sh
cd "$PROJECT_DIR"
mkdir -p logs

# Query split, default `valid` (test selects nothing); to read the test explicitly:
#   EVAL_SPLIT=test sbatch <questo script> [modello] [risoluzione]
EVAL_SPLIT="${EVAL_SPLIT:-valid}"

echo "=== $(date) | EVAL FULL head (head + head+whiten) | modelli: $(select_models "$1") | split: $EVAL_SPLIT ==="
nvidia-smi

for MODEL in $(select_models "$1"); do
  for POOL in $(poolings_for "$MODEL"); do
    for RES in $(select_resolutions "$MODEL" "${2:-}"); do
      VAR=$(variant_for "$POOL" "$RES")
      RFLAGS=$(res_flags "$RES")
      COMMON="model.name=$MODEL model.variant=$VAR model.kwargs.pooling=$POOL $RFLAGS partial.enabled=false eval.split=$EVAL_SPLIT head.enabled=true"
      echo ""
      echo "########  EVAL FULL: $(label_for "$MODEL" "$POOL" "$RES") / head  ########"
      python -m src.vision.evaluation.evaluate $COMMON whitening.enabled=false
      echo ""
      echo "########  EVAL FULL: $(label_for "$MODEL" "$POOL" "$RES") / head+whiten  ########"
      python -m src.vision.evaluation.evaluate $COMMON whitening.enabled=true
    done
  done
done

echo ""
echo "=== completato: $(date) ==="
