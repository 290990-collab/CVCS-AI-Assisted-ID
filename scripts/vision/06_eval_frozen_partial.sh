#!/bin/bash
# scripts/vision/06_eval_frozen_partial.sh
# EVAL PARTIAL, contributi FROZEN: raw + whitening, per ogni (modello × pooling).
# Riusa solo lo STAGE A. Partial = valutazione principale (self-recovery + per-asse
# vs pianta completa); molto più pesante della full -> lanciare PER-MODELLO.
#
# Uso:
#   sbatch scripts/vision/06_eval_frozen_partial.sh dinov2   # consigliato: un modello per job
#   sbatch scripts/vision/06_eval_frozen_partial.sh          # tutti i modelli (molto lungo)
#
# $2 = una sola risoluzione (solo per gli encoder che ne hanno piu' di una,
# vedi resolutions_for in _common.sh): permette un job per risoluzione.

#SBATCH --job-name="vx_06_eval_frozen_partial"
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:2
#SBATCH --partition=all_usr_prod
#SBATCH --account=cvcs2026
# Esclude i nodi Blackwell (sm_120), su cui il PyTorch di floorplan-env crasha
# a ogni forward. Stessa whitelist di 07_eval_head_partial.sh.
#SBATCH --constraint="gpu_RTX5000_16G|gpu_2080_11G|gpu_2080Ti_11G|gpu_P100_16G|gpu_RTX_A5000_24G|gpu_A40_45G|gpu_L40S_45G"

source ~/floorplan-env/bin/activate
source /work/cvcs2026/ai_interior_design/CVCS-AI-Assisted-ID/scripts/vision/_common.sh
cd "$PROJECT_DIR"
mkdir -p logs

echo "=== $(date) | EVAL PARTIAL frozen (raw + whiten) | modelli: $(select_models "$1") ==="
nvidia-smi

for MODEL in $(select_models "$1"); do
  for POOL in $(poolings_for "$MODEL"); do
    for RES in $(select_resolutions "$MODEL" "${2:-}"); do
      VAR=$(variant_for "$POOL" "$RES")
      RFLAGS=$(res_flags "$RES")
      COMMON="model.name=$MODEL model.variant=$VAR model.kwargs.pooling=$POOL $RFLAGS partial.enabled=true eval.split=test"
      echo ""
      echo "########  EVAL PARTIAL: $(label_for "$MODEL" "$POOL" "$RES") / raw  ########"
      python -m src.vision.evaluation.evaluate $COMMON head.enabled=false whitening.enabled=false
      echo ""
      echo "########  EVAL PARTIAL: $(label_for "$MODEL" "$POOL" "$RES") / whiten  ########"
      python -m src.vision.evaluation.evaluate $COMMON head.enabled=false whitening.enabled=true
    done
  done
done

echo ""
echo "=== completato: $(date) ==="
