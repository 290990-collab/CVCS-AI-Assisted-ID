#!/bin/bash
# scripts/vision/05_eval_head_full.sh
# EVAL FULL, contributi con HEAD: head + head+whitening, per ogni (modello × pooling).
# Richiede STAGE A + STAGE C + STAGE D (head.pt presente). Query = split $EVAL_SPLIT (default valid).
# Confronta con 04 (frozen) per misurare il contributo della head.
#
# Uso:
#   sbatch scripts/vision/05_eval_head_full.sh            # tutti i modelli
#   sbatch scripts/vision/05_eval_head_full.sh dinov2     # un solo modello
#
# $2 = una sola risoluzione (solo per gli encoder che ne hanno piu' di una,
# vedi resolutions_for in _common.sh): permette un job per risoluzione.

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
# Esclude i nodi Blackwell (sm_120), su cui il PyTorch di floorplan-env crasha
# a ogni forward. Stessa whitelist di 07_eval_head_partial.sh.
#SBATCH --constraint="gpu_RTX5000_16G|gpu_2080_11G|gpu_2080Ti_11G|gpu_P100_16G|gpu_RTX_A5000_24G|gpu_A40_45G|gpu_L40S_45G"

source ~/floorplan-env/bin/activate
source /work/cvcs2026/ai_interior_design/CVCS-AI-Assisted-ID/scripts/vision/_common.sh
cd "$PROJECT_DIR"
mkdir -p logs

# Split delle query. Default `valid`: il vincolo DURO 1 dice che il test non
# sceglie nulla, e fino al 24 ago questi script avevano `test` cablato senza
# override (rilievo A1). Per rileggere il test va chiesto esplicitamente:
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
