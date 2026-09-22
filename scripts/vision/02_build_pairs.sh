#!/bin/bash
# scripts/vision/02_build_pairs.sh
# STAGE C — cache delle coppie positive (Fase 3) per ogni (modello × pooling valido).
# Per ogni pianta di train+valid: V viste degradate (+aug) encodate col backbone
# frozen -> embeddings/vision/<modello>/<pooling>/pairs.npz. Richiede lo STAGE A.
# PESANTE (V * 57k render+forward per config): conviene lanciarlo PER-MODELLO.
#
# Uso:
#   sbatch scripts/vision/02_build_pairs.sh dinov2     # consigliato: un modello per job (parallelo)
#   sbatch scripts/vision/02_build_pairs.sh            # tutti i modelli (lungo)
#
# $2 = una sola risoluzione (solo per gli encoder che ne hanno piu' di una,
# vedi resolutions_for in _common.sh): permette un job per risoluzione.
# env POOLS = pooling (vuoto = tutti) · EXTRA = override dotlist (15 set 2026, status.md §46):
#   POOLS=gem EXTRA="training.damage=nowalls_random" sbatch scripts/vision/02_build_pairs.sh pespatial

#SBATCH --job-name="vx_02_build_pairs"
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=18:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
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

POOLS="${POOLS:-}"          # vuoto = tutti i pooling dell'encoder (es. POOLS=gem)
EXTRA="${EXTRA:-}"          # override dotlist aggiuntivi (es. EXTRA="training.damage=nowalls_random")

echo "=== $(date) | STAGE C coppie positive | modelli: $(select_models "$1") | pooling: ${POOLS:-tutti} | extra: ${EXTRA:-nessuno} ==="
nvidia-smi

for MODEL in $(select_models "$1"); do
  for POOL in ${POOLS:-$(poolings_for "$MODEL")}; do
    for RES in $(select_resolutions "$MODEL" "${2:-}"); do
      VAR=$(variant_for "$POOL" "$RES")
      RFLAGS=$(res_flags "$RES")
      echo ""
      echo "########  PAIRS: $(label_for "$MODEL" "$POOL" "$RES")  ########"
      python -m src.vision.data.projection_pairs \
          model.name=$MODEL model.variant=$VAR model.kwargs.pooling=$POOL $RFLAGS $EXTRA
    done
  done
done

echo ""
echo "=== completato: $(date) ==="
