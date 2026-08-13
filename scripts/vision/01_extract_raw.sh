#!/bin/bash
# scripts/vision/01_extract_raw.sh
# STAGE A — estrazione embedding RAW per ogni (modello × pooling valido).
# Salva embeddings/vision/<modello>/<pooling>/embeddings.npy (RAW, senza whitening).
# Unico artefatto costoso: lo riusano whitening/head e tutte le valutazioni.
#
# Uso:
#   sbatch scripts/vision/01_extract_raw.sh            # tutti i modelli (sequenziale)
#   sbatch scripts/vision/01_extract_raw.sh dinov2     # un solo modello (per parallelizzare)
#   sbatch scripts/vision/01_extract_raw.sh tipsv2 448 # una sola risoluzione ($2)
#
# $2 serve solo agli encoder con piu' di una risoluzione (vedi resolutions_for
# in _common.sh): permette di lanciare un job per risoluzione, in parallelo.

#SBATCH --job-name="vx_01_extract_raw"
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
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
mkdir -p logs embeddings/vision

echo "=== $(date) | STAGE A estrazione RAW | modelli: $(select_models "$1") ==="
nvidia-smi

for MODEL in $(select_models "$1"); do
  for POOL in $(poolings_for "$MODEL"); do
    for RES in $(select_resolutions "$MODEL" "${2:-}"); do
      VAR=$(variant_for "$POOL" "$RES")
      echo ""
      echo "########  RAW: $(label_for "$MODEL" "$POOL" "$RES")  ########"
      python -m tests.test_vision_retrieval \
          model.name=$MODEL model.variant=$VAR model.kwargs.pooling=$POOL $(res_flags "$RES")
    done
  done
done

echo ""
echo "=== completato: $(date) ==="
