#!/bin/bash
# Stage A: extract RAW embeddings for each (model x valid pooling).
# Writes embeddings/vision/<model>/<pooling>/embeddings.npy (raw, no whitening); the one expensive artefact, reused downstream.
#
# Uso:
#   sbatch scripts/vision/01_extract_raw.sh            # all models (sequential)
#   sbatch scripts/vision/01_extract_raw.sh dinov2     # one model (to parallelise)
#   sbatch scripts/vision/01_extract_raw.sh tipsv2 448 # one resolution ($2)
#
# $2 = single resolution (only for multi-resolution encoders, see resolutions_for in _common.sh): one job per resolution.

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
# Exclude Blackwell nodes (sm_120): floorplan-env PyTorch crashes on every forward. Same whitelist as 07_eval_head_partial.sh.
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
      python -m src.vision.run_retrieval \
          model.name=$MODEL model.variant=$VAR model.kwargs.pooling=$POOL $(res_flags "$RES")
    done
  done
done

echo ""
echo "=== completato: $(date) ==="
