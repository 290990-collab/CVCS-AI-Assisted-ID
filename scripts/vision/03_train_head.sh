#!/bin/bash
# Stage D: train the projection head for each (model x valid pooling).
# Trains on pairs.npz (cached vectors, frozen backbone) -> head.pt in the same folder. Needs stage C; fast (MLP only).
#
# Uso:
#   sbatch scripts/vision/03_train_head.sh             # all models
#   sbatch scripts/vision/03_train_head.sh dinov2      # one model
#
# $2 = single resolution (only for multi-resolution encoders, see resolutions_for in _common.sh): one job per resolution.

#SBATCH --job-name="vx_03_train_head"
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --partition=all_usr_prod
#SBATCH --account=cvcs2026
# Exclude Blackwell nodes (sm_120): floorplan-env PyTorch crashes on every forward. Same whitelist as 07_eval_head_partial.sh.
#SBATCH --constraint="gpu_RTX5000_16G|gpu_2080_11G|gpu_2080Ti_11G|gpu_P100_16G|gpu_RTX_A5000_24G|gpu_A40_45G|gpu_L40S_45G"

source ~/floorplan-env/bin/activate
source /work/cvcs2026/ai_interior_design/CVCS-AI-Assisted-ID/scripts/vision/_common.sh
cd "$PROJECT_DIR"
mkdir -p logs

echo "=== $(date) | STAGE D training head | modelli: $(select_models "$1") ==="
nvidia-smi

for MODEL in $(select_models "$1"); do
  for POOL in $(poolings_for "$MODEL"); do
    for RES in $(select_resolutions "$MODEL" "${2:-}"); do
      VAR=$(variant_for "$POOL" "$RES")
      RFLAGS=$(res_flags "$RES")
      echo ""
      echo "########  TRAIN HEAD: $(label_for "$MODEL" "$POOL" "$RES")  ########"
      python -m src.vision.training.train_projection \
          model.name=$MODEL model.variant=$VAR model.kwargs.pooling=$POOL $RFLAGS
    done
  done
done

echo ""
echo "=== completato: $(date) ==="
