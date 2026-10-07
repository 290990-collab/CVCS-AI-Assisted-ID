#!/bin/bash
# Stage C: cache positive pairs for each (model x valid pooling).
# Per train+valid plan: V degraded views (+aug) encoded with the frozen backbone -> embeddings/vision/<model>/<pooling>/pairs.npz.
# Needs stage A. Heavy (V * 57k render+forward per config): run per model.
#
# Uso:
#   sbatch scripts/vision/02_build_pairs.sh dinov2     # recommended: one model per job
#   sbatch scripts/vision/02_build_pairs.sh            # all models (long)
#
# $2 = single resolution (only for multi-resolution encoders, see resolutions_for in _common.sh): one job per resolution.
# env POOLS = pooling (empty = all) · EXTRA = dotlist override:
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
# Exclude Blackwell nodes (sm_120): floorplan-env PyTorch crashes on every forward. Same whitelist as 07_eval_head_partial.sh.
#SBATCH --constraint="gpu_RTX5000_16G|gpu_2080_11G|gpu_2080Ti_11G|gpu_P100_16G|gpu_RTX_A5000_24G|gpu_A40_45G|gpu_L40S_45G"

source ~/floorplan-env/bin/activate
source /work/cvcs2026/ai_interior_design/CVCS-AI-Assisted-ID/scripts/vision/_common.sh
cd "$PROJECT_DIR"
mkdir -p logs

POOLS="${POOLS:-}"          # empty = all poolings of the encoder (e.g. POOLS=gem)
EXTRA="${EXTRA:-}"          # extra dotlist overrides (e.g. EXTRA="training.damage=nowalls_random")

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
