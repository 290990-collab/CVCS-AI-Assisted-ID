#!/bin/bash
# Retrain the head with epoch selection on the PARTIAL retrieval probe.
#
# Two steps per encoder/pooling (frozen backbone):
#   1. retrieval_probe: degrade the probe queries (valid, disjoint from the eval queries) and save their RAW features -> probe_partial.npz
#   2. train_projection with training.selection=probe_partial: pick the epoch on probe AUC instead of InfoNCE val-loss.
#      The checkpoint goes to head_probe.pt; head.pt is untouched.
#
# Probe gallery = eval.gallery_names of the config (shared). Then evaluate:
#   PERQUERY_DIR=results/perquery/vision_partial_valid_B EXTRA="head.file=head_probe.pt" \
#     sbatch scripts/evaluation/05_perquery_vision_partial_valid.sh <enc> head
#
# Uso:
#   sbatch scripts/vision/09_train_head_probe.sh dinov3
#   sbatch scripts/vision/09_train_head_probe.sh            # the 5 encoders with pairs.npz
#   EPOCHS=200 HEAD_FILE=head_probe_e200.pt REUSE_PROBE=1 \
#     sbatch scripts/vision/09_train_head_probe.sh dinov3 natural   # epoch-cap diagnostic
#     $2 = pooling (empty = all) · EPOCHS = training.epochs (empty = config)
#     HEAD_FILE = checkpoint (default head_probe.pt) · REUSE_PROBE=1 reuses probe_partial.npz
#     PATIENCE = training.patience (0 = no early stop: best over all epochs)
#     EXTRA = dotlist overrides for probe and training, e.g.
#       EPOCHS=5000 PATIENCE=100 HEAD_FILE=head_nowalls_conv.pt EXTRA="training.damage=nowalls_random" \
#         sbatch scripts/vision/09_train_head_probe.sh pespatial gem   # pairs of the same damage; no REUSE_PROBE
#
# Needs pairs.npz (stage 02): exists only for dinov2, dinov3, siglip2, radio, ijepa.

#SBATCH --job-name="vx_09_train_head_probe"
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --partition=all_usr_prod
#SBATCH --account=cvcs2026
#SBATCH --constraint="gpu_RTX5000_16G|gpu_2080_11G|gpu_2080Ti_11G|gpu_P100_16G|gpu_RTX_A5000_24G|gpu_A40_45G|gpu_L40S_45G"

set -uo pipefail
umask 002

source ~/floorplan-env/bin/activate
source /work/cvcs2026/ai_interior_design/CVCS-AI-Assisted-ID/scripts/vision/_common.sh
cd "$PROJECT_DIR" || exit 1
mkdir -p logs

MODELS="${1:-dinov2 dinov3 siglip2 radio ijepa}"
POOL_ARG="${2:-}"
HEAD_FILE="${HEAD_FILE:-head_probe.pt}"
EXTRA="${EXTRA:-}"          # extra dotlist overrides (e.g. EXTRA="training.damage=nowalls_random")
EPOCH_FLAGS=""
[ -n "${EPOCHS:-}" ] && EPOCH_FLAGS="training.epochs=$EPOCHS"
[ -n "${PATIENCE:-}" ] && EPOCH_FLAGS="$EPOCH_FLAGS training.patience=$PATIENCE"
export PYTHONUNBUFFERED=1

echo "=== $(date) | B.6 head con probe partial | modelli: $MODELS | pooling: ${POOL_ARG:-tutti} | head: $HEAD_FILE | ${EPOCH_FLAGS:-epoche dal config} | reuse probe: ${REUSE_PROBE:-0} ==="
grep -n "gallery_names" configs/vision_retrieval.yaml | head -1
nvidia-smi

FAILED=0
for MODEL in $MODELS; do
  for POOL in ${POOL_ARG:-$(poolings_for "$MODEL")}; do
    for RES in $(select_resolutions "$MODEL" ""); do
      VAR=$(variant_for "$POOL" "$RES")
      LBL=$(label_for "$MODEL" "$POOL" "$RES")
      COMMON="model.name=$MODEL model.variant=$VAR model.kwargs.pooling=$POOL $(res_flags "$RES") $EXTRA"
      if [ ! -f "embeddings/vision/$MODEL/$VAR/pairs.npz" ]; then
        echo "!! pairs.npz mancante per $LBL — salto"; continue
      fi
      echo ""
      echo "########  B.6 PROBE: $LBL  ########"
      if [ "${REUSE_PROBE:-0}" = "1" ] && [ -f "embeddings/vision/$MODEL/$VAR/probe_partial.npz" ]; then
        echo "probe gia' costruita: riuso embeddings/vision/$MODEL/$VAR/probe_partial.npz"
      else
        python -m src.vision.training.retrieval_probe $COMMON \
          || { echo "!! FALLITA probe: $LBL" >&2; FAILED=$((FAILED + 1)); continue; }
      fi
      echo "########  B.6 TRAIN: $LBL  ########"
      python -m src.vision.training.train_projection $COMMON \
          training.selection=probe_partial head.file=$HEAD_FILE $EPOCH_FLAGS \
        || { echo "!! FALLITO training: $LBL" >&2; FAILED=$((FAILED + 1)); }
    done
  done
done

echo ""
echo "=== completato: $(date) | fallimenti: $FAILED ==="
[ $FAILED -eq 0 ] || exit 1
