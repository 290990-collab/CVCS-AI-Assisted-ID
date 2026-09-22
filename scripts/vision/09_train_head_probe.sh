#!/bin/bash
# scripts/vision/09_train_head_probe.sh
# FASE B.6 — head riallenata con selezione sulla probe di retrieval PARTIAL.
#
# Due passi per ogni encoder/pooling (backbone frozen):
#   1. retrieval_probe: degrada le query della probe (valid, DISGIUNTE dalle
#      query di valutazione) e ne salva le feature RAW -> probe_partial.npz
#   2. train_projection con training.selection=probe_partial: sceglie l'epoca
#      sull'AUC della probe (criterio A.5) invece che sulla val-loss InfoNCE.
#      Il checkpoint va in head_probe.pt: head.pt (storico, § 24) NON si tocca.
#
# Gallery della probe = eval.gallery_names del config (condivisa, B.3).
# Poi la valutazione, sul protocollo B:
#   PERQUERY_DIR=results/perquery/vision_partial_valid_B EXTRA="head.file=head_probe.pt" \
#     sbatch scripts/evaluation/05_perquery_vision_partial_valid.sh <enc> head
#
# Uso:
#   sbatch scripts/vision/09_train_head_probe.sh dinov3
#   sbatch scripts/vision/09_train_head_probe.sh            # i 5 encoder con pairs.npz
#   EPOCHS=200 HEAD_FILE=head_probe_e200.pt REUSE_PROBE=1 \
#     sbatch scripts/vision/09_train_head_probe.sh dinov3 natural   # diagnostico tetto epoche
#     $2 = pooling (vuoto = tutti) · EPOCHS = training.epochs (vuoto = config)
#     HEAD_FILE = checkpoint (default head_probe.pt) · REUSE_PROBE=1 riusa probe_partial.npz
#     PATIENCE = training.patience (0 = nessun early stop: si sceglie il massimo su tutte le epoche)
#     EXTRA = override dotlist per probe e training (15 set, status.md §46), es.
#       EPOCHS=5000 PATIENCE=100 HEAD_FILE=head_nowalls_conv.pt EXTRA="training.damage=nowalls_random" \
#         sbatch scripts/vision/09_train_head_probe.sh pespatial gem   # coppie dello stesso danno; niente REUSE_PROBE
#
# ⚠️ Serve pairs.npz (STAGE 02): esiste solo per dinov2, dinov3, siglip2, radio, ijepa.

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
EXTRA="${EXTRA:-}"          # override dotlist per probe E training (es. EXTRA="training.damage=nowalls_random")
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
