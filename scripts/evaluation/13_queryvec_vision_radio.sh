#!/bin/bash
# Control "different models, same information": second vision encoder `radio/natural/whiten` (whitening fit
# on train), frozen: per-query files + damaged-query vectors (qvec/1) from the same forward. Twin of 08
# (same damage `nowalls_random` f = 0.0 0.25 0.5 0.75, same removed rooms), VALID only.
#
# Same keys as the radio run of the grid (05 with POOLS=natural, native resolution = no res_flags, preset
# batch), so the vectors are those of the grid. RADIO materialises O(N^2) attention; queries are encoded
# one at a time here anyway.
# Output folders (fixed, whitelisted):
#   results/perquery/fusion_branches_valid   (per-query of this job, used by C3)
#   results/queryvec/valid                   (qvec/1)
# Env PERQUERY_DIR / EXTRA / POOLS / TRANSFORMS / RES are ignored on purpose.
#
# Usage:
#   sbatch scripts/evaluation/13_queryvec_vision_radio.sh
#   FORCE=1 sbatch scripts/evaluation/13_queryvec_vision_radio.sh   # overwrite files already there
#
# Output (4 + 4 files):
#   results/perquery/fusion_branches_valid/vision_radio_natural_whiten-train_partial-nowalls-random-f<f>_valid.npz
#   results/queryvec/valid/vision_radio_natural_whiten-train_partial-nowalls-random-f<f>_valid.npz

#SBATCH --job-name="ev_13_queryvec_vision_radio"
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=all_usr_prod
#SBATCH --account=cvcs2026
# Excludes the Blackwell nodes (sm_120): same whitelist as 05.
#SBATCH --constraint="gpu_RTX5000_16G|gpu_2080_11G|gpu_2080Ti_11G|gpu_P100_16G|gpu_RTX_A5000_24G|gpu_A40_45G|gpu_L40S_45G"

set -uo pipefail
umask 002

source ~/floorplan-env/bin/activate
source /work/cvcs2026/ai_interior_design/CVCS-AI-Assisted-ID/scripts/vision/_common.sh
cd "$PROJECT_DIR" || exit 1

# valid only: no split argument
if [ $# -gt 0 ]; then
  echo "!! ERRORE: nessun argomento ammesso (solo valid, status.md §51), ricevuto '$*'" >&2
  exit 1
fi
SPLIT=valid

# Fixed config: not arguments, not env.
MODEL=radio
POOL=natural
TAG="vision_${MODEL}_${POOL}_whiten-train"
FRACTIONS="0.0 0.25 0.5 0.75"

# Fixed output folders + explicit whitelist.
PERQUERY_DIR="results/perquery/fusion_branches_${SPLIT}"
QVEC_DIR="results/queryvec/${SPLIT}"
for D in "$PERQUERY_DIR" "$QVEC_DIR"; do
  case "$D" in
    results/perquery/fusion_branches_valid|results/queryvec/valid) ;;
    *) echo "!! ERRORE: cartella di output non ammessa: '$D' (solo fusion_branches_* e queryvec/*)" >&2
       exit 1 ;;
  esac
done
[ -n "${EXTRA:-}${POOLS:-}${TRANSFORMS:-}${RES:-}" ] && \
  echo "⚠️  EXTRA/POOLS/TRANSFORMS/RES impostate nell'ambiente: IGNORATE (config fissa di §51)"

# Refuse to overwrite a finished run by accident (FORCE=1 to redo it).
EXISTING=$(ls "$QVEC_DIR"/${TAG}_partial-nowalls-random-f*_"${SPLIT}".npz \
              "$PERQUERY_DIR"/${TAG}_partial-nowalls-random-f*_"${SPLIT}".npz 2>/dev/null | wc -l)
if [ "$EXISTING" -gt 0 ] && [ "${FORCE:-0}" != "1" ]; then
  echo "!! ERRORE: $EXISTING file qvec/per-query gia' presenti per $TAG: rilancia con FORCE=1 se voluto" >&2
  exit 1
fi
mkdir -p logs "$PERQUERY_DIR" "$QVEC_DIR"

echo "=== $(date) | CONTROL §51 — radio query vectors | split $SPLIT | $TAG ==="
echo "per-query -> $PERQUERY_DIR | qvec -> $QVEC_DIR | nowalls_random f = $FRACTIONS"
nvidia-smi

START_MARK=$(mktemp)
# shellcheck disable=SC2046
python -m src.vision.evaluation.evaluate \
  model.name=$MODEL model.variant=$(variant_for "$POOL" native) model.kwargs.pooling=$POOL \
  head.enabled=false whitening.enabled=true \
  partial.enabled=true \
  partial.strategies.random.enabled=false \
  partial.strategies.semantic.enabled=false partial.strategies.topology.enabled=false \
  partial.strategies.crop.enabled=false partial.strategies.patch.enabled=false \
  partial.strategies.nowalls_random.enabled=true \
  "partial.strategies.nowalls_random.fractions=[0.0,0.25,0.5,0.75]" \
  partial.strategies.nowalls_semantic.enabled=false partial.strategies.nowalls_topology.enabled=false \
  eval.split="$SPLIT" eval.perquery_dir="$PERQUERY_DIR" eval.query_vectors_dir="$QVEC_DIR"
RC=$?

# every expected file must exist and be new (the exit code alone is not trusted)
MISSING=0
for F in $FRACTIONS; do
  for P in "$PERQUERY_DIR/${TAG}_partial-nowalls-random-f${F}_${SPLIT}.npz" \
           "$QVEC_DIR/${TAG}_partial-nowalls-random-f${F}_${SPLIT}.npz"; do
    if [ ! -f "$P" ] || [ ! "$P" -nt "$START_MARK" ]; then
      echo "!! MANCANTE o vecchio: $P" >&2
      MISSING=$((MISSING + 1))
    fi
  done
done
rm -f "$START_MARK"

echo ""
echo "=== completato: $(date) | rc=$RC | file mancanti: $MISSING ==="
ls -la "$QVEC_DIR"
[ $RC -eq 0 ] && [ $MISSING -eq 0 ] || exit 1
echo "PROSSIMO PASSO: sbatch scripts/evaluation/14_late_fusion_visionvision.sh"
