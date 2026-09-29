#!/bin/bash
# scripts/evaluation/08_queryvec_vision.sh
# LATE FUSION (status.md §49) — vision branch: per-query files + vectors of the
# damaged queries (contract qvec/1, src/evaluation/query_vectors.py), from the
# SAME forward. Config fixed by the pre-registration: pespatial / gem / whiten
# (whitening fit on train), frozen, damage `nowalls_random` f = 0.0 0.25 0.5 0.75.
#
# Why a new script (not 05 with env): the output folders are FIXED here and not
# overridable. 05 without PERQUERY_DIR writes into results/perquery/vision_partial_valid
# (historical files); this script can only write into
#   results/perquery/fusion_branches_<split>   (per-query of this job, used by C3)
#   results/queryvec/<split>                   (qvec/1)
# Env PERQUERY_DIR / EXTRA / POOLS / TRANSFORMS are IGNORED on purpose.
#
# Use:
#   sbatch scripts/evaluation/08_queryvec_vision.sh valid
#   sbatch scripts/evaluation/08_queryvec_vision.sh test      # ONLY in the pre-registered test group
#   FORCE=1 sbatch ...                                        # overwrite files already there
#
# Output (4 + 4 files):
#   results/perquery/fusion_branches_<split>/vision_pespatial_gem_whiten-train_partial-nowalls-random-f<f>_<split>.npz
#   results/queryvec/<split>/vision_pespatial_gem_whiten-train_partial-nowalls-random-f<f>_<split>.npz

#SBATCH --job-name="ev_08_queryvec_vision"
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

SPLIT="${1:-}"
case "$SPLIT" in
  valid|test) ;;
  *) echo "!! ERRORE: serve lo split come primo argomento (valid | test), ricevuto '$SPLIT'" >&2
     exit 1 ;;
esac

# Fixed config of the pre-registration (§49): not arguments, not env.
MODEL=pespatial
POOL=gem
TAG="vision_${MODEL}_${POOL}_whiten-train"
FRACTIONS="0.0 0.25 0.5 0.75"

# Fixed output folders + explicit whitelist (belt and braces against edits).
PERQUERY_DIR="results/perquery/fusion_branches_${SPLIT}"
QVEC_DIR="results/queryvec/${SPLIT}"
for D in "$PERQUERY_DIR" "$QVEC_DIR"; do
  case "$D" in
    results/perquery/fusion_branches_valid|results/perquery/fusion_branches_test|results/queryvec/valid|results/queryvec/test) ;;
    *) echo "!! ERRORE: cartella di output non ammessa: '$D' (solo fusion_branches_* e queryvec/*)" >&2
       exit 1 ;;
  esac
done
[ -n "${EXTRA:-}${POOLS:-}${TRANSFORMS:-}" ] && \
  echo "⚠️  EXTRA/POOLS/TRANSFORMS impostate nell'ambiente: IGNORATE (config fissa di §49)"

# Refuse to overwrite a finished run by accident (FORCE=1 to redo it).
EXISTING=$(ls "$QVEC_DIR"/${TAG}_partial-nowalls-random-f*_"${SPLIT}".npz 2>/dev/null | wc -l)
if [ "$EXISTING" -gt 0 ] && [ "${FORCE:-0}" != "1" ]; then
  echo "!! ERRORE: $EXISTING file qvec gia' presenti in $QVEC_DIR per $TAG: rilancia con FORCE=1 se voluto" >&2
  exit 1
fi
mkdir -p logs "$PERQUERY_DIR" "$QVEC_DIR"

echo "=== $(date) | LATE FUSION — vision query vectors | split $SPLIT | $TAG ==="
echo "per-query -> $PERQUERY_DIR | qvec -> $QVEC_DIR | nowalls_random f = $FRACTIONS"
[ "$SPLIT" = "test" ] && echo "⚠️  TEST: solo nel gruppo di job pre-registrato, con alpha* gia' fissato sul valid."
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

# The exit code alone is not trusted: every expected file must exist and be new.
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
echo "PROSSIMO PASSO: scripts/evaluation/09_queryvec_graph.sh $SPLIT (poi 10_late_fusion.sh $SPLIT)"
