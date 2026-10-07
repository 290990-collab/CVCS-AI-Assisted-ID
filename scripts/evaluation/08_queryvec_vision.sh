#!/bin/bash
# Late fusion, vision branch: per-query files + vectors of the damaged queries (contract qvec/1,
# src/evaluation/query_vectors.py) from the same forward. Fixed config: pespatial / gem / whiten
# (whitening fit on train), frozen, damage `nowalls_random` f = 0.0 0.25 0.5 0.75.
#
# Separate from 05 because the output folders are fixed here and not overridable (05 without PERQUERY_DIR
# writes into results/perquery/vision_partial_valid, historical files):
#   results/perquery/fusion_branches_<split>   (per-query of this job, used by C3)
#   results/queryvec/<split>                   (qvec/1)
# Env PERQUERY_DIR / EXTRA / POOLS / TRANSFORMS are ignored on purpose.
#
# Usage:
#   sbatch scripts/evaluation/08_queryvec_vision.sh valid
#   sbatch scripts/evaluation/08_queryvec_vision.sh test      # only in the pre-registered test group
#   FORCE=1 sbatch ...                                        # overwrite files already there
#   sbatch scripts/evaluation/08_queryvec_vision.sh valid 100042   # multi-seed replica
#
# Output (4 + 4 files):
#   results/perquery/fusion_branches_<split>/vision_pespatial_gem_whiten-train_partial-nowalls-random-f<f>_<split>.npz
#   results/queryvec/<split>/vision_pespatial_gem_whiten-train_partial-nowalls-random-f<f>_<split>.npz
#
# Seed mode (optional $2 = S, digits only): damage seed `partial.seed=S` (query sample unchanged, eval.seed 42),
# crop + patch also on (per-query only: they remove no rooms, so no qvec), outputs only under seeds/s<S>/:
#   results/perquery/seeds/s<S>/fusion_branches_<split>/  (4 nowalls-random + 3 crop + 3 patch)
#   results/queryvec/seeds/s<S>/<split>/                  (4 nowalls-random)
# Without $2 the command line and the folders are the historical ones.

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
SEED="${2:-}"
if [ -n "$SEED" ] && ! [[ "$SEED" =~ ^[0-9]+$ ]]; then
  echo "!! ERRORE: il seed (secondo argomento) deve essere un intero, ricevuto '$SEED'" >&2
  exit 1
fi

# Fixed config: not arguments, not env.
MODEL=pespatial
POOL=gem
TAG="vision_${MODEL}_${POOL}_whiten-train"
FRACTIONS="0.0 0.25 0.5 0.75"
DAMAGE_FRACTIONS="0.25 0.5 0.75"   # crop/patch, seed mode only (configs/vision_retrieval.yaml)

# Fixed output folders + explicit whitelist.
# Damage switches: historical mode = crop/patch off, no partial.seed (YAML's 42).
CROP=false
PATCH=false
SEED_FLAGS=()
if [ -z "$SEED" ]; then
  PERQUERY_DIR="results/perquery/fusion_branches_${SPLIT}"
  QVEC_DIR="results/queryvec/${SPLIT}"
  for D in "$PERQUERY_DIR" "$QVEC_DIR"; do
    case "$D" in
      results/perquery/fusion_branches_valid|results/perquery/fusion_branches_test|results/queryvec/valid|results/queryvec/test) ;;
      *) echo "!! ERRORE: cartella di output non ammessa: '$D' (solo fusion_branches_* e queryvec/*)" >&2
         exit 1 ;;
    esac
  done
else
  PERQUERY_DIR="results/perquery/seeds/s${SEED}/fusion_branches_${SPLIT}"
  QVEC_DIR="results/queryvec/seeds/s${SEED}/${SPLIT}"
  for D in "$PERQUERY_DIR" "$QVEC_DIR"; do
    case "$D" in
      "results/perquery/seeds/s${SEED}/fusion_branches_${SPLIT}"|"results/queryvec/seeds/s${SEED}/${SPLIT}") ;;
      *) echo "!! ERRORE: cartella di output non ammessa: '$D' (solo seeds/s${SEED}/...)" >&2
         exit 1 ;;
    esac
  done
  CROP=true
  PATCH=true
  SEED_FLAGS=("partial.seed=$SEED")
fi
[ -n "${EXTRA:-}${POOLS:-}${TRANSFORMS:-}" ] && \
  echo "⚠️  EXTRA/POOLS/TRANSFORMS impostate nell'ambiente: IGNORATE (config fissa di §49)"

# Refuse to overwrite a finished run by accident (FORCE=1 to redo it).
EXISTING=$(ls "$QVEC_DIR"/${TAG}_partial-nowalls-random-f*_"${SPLIT}".npz 2>/dev/null | wc -l)
if [ -n "$SEED" ]; then
  EXISTING=$((EXISTING + $(ls "$PERQUERY_DIR"/${TAG}_partial-*_"${SPLIT}".npz 2>/dev/null | wc -l)))
fi
if [ "$EXISTING" -gt 0 ] && [ "${FORCE:-0}" != "1" ]; then
  echo "!! ERRORE: $EXISTING file qvec gia' presenti in $QVEC_DIR per $TAG: rilancia con FORCE=1 se voluto" >&2
  exit 1
fi
mkdir -p logs "$PERQUERY_DIR" "$QVEC_DIR"

echo "=== $(date) | LATE FUSION — vision query vectors | split $SPLIT | $TAG ==="
echo "per-query -> $PERQUERY_DIR | qvec -> $QVEC_DIR | nowalls_random f = $FRACTIONS"
[ -n "$SEED" ] && echo "SEED MODE: partial.seed=$SEED | crop=$CROP patch=$PATCH (f = $DAMAGE_FRACTIONS, solo per-query)"
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
  partial.strategies.crop.enabled=$CROP partial.strategies.patch.enabled=$PATCH \
  partial.strategies.nowalls_random.enabled=true \
  "partial.strategies.nowalls_random.fractions=[0.0,0.25,0.5,0.75]" \
  partial.strategies.nowalls_semantic.enabled=false partial.strategies.nowalls_topology.enabled=false \
  eval.split="$SPLIT" eval.perquery_dir="$PERQUERY_DIR" eval.query_vectors_dir="$QVEC_DIR" \
  "${SEED_FLAGS[@]}"
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
# Seed mode: crop/patch per-query files too (needed by `robustness_auc rank --robust`).
if [ -n "$SEED" ]; then
  for S in crop patch; do
    for F in $DAMAGE_FRACTIONS; do
      P="$PERQUERY_DIR/${TAG}_partial-${S}-f${F}_${SPLIT}.npz"
      if [ ! -f "$P" ] || [ ! "$P" -nt "$START_MARK" ]; then
        echo "!! MANCANTE o vecchio: $P" >&2
        MISSING=$((MISSING + 1))
      fi
    done
  done
fi
rm -f "$START_MARK"

echo ""
echo "=== completato: $(date) | rc=$RC | file mancanti: $MISSING ==="
ls -la "$QVEC_DIR"
[ $RC -eq 0 ] && [ $MISSING -eq 0 ] || exit 1
echo "PROSSIMO PASSO: scripts/evaluation/09_queryvec_graph.sh $SPLIT${SEED:+ $SEED} (poi 10_late_fusion.sh $SPLIT${SEED:+ $SEED})"
