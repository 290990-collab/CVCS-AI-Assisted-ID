#!/bin/bash
# Late fusion, CPU job: fuses the vectors of 08 (vision) and 09 (graph) for every alpha and writes per-query
# files (perquery/1) for the damaged plans (f = 0.0 0.25 0.5 0.75) and the full plan. No GPU.
#
# Inputs and output are fixed (no env overrides):
#   results/queryvec/<split>/vision_pespatial_gem_whiten-train_*   (08)
#   results/queryvec/<split>/graph_gat_asymrob_*                   (09)
#   gallery vectors: the paths pinned (sha1) in the qvec meta
#   -> results/perquery/fusion_<split>/fusion_a<alpha>_{partial-nowalls-random-f<f>,full}_<split>.npz
#
# Alphas: valid = grid {0, 0.1, ..., 1}; test = only {0, alpha*, 1} from the valid select json
# (ALPHAS_FROM is mandatory on the test).
#
# Usage:
#   sbatch scripts/evaluation/10_late_fusion.sh valid
#   ALPHAS_FROM=results/fusion/select_valid.json sbatch scripts/evaluation/10_late_fusion.sh test
#   FORCE=1 ...                                  # overwrite fused files already there
#
# Seed mode (optional $2 = S, digits only; multi-seed replica): inputs from 08/09 run with the same S,
# output only under seeds/s<S>/:
#   results/queryvec/seeds/s<S>/<split>/vision_pespatial_gem_whiten-train_*
#   results/queryvec/seeds/s<S>/<split>/graph_gat_asymrob_s<S>_*
#   -> results/perquery/seeds/s<S>/fusion_<split>/
# On the test ALPHAS_FROM is derived = results/fusion/seeds/s<S>/select_valid.json
# (an ALPHAS_FROM in the env pointing elsewhere is an error).
#   sbatch scripts/evaluation/10_late_fusion.sh valid 100042
#   sbatch scripts/evaluation/10_late_fusion.sh test 100042
# Without $2 the command line and the folders are the historical ones.

#SBATCH --job-name="ev_10_late_fusion"
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=all_usr_prod
#SBATCH --account=cvcs2026

set -uo pipefail
umask 002

PROJECT_DIR="/work/cvcs2026/ai_interior_design/CVCS-AI-Assisted-ID"
source ~/floorplan-env/bin/activate
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

ALPHAS_FROM="${ALPHAS_FROM:-}"
if [ -z "$SEED" ]; then
  VISION_QVEC="results/queryvec/${SPLIT}/vision_pespatial_gem_whiten-train"
  GRAPH_QVEC="results/queryvec/${SPLIT}/graph_gat_asymrob"
  OUT="results/perquery/fusion_${SPLIT}"
  case "$OUT" in
    results/perquery/fusion_valid|results/perquery/fusion_test) ;;
    *) echo "!! ERRORE: cartella di output non ammessa: '$OUT'" >&2; exit 1 ;;
  esac
else
  VISION_QVEC="results/queryvec/seeds/s${SEED}/${SPLIT}/vision_pespatial_gem_whiten-train"
  GRAPH_QVEC="results/queryvec/seeds/s${SEED}/${SPLIT}/graph_gat_asymrob_s${SEED}"
  OUT="results/perquery/seeds/s${SEED}/fusion_${SPLIT}"
  case "$OUT" in
    "results/perquery/seeds/s${SEED}/fusion_${SPLIT}") ;;
    *) echo "!! ERRORE: cartella di output non ammessa: '$OUT'" >&2; exit 1 ;;
  esac
  if [ "$SPLIT" = "test" ]; then
    SEL_VALID="results/fusion/seeds/s${SEED}/select_valid.json"
    if [ -n "$ALPHAS_FROM" ] && [ "$ALPHAS_FROM" != "$SEL_VALID" ]; then
      echo "!! ERRORE: in modalita' seed ALPHAS_FROM e' derivato ($SEL_VALID): togli ALPHAS_FROM='$ALPHAS_FROM' dall'ambiente" >&2
      exit 1
    fi
    ALPHAS_FROM="$SEL_VALID"
  fi
fi

ALPHA_FLAGS=()
if [ "$SPLIT" = "test" ]; then
  if [ -z "$ALPHAS_FROM" ] || [ ! -f "$ALPHAS_FROM" ]; then
    echo "!! ERRORE: sul test serve ALPHAS_FROM=<select json del valid> (esistente): il test non sceglie alpha" >&2
    exit 1
  fi
elif [ -n "$ALPHAS_FROM" ]; then
  echo "!! ERRORE: sul valid si usa la griglia intera di alpha: togli ALPHAS_FROM dall'ambiente" >&2
  exit 1
fi
if [ -n "$ALPHAS_FROM" ]; then
  [ -f "$ALPHAS_FROM" ] || { echo "!! ERRORE: ALPHAS_FROM non trovato: $ALPHAS_FROM" >&2; exit 1; }
  ALPHA_FLAGS=(--alphas-from "$ALPHAS_FROM")
fi

EXISTING=$(ls "$OUT"/fusion_a*_"${SPLIT}".npz 2>/dev/null | wc -l)
if [ "$EXISTING" -gt 0 ] && [ "${FORCE:-0}" != "1" ]; then
  echo "!! ERRORE: $EXISTING file fusi gia' presenti in $OUT: rilancia con FORCE=1 se voluto" >&2
  exit 1
fi
mkdir -p logs "$OUT"

# FAISS/BLAS threads = the CPUs of the job.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "=== $(date) | LATE FUSION | split $SPLIT | out $OUT | alpha: ${ALPHAS_FROM:-griglia 0..1 passo 0.1} ==="
python -m src.evaluation.late_fusion run --split "$SPLIT" \
  --gallery-names results/shared_gallery.json \
  --vision-qvec "$VISION_QVEC" --graph-qvec "$GRAPH_QVEC" \
  --fractions 0.0 0.25 0.5 0.75 \
  "${ALPHA_FLAGS[@]}" \
  --out "$OUT"
RC=$?

echo ""
echo "=== completato: $(date) | rc=$RC ==="
ls "$OUT" | wc -l
[ $RC -eq 0 ] || exit 1
echo ""
if [ -n "$SEED" ]; then
  echo "PROSSIMO PASSO (CPU, secondi): PRIMA i controlli, poi — solo se PASS — la scelta di alpha:"
  echo "  bash scripts/evaluation/15_seed_fusion_select.sh check $SPLIT $SEED"
  echo "  bash scripts/evaluation/15_seed_fusion_select.sh select $SPLIT $SEED"
  exit 0
fi
echo "PROSSIMO PASSO (CPU, secondi): PRIMA i controlli, poi — solo se tutti PASS — la scelta di alpha:"
echo "  python -m src.evaluation.fusion_select check --split $SPLIT --fusion-dir $OUT \\"
echo "      --vision-qvec $VISION_QVEC --graph-qvec $GRAPH_QVEC \\"
echo "      --vision-perquery results/perquery/fusion_branches_${SPLIT}/vision_pespatial_gem_whiten-train \\"
if [ "$SPLIT" = "valid" ]; then
  echo "      --graph-perquery  results/perquery/fusion_branches_${SPLIT}/graph_gat_asymrob"
  echo "  python -m src.evaluation.fusion_select select --split valid --fusion-dir $OUT"
else
  echo "      --graph-perquery  results/perquery/fusion_branches_${SPLIT}/graph_gat_asymrob \\"
  echo "      --alphas 0 <alpha* da $ALPHAS_FROM> 1"
  echo "  python -m src.evaluation.fusion_select select --split test --fusion-dir $OUT --fixed-alpha-from $ALPHAS_FROM"
fi
