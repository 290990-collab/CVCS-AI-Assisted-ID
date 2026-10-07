#!/bin/bash
# Secondary fusion with the vision head: vision branch of ONE replica, per-query files + vectors of the
# damaged queries (qvec/1) of `pespatial/gem` + head `head_nowalls_conv.pt` + whitening fit on train after the
# head (= head+whiten-train), damage `nowalls_random` f = 0.0 0.25 0.5 0.75 with damage seed S (= the
# replica's seed). Same command as scripts/evaluation/08_queryvec_vision.sh, plus the head; crop/patch off.
#
# Writes (new) results/final_pipeline/vision_head/{perquery,queryvec}/s<S>/<split>/
# (paths from `python -m src.evaluation.fusion_head paths`). Refuses if any output exists.
#
# Usage:
#   sbatch scripts/final_pipeline/06_queryvec_vision_head.sh valid 42   # then 100042 200042 300042
#   sbatch scripts/final_pipeline/06_queryvec_vision_head.sh test 42    # only after the test pre-registration (TEST_PREREGISTERED)
# Then: sbatch scripts/final_pipeline/07_fusion_head.sh valid <S>

#SBATCH --job-name="rg_06_vision_head"
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
# Excludes the Blackwell nodes (sm_120): same whitelist as 08.
#SBATCH --constraint="gpu_RTX5000_16G|gpu_2080_11G|gpu_2080Ti_11G|gpu_P100_16G|gpu_RTX_A5000_24G|gpu_A40_45G|gpu_L40S_45G"

set -uo pipefail
umask 002

source ~/floorplan-env/bin/activate
source /work/cvcs2026/ai_interior_design/CVCS-AI-Assisted-ID/scripts/vision/_common.sh
cd "$PROJECT_DIR" || exit 1
mkdir -p logs

SPLIT="${1:-}"; S="${2:-}"
case "$SPLIT" in
  valid) ;;
  test)  # only after the approved test pre-registration
    [ -f results/final_pipeline/TEST_PREREGISTERED ] || { echo "!! test bloccato: manca results/final_pipeline/TEST_PREREGISTERED (status.md §62)" >&2; exit 1; } ;;
  *) echo "!! split valid|test, ricevuto '$SPLIT'" >&2; exit 1 ;;
esac
[[ "$S" =~ ^[0-9]+$ ]] || { echo "!! seed non valido: '$S'" >&2; exit 1; }

eval "$(python -m src.evaluation.fusion_head paths --seed "$S" --split "$SPLIT")" || exit 1
TAG="vision_pespatial_gem_head-nowalls-conv+whiten-train"
HEAD="embeddings/vision/pespatial/gem/head_nowalls_conv.pt"
HEAD_SHA1="49a7405949ac9fe9abf1feb49c2dbbce6a3de60c"
FRACTIONS="0.0 0.25 0.5 0.75"

[ "$(sha1sum "$HEAD" | cut -d' ' -f1)" = "$HEAD_SHA1" ] || { echo "!! sha1 della head diverso da $HEAD_SHA1" >&2; exit 1; }
for D in "$VH_PQ_DIR" "$VH_QV_DIR"; do
  if [ -n "$(ls -A "$D" 2>/dev/null)" ]; then echo "!! $D non vuota: niente sovrascritture" >&2; exit 1; fi
done
mkdir -p "$VH_PQ_DIR" "$VH_QV_DIR"

echo "=== $(date) | vision CON HEAD | s$S | $SPLIT | $TAG ==="
echo "per-query -> $VH_PQ_DIR | qvec -> $VH_QV_DIR | nowalls_random f = $FRACTIONS | partial.seed=$S"
nvidia-smi

START_MARK=$(mktemp)
python -m src.vision.evaluation.evaluate \
  model.name=pespatial model.variant=$(variant_for gem native) model.kwargs.pooling=gem \
  head.enabled=true head.file=head_nowalls_conv.pt whitening.enabled=true whitening.fit_split=train \
  partial.enabled=true \
  partial.strategies.random.enabled=false \
  partial.strategies.semantic.enabled=false partial.strategies.topology.enabled=false \
  partial.strategies.crop.enabled=false partial.strategies.patch.enabled=false \
  partial.strategies.nowalls_random.enabled=true \
  "partial.strategies.nowalls_random.fractions=[0.0,0.25,0.5,0.75]" \
  partial.strategies.nowalls_semantic.enabled=false partial.strategies.nowalls_topology.enabled=false \
  eval.split="$SPLIT" eval.perquery_dir="$VH_PQ_DIR" eval.query_vectors_dir="$VH_QV_DIR" \
  "partial.seed=$S"
RC=$?

MISSING=0
for F in $FRACTIONS; do
  for P in "$VH_PQ_DIR/${TAG}_partial-nowalls-random-f${F}_${SPLIT}.npz" \
           "$VH_QV_DIR/${TAG}_partial-nowalls-random-f${F}_${SPLIT}.npz"; do
    if [ ! -f "$P" ] || [ ! "$P" -nt "$START_MARK" ]; then echo "!! MANCANTE o vecchio: $P" >&2; MISSING=$((MISSING + 1)); fi
  done
done
rm -f "$START_MARK"
[ "$(sha1sum "$HEAD" | cut -d' ' -f1)" = "$HEAD_SHA1" ] || { echo "!! la head e' cambiata durante il job" >&2; MISSING=$((MISSING + 1)); }

echo "=== completato: $(date) | rc=$RC | file mancanti: $MISSING ==="
[ $RC -eq 0 ] && [ $MISSING -eq 0 ] || exit 1
echo "PROSSIMO PASSO: sbatch scripts/final_pipeline/07_fusion_head.sh $SPLIT $S"
