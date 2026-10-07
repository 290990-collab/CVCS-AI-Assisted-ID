#!/bin/bash
# Head v2, vision branch of ONE replica for the fusion with W: per-query + vectors of the damaged queries
# (qvec/1) of pespatial/gem + whitening (checkpoint's) + head v2 on the query, rooms removed without walls,
# f = 0.0 0.25 0.5 0.75, damage seed S (= the replica's). Same command as 06_queryvec_vision_head.sh, head v2 instead.
# Writes (new) results/final_pipeline/vision_head_v2/{perquery,queryvec}/s<S>/<split>/ (refuses if any output exists).
#
# Usage:
#   sbatch scripts/final_pipeline/15_queryvec_vision_head_v2.sh valid 42      # then 100042 200042 300042
# Then: sbatch scripts/final_pipeline/16_fusion_head_v2.sh valid <S>

#SBATCH --job-name="rg_15_vision_head_v2"
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
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

SPLIT="${1:-}"; S="${2:-}"
case "$SPLIT" in valid|test) ;; *) echo "!! split valid|test, ricevuto '$SPLIT'" >&2; exit 1 ;; esac
[[ "$S" =~ ^[0-9]+$ ]] || { echo "!! seed non valido: '$S'" >&2; exit 1; }
python -m src.evaluation.multiseed_runs gate --split "$SPLIT" || exit 1
[ -f embeddings/vision/pespatial/gem/head_v2.pt ] || { echo "!! manca head_v2.pt (11_train_head_v2.sh)" >&2; exit 1; }
eval "$(python -m src.evaluation.fusion_head_v2 paths --seed "$S" --split "$SPLIT")" || exit 1
TAG="vision_pespatial_gem_whiten-train+qhead-v2"
FRACTIONS="0.0 0.25 0.5 0.75"
for D in "$VH_PQ_DIR" "$VH_QV_DIR"; do
  if [ -n "$(ls -A "$D" 2>/dev/null)" ]; then echo "!! $D non vuota: niente sovrascritture" >&2; exit 1; fi
done
mkdir -p "$VH_PQ_DIR" "$VH_QV_DIR"
echo "=== $(date) | vision CON HEAD v2 | s$S | $SPLIT | per-query -> $VH_PQ_DIR | qvec -> $VH_QV_DIR ==="
nvidia-smi
START_MARK=$(mktemp)
python -m src.vision.evaluation.evaluate \
  model.name=pespatial model.variant=$(variant_for gem native) model.kwargs.pooling=gem \
  head.enabled=false whitening.enabled=true whitening.fit_split=train query_head.enabled=true query_head.file=head_v2.pt \
  partial.enabled=true partial.strategies.random.enabled=false \
  partial.strategies.semantic.enabled=false partial.strategies.topology.enabled=false \
  partial.strategies.crop.enabled=false partial.strategies.patch.enabled=false \
  partial.strategies.nowalls_random.enabled=true "partial.strategies.nowalls_random.fractions=[0.0,0.25,0.5,0.75]" \
  partial.strategies.nowalls_semantic.enabled=false partial.strategies.nowalls_topology.enabled=false \
  eval.split="$SPLIT" eval.perquery_dir="$VH_PQ_DIR" eval.query_vectors_dir="$VH_QV_DIR" "partial.seed=$S"
RC=$?
MISSING=0
for F in $FRACTIONS; do
  for P in "$VH_PQ_DIR/${TAG}_partial-nowalls-random-f${F}_${SPLIT}.npz" "$VH_QV_DIR/${TAG}_partial-nowalls-random-f${F}_${SPLIT}.npz"; do
    [ -f "$P" ] && [ "$P" -nt "$START_MARK" ] || { echo "!! MANCANTE o vecchio: $P" >&2; MISSING=$((MISSING + 1)); }
  done
done
rm -f "$START_MARK"
echo "=== completato: $(date) | rc=$RC | file mancanti: $MISSING ==="
[ $RC -eq 0 ] && [ $MISSING -eq 0 ] || exit 1
