#!/bin/bash
# Head v2: fusion W + vision with head v2, ONE replica. CPU only.
# Same fusion code as 07_fusion_head.sh (`late_fusion.run`, alpha grid 0..1 step 0.1), run through
# `src.evaluation.fusion_head_v2 run`: head v2 is applied to the vision query vectors after the whitening, the
# vision gallery stays frozen; first it checks head(whiten(raw)) = the vision job's final query vectors (<= 1e-4).
# W = the replica's multi-seed evaluation (own gallery, round2.graph_source).
# Writes (new) results/final_pipeline/fusion_head_v2/s<S>/fusion_<split>/ + head_v2_fusion_<split>.json.
#
# Usage:
#   sbatch scripts/final_pipeline/16_fusion_head_v2.sh valid 42        # then 100042 200042 300042
# Then: bash scripts/final_pipeline/17_fusion_head_v2_select.sh check valid <S>

#SBATCH --job-name="rg_16_fusion_head_v2"
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=06:00:00
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
mkdir -p logs

SPLIT="${1:-}"; S="${2:-}"
case "$SPLIT" in valid|test) ;; *) echo "!! split '$SPLIT' (valid | test)" >&2; exit 1 ;; esac
[[ "$S" =~ ^[0-9]+$ ]] || { echo "!! seed non valido: '$S'" >&2; exit 1; }
run() { if [ "${RG_DRY_RUN:-0}" = "1" ]; then echo "+ $*"; else "$@"; fi; }
python -m src.evaluation.multiseed_runs gate --split "$SPLIT" || exit 1

eval "$(python -m src.evaluation.fusion_head_v2 paths --seed "$S" --split "$SPLIT")" || exit 1
ALPHA_FLAGS=()
if [ "$SPLIT" = "test" ]; then
  [ -f "$SELECT_VALID_JSON" ] || { echo "!! sul test serve $SELECT_VALID_JSON" >&2; exit 1; }
  ALPHA_FLAGS=(--alphas-from "$SELECT_VALID_JSON")
fi
EXISTING=$(ls "$FUSION_DIR"/fusion_a*_"${SPLIT}".npz 2>/dev/null | wc -l)
if [ "$EXISTING" -gt 0 ] || [ -e "$HEAD_JSON" ]; then
  echo "!! ERRORE: file gia' presenti in $FUSION_DIR o $HEAD_JSON — niente sovrascritture" >&2; exit 1
fi
[ "${RG_DRY_RUN:-0}" = "1" ] || mkdir -p "$FUSION_DIR"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export CUDA_VISIBLE_DEVICES=""

echo "=== $(date) | fusione CON HEAD v2 | W = $W_ENC $W_CFG (secondo giro) | s$S | $SPLIT -> $FUSION_DIR ==="
run python -m src.evaluation.fusion_head_v2 run --split "$SPLIT" \
  --gallery-names results/shared_gallery.json \
  --vision-qvec "$VISION_QVEC" --graph-qvec "$GRAPH_QVEC" \
  --fractions 0.0 0.25 0.5 0.75 \
  "${ALPHA_FLAGS[@]}" \
  --out "$FUSION_DIR"
RC=$?
echo "=== completato: $(date) | rc=$RC ==="
exit $RC
