#!/bin/bash
# Secondary fusion: W + vision with head, ONE replica. CPU only.
# Test: only after TEST_PREREGISTERED, alphas {0, alpha*_H, 1} with alpha*_H from the replica's
# fusion_head/s<S>/select_valid.json.
# Same fusion code as 03_fusion_frozen_vision.sh (`late_fusion.run`, alpha grid 0..1 step 0.1), run through
# `src.evaluation.fusion_head run`, which applies the frozen head to the RAW vision vectors before the
# whitening and first checks head + whitening = the vision job's final query vectors (<= 1e-5).
#
# Inputs (read only): graph qvec of W (02_eval_graph.sh), vision-with-head qvec (06_queryvec_vision_head.sh).
# Writes (new) results/final_pipeline/fusion_head/s<S>/fusion_<split>/ + head_fusion_<split>.json.
#
# Usage:
#   sbatch scripts/final_pipeline/07_fusion_head.sh valid 42        # then 100042 200042 300042
#   sbatch scripts/final_pipeline/07_fusion_head.sh test 42         # only after the test pre-registration
# Then: bash scripts/final_pipeline/08_fusion_head_select.sh check valid <S>

#SBATCH --job-name="rg_07_fusion_head"
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
mkdir -p logs

SPLIT="${1:-}"; S="${2:-}"
case "$SPLIT" in valid|test) ;; *) echo "!! split '$SPLIT' (valid | test)" >&2; exit 1 ;; esac
[[ "$S" =~ ^[0-9]+$ ]] || { echo "!! seed non valido: '$S'" >&2; exit 1; }
run() { if [ "${RG_DRY_RUN:-0}" = "1" ]; then echo "+ $*"; else "$@"; fi; }

eval "$(python -m src.evaluation.fusion_head paths --seed "$S" --split "$SPLIT")" || exit 1
ALPHA_FLAGS=()
if [ "$SPLIT" = "test" ]; then
  python -m src.evaluation.graph_config_select test-gate --encoder "$W_ENC" --cfg "$W_CFG" --seed "$S" || exit 1
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

echo "=== $(date) | fusione CON HEAD | W = $W_ENC $W_CFG | s$S | $SPLIT -> $FUSION_DIR ==="
run python -m src.evaluation.fusion_head run --split "$SPLIT" \
  --gallery-names results/shared_gallery.json \
  --vision-qvec "$VISION_QVEC" --graph-qvec "$GRAPH_QVEC" \
  --fractions 0.0 0.25 0.5 0.75 \
  "${ALPHA_FLAGS[@]}" \
  --out "$FUSION_DIR"
RC=$?
echo "=== completato: $(date) | rc=$RC ==="
echo "PROSSIMO PASSO: bash scripts/final_pipeline/08_fusion_head_select.sh check $SPLIT $S"
exit $RC
