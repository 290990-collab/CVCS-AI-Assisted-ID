#!/bin/bash
# Late fusion of the winner W (selection/final.json) with the frozen vision `pespatial/gem/whiten`, for ONE
# replica (`src.evaluation.late_fusion run`, alpha grid 0..1 step 0.1 on valid; on test only {0, alpha*(S), 1}
# from the replica's select_valid.json). CPU only.
#
# Inputs (read only): graph qvec of W written by 02_eval_graph.sh; vision qvec of seed 42 or of seeds
# 100042/200042/300042: same removed rooms, checked by `fusion_select check` C1.
# Writes (new) results/final_pipeline/fusion/s<S>/fusion_<split>/. Refuses if not empty.
#
# Usage:
#   sbatch scripts/final_pipeline/03_fusion_frozen_vision.sh valid 42        # then 100042 200042 300042
#   sbatch scripts/final_pipeline/03_fusion_frozen_vision.sh test 42         # only after the test pre-registration
#   RG_DRY_RUN=1 bash scripts/final_pipeline/03_fusion_frozen_vision.sh valid 42   # prints the command
# Then: bash scripts/final_pipeline/04_fusion_frozen_vision_select.sh check <split> <S>

#SBATCH --job-name="rg_03_fusion"
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
case "$SPLIT" in valid|test) ;; *) echo "!! uso: 03_fusion_frozen_vision.sh <valid|test> <seed>" >&2; exit 1 ;; esac
[[ "$S" =~ ^[0-9]+$ ]] || { echo "!! seed non valido: '$S'" >&2; exit 1; }
SMOKE_FLAG=""
[ "${RG_SMOKE:-0}" = "1" ] && SMOKE_FLAG="--smoke"
run() { if [ "${RG_DRY_RUN:-0}" = "1" ]; then echo "+ $*"; else "$@"; fi; }

PATHS=$(python -m src.evaluation.graph_config_select fusion-paths --seed "$S" --split "$SPLIT" $SMOKE_FLAG) || exit 1
eval "$PATHS"

ALPHA_FLAGS=()
if [ "$SPLIT" = "test" ]; then
  python -m src.evaluation.graph_config_select test-gate --encoder "$W_ENC" --cfg "$W_CFG" --seed "$S" $SMOKE_FLAG || exit 1
  ALPHA_FLAGS=(--alphas-from "$SELECT_VALID_JSON")
fi

EXISTING=$(ls "$FUSION_DIR"/fusion_a*_"${SPLIT}".npz 2>/dev/null | wc -l)
if [ "$EXISTING" -gt 0 ]; then
  echo "!! ERRORE: $EXISTING file fusi gia' presenti in $FUSION_DIR — niente sovrascritture" >&2
  exit 1
fi
[ "${RG_DRY_RUN:-0}" = "1" ] || mkdir -p "$FUSION_DIR"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "=== $(date) | RESET GRAPHS fusione | W = $W_ENC $W_CFG | s$S | $SPLIT -> $FUSION_DIR ==="
run python -m src.evaluation.late_fusion run --split "$SPLIT" \
  --gallery-names results/shared_gallery.json \
  --vision-qvec "$VISION_QVEC" --graph-qvec "$GRAPH_QVEC" \
  --fractions 0.0 0.25 0.5 0.75 \
  "${ALPHA_FLAGS[@]}" \
  --out "$FUSION_DIR"
RC=$?
echo "=== completato: $(date) | rc=$RC ==="
echo "PROSSIMO PASSO: bash scripts/final_pipeline/04_fusion_frozen_vision_select.sh check $SPLIT $S"
exit $RC
