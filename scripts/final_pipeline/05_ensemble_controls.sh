#!/bin/bash
# The three controls of the main fusion (W seed 42), valid only.
#   replica       W s42 + W s100042 (damage seed 42: needs `02_eval_graph.sh <W> 100042 ctrl`)
#   crossenc      W s42 + E s42 (E = best configuration of another encoder)
#   visionvision  pespatial + radio fused files (read only, independent of the graph): only
#                 D2 = G_F - G_V is recomputed with the new fusion
#
# Two steps per control:
#   sbatch scripts/final_pipeline/05_ensemble_controls.sh fuse replica|crossenc     (CPU job: late fusion)
#   bash   scripts/final_pipeline/05_ensemble_controls.sh decide replica|crossenc|visionvision   (seconds)
# Order: main fusion s42 select valid -> fuse/decide replica -> fuse/decide crossenc ->
#        decide visionvision (it reads G_C of the replica control).
# Writes (new) results/final_pipeline/controls/<control>/. Every json once.

#SBATCH --job-name="rg_05_controls"
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

STEP="${1:-}"; CTRL="${2:-}"
case "$STEP/$CTRL" in
  fuse/replica|fuse/crossenc|decide/replica|decide/crossenc|decide/visionvision) ;;
  *) echo "!! uso: 05_ensemble_controls.sh fuse <replica|crossenc> | decide <replica|crossenc|visionvision>" >&2; exit 1 ;;
esac
SMOKE_FLAG=""
[ "${RG_SMOKE:-0}" = "1" ] && SMOKE_FLAG="--smoke"
run() { if [ "${RG_DRY_RUN:-0}" = "1" ]; then echo "+ $*"; else "$@"; fi; }
refuse() { [ ! -e "$1" ] || { echo "!! $1 esiste gia': niente sovrascritture" >&2; exit 1; }; }

eval "$(python -m src.evaluation.graph_config_select fusion-paths --seed 42 --split valid $SMOKE_FLAG)" || exit 1
OUT="$CTRL_ROOT/$CTRL"
FDIR="$OUT/fusion_valid"
if [ "$CTRL" = "replica" ]; then
  G2_QVEC="$REP_QVEC"; G2_PQ="$REP_PQ"; G2_DESC="$W_ENC $W_CFG s100042 (danno seed 42)"
elif [ "$CTRL" = "crossenc" ]; then
  [ -n "${E_QVEC:-}" ] || { echo "!! nessun E in final.json" >&2; exit 1; }
  G2_QVEC="$E_QVEC"; G2_PQ="$E_PQ"; G2_DESC="$E_ENC $E_CFG s42"
fi
[ -f "$MAIN_SELECT_VALID" ] || { echo "!! prima la fusione principale: manca $MAIN_SELECT_VALID" >&2; exit 1; }
echo "=== $(date) | RESET GRAPHS controllo $CTRL | $STEP | W = $W_ENC $W_CFG s42 ==="

if [ "$STEP" = "fuse" ]; then
  EXISTING=$(ls "$FDIR"/fusion_a*_valid.npz 2>/dev/null | wc -l)
  [ "$EXISTING" -eq 0 ] || { echo "!! $EXISTING file fusi gia' in $FDIR" >&2; exit 1; }
  [ "${RG_DRY_RUN:-0}" = "1" ] || mkdir -p "$FDIR"
  export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
  echo "beta=1: $W_ENC $W_CFG s42 · beta=0: $G2_DESC"
  run python -m src.evaluation.late_fusion run --pair graph-graph --split valid \
    --gallery-names results/shared_gallery.json \
    --graph-qvec "$W42_QVEC" --graph2-qvec "$G2_QVEC" \
    --fractions 0.0 0.25 0.5 0.75 --out "$FDIR"
  RC=$?
  echo "=== completato: $(date) | rc=$RC ==="
  echo "PROSSIMO PASSO: bash scripts/final_pipeline/05_ensemble_controls.sh decide $CTRL"
  exit $RC
fi

# decide
if [ "$CTRL" = "visionvision" ]; then
  refuse "$OUT/complementarity_valid.json"
  [ -f "$CTRL_ROOT/replica/complementarity_valid.json" ] || { echo "!! prima: decide replica" >&2; exit 1; }
  [ "${RG_DRY_RUN:-0}" = "1" ] || mkdir -p "$OUT"
  run python -m src.evaluation.fusion_select complementarity --control-pair vision-vision \
    --true-select "$MAIN_SELECT_VALID" --true-dir "$MAIN_FUSION_DIR" \
    --control-select "$VV_SELECT" --control-dir "$VV_DIR" \
    --graphgraph-json "$CTRL_ROOT/replica/complementarity_valid.json" \
    --graphgraph-dir "$CTRL_ROOT/replica/fusion_valid" \
    --out-json "$OUT/complementarity_valid.json"
  exit $?
fi

refuse "$OUT/check_valid_reset.json"
refuse "$OUT/select_valid.json"; refuse "$OUT/complementarity_valid.json"
[ -d "$FDIR" ] || { echo "!! manca $FDIR: prima 'fuse $CTRL'" >&2; exit 1; }
if [ -f "$OUT/check_valid.json" ]; then
  echo "(rilettura di $OUT/check_valid.json, fusion_select check non rilanciato)"
else
  run python -m src.evaluation.fusion_select check --pair graph-graph --split valid --fusion-dir "$FDIR" \
    --graph-qvec "$W42_QVEC" --graph-perquery "$W42_PQ" \
    --graph2-qvec "$G2_QVEC" --graph2-perquery "$G2_PQ" \
    --out-json "$OUT/check_valid.json"
  echo "(rc di fusion_select check: $? — decide la regola RESET sotto)"
fi
run python -m src.evaluation.graph_config_select fusion-check --check-json "$OUT/check_valid.json" \
  --waiver "$OUT/c3_waiver_valid.json" --out-json "$OUT/check_valid_reset.json" || exit 1
run python -m src.evaluation.fusion_select select --pair graph-graph --split valid --fusion-dir "$FDIR" \
  --out-json "$OUT/select_valid.json" || exit 1
if [ "$CTRL" = "replica" ]; then
  run python -m src.evaluation.fusion_select complementarity --control-pair graph-graph \
    --true-select "$MAIN_SELECT_VALID" --true-dir "$MAIN_FUSION_DIR" \
    --control-select "$OUT/select_valid.json" --control-dir "$FDIR" \
    --out-json "$OUT/complementarity_valid.json"
else
  run python -m src.evaluation.graph_config_select crossenc \
    --true-select "$MAIN_SELECT_VALID" --true-dir "$MAIN_FUSION_DIR" \
    --control-select "$OUT/select_valid.json" --control-dir "$FDIR" \
    --out-json "$OUT/complementarity_valid.json"
fi
