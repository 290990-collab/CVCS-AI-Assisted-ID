#!/bin/bash
# The two graph + graph controls on the test, seed 42, beta fixed from the valid ({0, beta, 1}), then the
# confirmatory comparison with the fusion with head.
#   replica   W s42 + W s100042 evaluated on the test with damage seed 42 (`02_eval_graph.sh ... ctrltest`)
#   crossenc  W s42 + sage/comb s42 (test)
#
# Steps:
#   sbatch scripts/final_pipeline/10_ensemble_controls_test.sh fuse  replica|crossenc   (CPU job: late fusion)
#   bash   scripts/final_pipeline/10_ensemble_controls_test.sh check replica|crossenc   (checks + select, minutes)
#   bash   scripts/final_pipeline/10_ensemble_controls_test.sh decide                   (needs both + 08 select test 42)
# A C3 FAIL is overcome only by controls_test/<ctrl>/c3_waiver_test.json (src.evaluation.near_ties).
# Writes (new) results/final_pipeline/controls_test/. Every json once.

#SBATCH --job-name="rg_10_controls_test"
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
export CUDA_VISIBLE_DEVICES=""

STEP="${1:-}"; CTRL="${2:-}"
case "$STEP/$CTRL" in
  fuse/replica|fuse/crossenc|check/replica|check/crossenc|decide/) ;;
  *) echo "!! uso: 10_ensemble_controls_test.sh fuse|check <replica|crossenc> | decide" >&2; exit 1 ;;
esac
run() { if [ "${RG_DRY_RUN:-0}" = "1" ]; then echo "+ $*"; else "$@"; fi; }
refuse() { [ ! -e "$1" ] || { echo "!! $1 esiste gia': niente sovrascritture" >&2; exit 1; }; }

eval "$(python -m src.evaluation.ensemble_control_test paths)" || exit 1
[ -e results/final_pipeline/TEST_PREREGISTERED ] || { echo "!! manca results/final_pipeline/TEST_PREREGISTERED" >&2; exit 1; }

if [ "$STEP" = "decide" ]; then
  refuse "$CONFIRM_JSON"
  run python -m src.evaluation.ensemble_control_test decide
  exit $?
fi

U=$(echo "$CTRL" | tr a-z A-Z)
for V in QVEC PQ DIR CHECK CHECK_RESET WAIVER SELECT SELECT_VALID; do N="${U}_$V"; printf -v "C_$V" '%s' "${!N}"; done
[ -f "$C_SELECT_VALID" ] || { echo "!! manca $C_SELECT_VALID (controllo sul valid)" >&2; exit 1; }
BETA=$(python -c 'import json,sys; print(format(float(json.load(open(sys.argv[1]))["alpha_star"]), "g"))' "$C_SELECT_VALID") \
  || { echo "!! beta illeggibile in $C_SELECT_VALID" >&2; exit 1; }
echo "=== $(date) | RESET GRAPHS test, controllo $CTRL | $STEP | beta=1: $W_ENC $W_CFG s42 · beta=0: $C_QVEC | beta $BETA ==="

if [ "$STEP" = "fuse" ]; then
  EXISTING=$(ls "$C_DIR"/fusion_a*_test.npz 2>/dev/null | wc -l)
  [ "$EXISTING" -eq 0 ] || { echo "!! $EXISTING file fusi gia' in $C_DIR" >&2; exit 1; }
  [ "${RG_DRY_RUN:-0}" = "1" ] || mkdir -p "$C_DIR"
  export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
  run python -m src.evaluation.late_fusion run --pair graph-graph --split test \
    --gallery-names results/shared_gallery.json \
    --graph-qvec "$W42_QVEC" --graph2-qvec "$C_QVEC" \
    --fractions 0.0 0.25 0.5 0.75 --alphas-from "$C_SELECT_VALID" --out "$C_DIR"
  RC=$?
  echo "=== completato: $(date) | rc=$RC ==="
  echo "PROSSIMO PASSO: bash scripts/final_pipeline/10_ensemble_controls_test.sh check $CTRL"
  exit $RC
fi

# check (+ select with beta fixed from the valid)
[ -d "$C_DIR" ] || { echo "!! manca $C_DIR: prima 'fuse $CTRL'" >&2; exit 1; }
refuse "$C_CHECK_RESET"; refuse "$C_SELECT"
if [ -f "$C_CHECK" ]; then
  echo "(rilettura di $C_CHECK, fusion_select check non rilanciato)"
else
  run python -m src.evaluation.fusion_select check --pair graph-graph --split test --fusion-dir "$C_DIR" \
    --graph-qvec "$W42_QVEC" --graph-perquery "$W42_PQ" \
    --graph2-qvec "$C_QVEC" --graph2-perquery "$C_PQ" \
    --alphas 0 "$BETA" 1 --out-json "$C_CHECK"
  echo "(rc di fusion_select check: $? — decide la regola RESET sotto)"
fi
run python -m src.evaluation.graph_config_select fusion-check --check-json "$C_CHECK" \
  --waiver "$C_WAIVER" --out-json "$C_CHECK_RESET" || exit 1
run python -m src.evaluation.fusion_select select --pair graph-graph --split test --fusion-dir "$C_DIR" \
  --fixed-alpha-from "$C_SELECT_VALID" --out-json "$C_SELECT" || exit 1
echo "PROSSIMO PASSO: l'altro controllo, poi bash scripts/final_pipeline/10_ensemble_controls_test.sh decide"
