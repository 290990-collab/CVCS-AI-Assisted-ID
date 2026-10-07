#!/bin/bash
# Multi-seed round: the graph + graph controls ("same information, another model"):
#   A  W on replicas 100042/200042/300042: + second copy of W (next seed, same removed rooms), + sage/comb
#   C  gat/t05 and gat/ref (strength similar to the vision), seed 42: + second copy (s100042), + same change on SAGE
# The list is in src/evaluation/multiseed_runs.py (`python -m src.evaluation.multiseed_runs controls --split valid`).
#
# Steps:
#   sbatch --array=0-9 scripts/final_pipeline/13_ensemble_controls_multiseed.sh fuse valid     (CPU: late fusion, weight grid)
#   bash   scripts/final_pipeline/13_ensemble_controls_multiseed.sh check valid                  (checks + select, all controls)
#   (test, only after TEST_PREREGISTERED_R2: same with `test`; weight fixed from select_valid.json)
# A C3 FAIL is overcome only by <control>/c3_waiver_<split>.json (src.evaluation.near_ties).
# Writes (new) results/final_pipeline/round2/controls/<name>/. Every json once.

#SBATCH --job-name="rg_13_controls_r2"
#SBATCH --output=logs/%x_%A_%a.log
#SBATCH --error=logs/%x_%A_%a.err
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

STEP="${1:-}"; SPLIT="${2:-}"
case "$STEP/$SPLIT" in fuse/valid|fuse/test|check/valid|check/test) ;;
  *) echo "!! uso: 13_ensemble_controls_multiseed.sh <fuse|check> <valid|test>" >&2; exit 1 ;; esac
run() { if [ "${RG_DRY_RUN:-0}" = "1" ]; then echo "+ $*"; else "$@"; fi; }
python -m src.evaluation.multiseed_runs gate --split "$SPLIT" || exit 1

beta_of() { python -c 'import json,sys; print(format(float(json.load(open(sys.argv[1]))["alpha_star"]), "g"))' "$1"; }

if [ "$STEP" = "fuse" ]; then
  I="${SLURM_ARRAY_TASK_ID:-${3:-}}"
  [[ "$I" =~ ^[0-9]+$ ]] || { echo "!! indice mancante: usa sbatch --array=0-9" >&2; exit 1; }
  eval "$(python -m src.evaluation.multiseed_runs control-paths --split "$SPLIT" --index "$I")" || exit 1
  EXISTING=$(ls "$DIR"/fusion_a*_"$SPLIT".npz 2>/dev/null | wc -l)
  [ "$EXISTING" -eq 0 ] || { echo "!! $EXISTING file fusi gia' in $DIR" >&2; exit 1; }
  if [ "$SPLIT" = "valid" ]; then
    WEIGHTS=(--alphas $ALPHAS)
  else
    [ -f "$SELECT_VALID" ] || { echo "!! manca $SELECT_VALID (controllo sul valid)" >&2; exit 1; }
    WEIGHTS=(--alphas-from "$SELECT_VALID")
  fi
  [ "${RG_DRY_RUN:-0}" = "1" ] || mkdir -p "$DIR"
  export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
  echo "=== $(date) | secondo giro, controllo $NAME ($SPLIT) | beta=1: $G1_DESC · beta=0: $G2_DESC ==="
  run python -m src.evaluation.late_fusion run --pair graph-graph --split "$SPLIT" \
    --gallery-names results/shared_gallery.json --graph-qvec "$G1_QVEC" --graph2-qvec "$G2_QVEC" \
    --fractions 0.0 0.25 0.5 0.75 "${WEIGHTS[@]}" --out "$DIR"
  RC=$?
  echo "=== completato: $(date) | rc=$RC ==="
  exit $RC
fi

# check (+ select), every control of the split, on the login node (OMP_NUM_THREADS=4)
export OMP_NUM_THREADS=4
N=$(python -m src.evaluation.multiseed_runs controls --split "$SPLIT" | wc -l)
FAIL=0
for I in $(seq 0 $((N - 1))); do
  eval "$(python -m src.evaluation.multiseed_runs control-paths --split "$SPLIT" --index "$I")" || exit 1
  echo ""
  echo "=== $NAME ($SPLIT) ==="
  [ -d "$DIR" ] || { echo "!! manca $DIR: prima 'fuse'" >&2; FAIL=1; continue; }
  if [ -e "$SELECT" ]; then echo "(gia' fatto: $SELECT)"; continue; fi
  if [ "$SPLIT" = "valid" ]; then
    CHECK_ALPHAS="$ALPHAS"; SEL=(--alphas $ALPHAS)
  else
    CHECK_ALPHAS="0 $(beta_of "$SELECT_VALID") 1"; SEL=(--fixed-alpha-from "$SELECT_VALID")
  fi
  if [ ! -f "$CHECK" ]; then
    run python -m src.evaluation.fusion_select check --pair graph-graph --split "$SPLIT" --fusion-dir "$DIR" \
      --graph-qvec "$G1_QVEC" --graph-perquery "$G1_PQ" --graph2-qvec "$G2_QVEC" --graph2-perquery "$G2_PQ" \
      --alphas $CHECK_ALPHAS --out-json "$CHECK"
  fi
  if [ ! -f "$CHECK_RESET" ]; then
    run python -m src.evaluation.graph_config_select fusion-check --check-json "$CHECK" \
      --waiver "$WAIVER" --out-json "$CHECK_RESET" || { FAIL=1; echo "!! $NAME: controlli non superati (diagnosi dei pareggi)"; continue; }
  fi
  run python -m src.evaluation.fusion_select select --pair graph-graph --split "$SPLIT" --fusion-dir "$DIR" \
    "${SEL[@]}" --out-json "$SELECT" || FAIL=1
done
echo ""
echo "=== check $SPLIT finito | fallimenti: $FAIL ==="
exit $FAIL
