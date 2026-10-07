#!/bin/bash
# Evaluation of ONE graph training, in ONE job:
#   1. full plan (per-axis nDCG@10, needed by the tie-break and the full-plan cost);
#   2. rooms removed at random, f = 0.0 0.25 0.5 0.75, damage seed = the replica's seed, per-query +
#      query vectors (qvec/1, used by the fusion: no re-eval).
# The partial run is last: the qvec pin the sha1 of the embeddings.npy it writes.
#
# Splits:
#   valid     selection (stage 1/2)
#   test      only after the test pre-registration (gate in graph_config_select test-gate)
#   ctrl      valid, partial only, damage seed 42; only the W replica 100042, for the replica control, only
#             after the replica fusion s100042 (gate in graph_config_select ctrl-gate)
#   ctrltest  the same on the test, only after the checks of the test fusions of s100042, main and with head
#             (gate in graph_config_select ctrltest-gate)
#
# Writes (new) results/final_pipeline/{perquery,queryvec}/<split>/s<S>/graph_<key>_rg_<cfg>_s<S>_*
# (ctrl: .../valid/ctrl_p42/, ctrltest: .../test/ctrl_p42/) and rewrites embeddings.npy/names.json of its own rg_ folder.
# Refuses to start if one of its output files exists.
#
# Usage:
#   sbatch --dependency=afterok:<train id> scripts/final_pipeline/02_eval_graph.sh gat t02 100042 valid

#SBATCH --job-name="rg_02_eval"
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=all_usr_prod
#SBATCH --account=cvcs2026
#SBATCH --constraint="gpu_RTX5000_16G|gpu_2080_11G|gpu_2080Ti_11G|gpu_P100_16G|gpu_RTX_A5000_24G|gpu_A40_45G|gpu_L40S_45G"

set -uo pipefail
umask 002

PROJECT_DIR="/work/cvcs2026/ai_interior_design/CVCS-AI-Assisted-ID"
source ~/floorplan-env/bin/activate
cd "$PROJECT_DIR" || exit 1
mkdir -p logs

ENC="${1:-}"; CFG="${2:-}"; S="${3:-}"; SPLIT="${4:-}"
if [ -z "$ENC" ] || [ -z "$CFG" ] || ! [[ "$S" =~ ^[0-9]+$ ]]; then
  echo "!! ERRORE: uso: 02_eval_graph.sh <gcn|graph_sage|gat> <cfg> <seed> <valid|test|ctrl|ctrltest>" >&2
  exit 1
fi
case "$SPLIT" in
  valid|test|ctrl|ctrltest) ;;
  *) echo "!! ERRORE: split '$SPLIT' (attesi: valid | test | ctrl | ctrltest)" >&2; exit 1 ;;
esac
SMOKE_FLAG=""
[ "${RG_SMOKE:-0}" = "1" ] && SMOKE_FLAG="--smoke"

RC_PY=src.graph.final_graph_configs
PATHS=$(python -m $RC_PY paths --encoder "$ENC" --cfg "$CFG" --seed "$S" --split "$SPLIT" $SMOKE_FLAG) || exit 1
eval "$PATHS"
FLAGS=$(python -m $RC_PY eval-flags --encoder "$ENC" --cfg "$CFG" --seed "$S" --split "$SPLIT" $SMOKE_FLAG) || exit 1

[ -f "$DEST/encoder.pt" ] || { echo "!! ERRORE: checkpoint mancante: $DEST/encoder.pt" >&2; exit 1; }

# Gates: the test only after its pre-registration; ctrl/ctrltest only after the replica fusions.
if [ "$SPLIT" != "valid" ]; then
  python -m src.evaluation.graph_config_select "${SPLIT}-gate" --encoder "$ENC" --cfg "$CFG" --seed "$S" $SMOKE_FLAG \
    || exit 1
fi

# Guard: none of the output files may exist.
EXISTING=$(ls "$PQ_DIR/${TAG}"_*"_${EVAL_SPLIT}.npz" "$QV_DIR/${TAG}"_*"_${EVAL_SPLIT}.npz" 2>/dev/null | wc -l)
if [ "$EXISTING" -gt 0 ]; then
  echo "!! ERRORE: $EXISTING file di $TAG gia' presenti in $PQ_DIR o $QV_DIR — niente sovrascritture" >&2
  exit 1
fi
mkdir -p "$PQ_DIR" "$QV_DIR"

echo "=== $(date) | RESET GRAPHS eval | $ENC $CFG s$S | split $SPLIT ($EVAL_SPLIT) | danno seed $PARTIAL_SEED ==="
echo "per-query -> $PQ_DIR | qvec -> $QV_DIR"
echo "flags: $FLAGS"
nvidia-smi
START_MARK=$(mktemp)
RC=0

CTRL=0
case "$SPLIT" in ctrl|ctrltest) CTRL=1 ;; esac

if [ $CTRL -eq 0 ]; then
  echo ""
  echo "########  pianta intera  ########"
  eval "python -m src.graph.evaluation.graph_evaluate $FLAGS --perquery-out \"\$PQ_DIR\"" || RC=1
fi

if [ $RC -eq 0 ]; then
  echo ""
  echo "########  stanze tolte (random), con i vettori delle query  ########"
  eval "python -m src.graph.evaluation.graph_evaluate $FLAGS --perquery-out \"\$PQ_DIR\" \
    --partial --partial-strategies random --partial-fractions 0.0 0.25 0.5 0.75 \
    --partial-seed \"\$PARTIAL_SEED\" --query-vectors-out \"\$QV_DIR\"" || RC=1
fi

# Every expected file must be there AND written by this job.
EXPECTED=()
[ $CTRL -eq 0 ] && EXPECTED+=("$PQ_DIR/${TAG}_full_${EVAL_SPLIT}.npz")
for F in 0.0 0.25 0.5 0.75; do
  EXPECTED+=("$PQ_DIR/${TAG}_partial-random-f${F}_${EVAL_SPLIT}.npz" "$QV_DIR/${TAG}_partial-random-f${F}_${EVAL_SPLIT}.npz")
done
MISSING=0
for P in "${EXPECTED[@]}"; do
  if [ ! -f "$P" ] || [ ! "$P" -nt "$START_MARK" ]; then
    echo "!! MANCANTE o vecchio: $P" >&2
    MISSING=$((MISSING + 1))
  fi
done
rm -f "$START_MARK"

echo ""
echo "=== completato: $(date) | rc=$RC | file mancanti: $MISSING ==="
[ $RC -eq 0 ] && [ $MISSING -eq 0 ] || exit 1
