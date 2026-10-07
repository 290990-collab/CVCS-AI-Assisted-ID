#!/bin/bash
# Multi-seed round: evaluation of ONE already trained graph with a given damage seed, removed rooms only
# (f = 0 0.25 0.5 0.75), per-query + query vectors.
# The list of evaluations is in src/evaluation/multiseed_runs.py (`python -m src.evaluation.multiseed_runs evals --split valid`).
#
# Differences from 02_eval_graph.sh: the gallery embeddings go to a new folder (graph_evaluate --gallery-out), so
# nothing in embeddings/graph/ is rewritten; outputs under results/final_pipeline/round2/ only.
# Gates: valid after results/final_pipeline/ROUND2_PREREGISTERED, test after TEST_PREREGISTERED_R2.
#
# Usage:
#   sbatch --array=0-10 scripts/final_pipeline/12_eval_graph_multiseed.sh valid
#   sbatch --array=0-22 scripts/final_pipeline/12_eval_graph_multiseed.sh test        (only after the test pre-registration)

#SBATCH --job-name="rg_12_eval_r2"
#SBATCH --output=logs/%x_%A_%a.log
#SBATCH --error=logs/%x_%A_%a.err
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

SPLIT="${1:-}"
I="${SLURM_ARRAY_TASK_ID:-${2:-}}"
case "$SPLIT" in valid|test) ;; *) echo "!! uso: 12_eval_graph_multiseed.sh <valid|test> (indice da --array)" >&2; exit 1 ;; esac
[[ "$I" =~ ^[0-9]+$ ]] || { echo "!! indice mancante: usa sbatch --array=..." >&2; exit 1; }

python -m src.evaluation.multiseed_runs gate --split "$SPLIT" || exit 1
PATHS=$(python -m src.evaluation.multiseed_runs eval-paths --split "$SPLIT" --index "$I") || exit 1
eval "$PATHS"
FLAGS=$(python -m src.graph.final_graph_configs eval-flags --encoder "$ENC" --cfg "$CFG" --seed "$SEED" --split "$SPLIT") || exit 1
[ -f "$DEST/encoder.pt" ] || { echo "!! checkpoint mancante: $DEST/encoder.pt" >&2; exit 1; }

EXISTING=$(ls "$PQ_DIR/${TAG}"_*"_${EVAL_SPLIT}.npz" "$QV_DIR/${TAG}"_*"_${EVAL_SPLIT}.npz" "$GALLERY_DIR/embeddings.npy" 2>/dev/null | wc -l)
[ "$EXISTING" -eq 0 ] || { echo "!! $EXISTING file di $TAG gia' presenti (round2): niente sovrascritture" >&2; exit 1; }
mkdir -p "$PQ_DIR" "$QV_DIR"

echo "=== $(date) | RESET secondo giro | task $I | $ENC $CFG s$SEED | split $SPLIT | danno seed $PARTIAL_SEED ==="
echo "per-query -> $PQ_DIR | qvec -> $QV_DIR | gallery -> $GALLERY_DIR"
nvidia-smi
START_MARK=$(mktemp)
RC=0
eval "python -m src.graph.evaluation.graph_evaluate $FLAGS --save-dir \"\$DEST\" --gallery-out \"\$GALLERY_DIR\" \
  --perquery-out \"\$PQ_DIR\" --partial --partial-strategies random --partial-fractions 0.0 0.25 0.5 0.75 \
  --partial-seed \"\$PARTIAL_SEED\" --query-vectors-out \"\$QV_DIR\"" || RC=1

MISSING=0
for P in "$GALLERY_DIR/embeddings.npy"; do
  [ -f "$P" ] && [ "$P" -nt "$START_MARK" ] || { echo "!! MANCANTE o vecchio: $P" >&2; MISSING=$((MISSING + 1)); }
done
for F in 0.0 0.25 0.5 0.75; do
  for P in "$PQ_DIR/${TAG}_partial-random-f${F}_${EVAL_SPLIT}.npz" "$QV_DIR/${TAG}_partial-random-f${F}_${EVAL_SPLIT}.npz"; do
    [ -f "$P" ] && [ "$P" -nt "$START_MARK" ] || { echo "!! MANCANTE o vecchio: $P" >&2; MISSING=$((MISSING + 1)); }
  done
done
rm -f "$START_MARK"
echo "=== completato: $(date) | rc=$RC | file mancanti: $MISSING ==="
[ $RC -eq 0 ] && [ $MISSING -eq 0 ] || exit 1
