#!/bin/bash
# ONE graph training: encoder x configuration x seed.
#
# Recipe: encoder YAML (unchanged) + asym_partial pairs + epoch chosen on the robustness probe, 600 epochs,
# patience 0, shadow `_selfull` (as gat/asymrob) + the configuration's overrides. Flags and paths come from
# src/graph/final_graph_configs.py (the only table): this script spells none of them.
#
# Writes (new) embeddings/graph/<key>/rg_<cfg>_s<S>{,_selfull}/ (+ reset_recipe.json).
# Refuses to start if the destination exists.
#
# Usage:
#   sbatch scripts/final_pipeline/01_train_graph.sh gat t02 100042
#   (encoder = YAML basename: gcn | graph_sage | gat; seed in 42 100042 200042 300042)
#   RG_SMOKE=1 sbatch --time=00:30:00 scripts/final_pipeline/01_train_graph.sh gcn ref 42   # smoke test, 2 epochs, prefix rgsmoke
#
# Then: sbatch --dependency=afterok:<id> scripts/final_pipeline/02_eval_graph.sh <same args> valid

#SBATCH --job-name="rg_01_train"
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
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

ENC="${1:-}"; CFG="${2:-}"; S="${3:-}"
if [ -z "$ENC" ] || [ -z "$CFG" ] || ! [[ "$S" =~ ^[0-9]+$ ]]; then
  echo "!! ERRORE: uso: 01_train_graph.sh <gcn|graph_sage|gat> <cfg> <seed>" >&2
  exit 1
fi
SMOKE_FLAG=""
[ "${RG_SMOKE:-0}" = "1" ] && SMOKE_FLAG="--smoke"

RC_PY=src.graph.final_graph_configs
PATHS=$(python -m $RC_PY paths --encoder "$ENC" --cfg "$CFG" --seed "$S" --split valid $SMOKE_FLAG) || exit 1
eval "$PATHS"
FLAGS=$(python -m $RC_PY train-flags --encoder "$ENC" --cfg "$CFG" --seed "$S" $SMOKE_FLAG) || exit 1

# guard: never overwrite (shadow folder included)
for D in "$DEST" "${DEST}_selfull"; do
  if [ -e "$D" ]; then
    echo "!! ERRORE: esiste gia' $D — niente sovrascritture (VERIFICA_GRAFI.md §7.5: si toglie" >&2
    echo "   solo una cartella INCOMPLETA di un training fallito, a mano, e lo si annota)" >&2
    exit 1
  fi
done

echo "=== $(date) | RESET GRAPHS training | $ENC $CFG s$S | $VARIANT ${SMOKE_FLAG:+| SMOKE} ==="
echo "dest: $DEST (+ _selfull) | epochs: $EPOCHS"
echo "flags: $FLAGS"
nvidia-smi

eval "python -m src.graph.training.train_gnn $FLAGS"
RC=$?

if [ $RC -eq 0 ]; then
  python -m $RC_PY recipe-json --encoder "$ENC" --cfg "$CFG" --seed "$S" $SMOKE_FLAG \
    > "$DEST/reset_recipe.json" || RC=1
fi
for F in encoder.pt training_summary.json reset_recipe.json; do
  [ -f "$DEST/$F" ] || { echo "!! MANCANTE: $DEST/$F" >&2; RC=1; }
done

echo ""
echo "=== completato: $(date) | rc=$RC ==="
exit $RC
