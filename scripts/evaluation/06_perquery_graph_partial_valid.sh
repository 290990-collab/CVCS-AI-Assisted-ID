#!/bin/bash
# Degradation curve under masking on the GRAPH branch, on VALID, with per-query persistence.
# Twin of 05_perquery_vision_partial_valid.sh.
#
# Needed to apply the robustness criterion (AUC of the self-recovery MRR curve, f in {0.25, 0.5, 0.75})
# to the graph branch.
#
# Delegates to scripts/graph/04_eval_gnn.sh (as 04_perquery_graph.sh), passing PERQUERY_OUT and
# GRAPH_PARTIAL_FLAGS. Split and gallery_names come from configs/graph_retrieval.yaml.
#
# Pairing with vision requires the shared gallery: `gallery_names: results/shared_gallery.json` in BOTH YAMLs.
# The per-query seed is `seed + qi`, and `qi` matches only with the canonical order.
# graph_evaluate rewrites embeddings.npy/names.json of the checkpoint with the restricted, reordered gallery
# (always row-aligned).
#
# Usage:
#   sbatch scripts/evaluation/06_perquery_graph_partial_valid.sh            # baseline + the 6 variants of 04
#   sbatch scripts/evaluation/06_perquery_graph_partial_valid.sh gcn tau02  # a single pair
#
# Output: results/perquery/graph_partial_valid/graph_<enc_variant>_partial-random-f<frac>_valid.npz

#SBATCH --job-name="ev_06_perquery_graph_partial_valid"
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=all_usr_prod
#SBATCH --account=cvcs2026
# Excludes the Blackwell nodes (sm_120): same whitelist as 05.
#SBATCH --constraint="gpu_RTX5000_16G|gpu_2080_11G|gpu_2080Ti_11G|gpu_P100_16G|gpu_RTX_A5000_24G|gpu_A40_45G|gpu_L40S_45G"

set -uo pipefail
umask 002

PROJECT_DIR="/work/cvcs2026/ai_interior_design/CVCS-AI-Assisted-ID"
cd "$PROJECT_DIR" || exit 1

export PERQUERY_OUT="${PERQUERY_OUT:-results/perquery/graph_partial_valid}"
# `random` only: the only strategy of the robustness criterion.
export GRAPH_PARTIAL_FLAGS="--partial --partial-strategies random --partial-fractions 0.0 0.25 0.5 0.75"
mkdir -p logs "$PERQUERY_OUT"

DEFAULT_RUNS=(
  "hist"
  "gcn base"
  "gcn tau02"
  "graph_sage base"
  "graph_sage nd01"
  "gat base"
  "gat nosym"
)

if [ $# -gt 0 ]; then
  RUNS=("$*")
else
  RUNS=("${DEFAULT_RUNS[@]}")
fi

echo "=== $(date) | PARTIAL per-query ramo graph (C.0) | per-query -> $PERQUERY_OUT ==="
echo "partial: $GRAPH_PARTIAL_FLAGS | run: ${#RUNS[@]}"
grep -n "^gallery_names\|^split" configs/graph_retrieval.yaml
nvidia-smi

FAILED=0
for RUN in "${RUNS[@]}"; do
  echo ""
  echo "=============================================================="
  echo "[06_perquery_graph_partial_valid] $RUN"
  echo "=============================================================="
  # shellcheck disable=SC2086  # $RUN must split into <target> <variant>
  bash scripts/graph/04_eval_gnn.sh $RUN
  RC=$?
  [ $RC -ne 0 ] && { echo "!! FALLITA: '$RUN' (rc=$RC)" >&2; FAILED=$((FAILED + 1)); }
done

echo ""
echo "=== completato: $(date) | run fallite: $FAILED ==="
ls -la "$PERQUERY_OUT"

[ $FAILED -eq 0 ] || exit 1
