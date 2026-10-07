#!/bin/bash
# Graph branch: rerun the known evaluations with per-query persistence.
#
# The graph variant is selected on valid by the training-time `RetrievalProbe`
# (src/graph/training/train_gnn.py:174-181), so per-query files of the already selected variants suffice
# and paired comparisons become possible.
#
# Delegates to scripts/graph/04_eval_gnn.sh, the only place where architecture flags are derived from the
# training YAML (a divergence would break checkpoint reload); this script only adds PERQUERY_OUT.
#
# The split comes from configs/graph_retrieval.yaml (`valid`); for the protocol-B test use 07_perquery_graph_test.sh.
#
# Usage:
#   sbatch scripts/evaluation/04_perquery_graph.sh            # baseline + the 3 winners + the 3 bases
#   sbatch scripts/evaluation/04_perquery_graph.sh gcn tau02  # a single encoder/variant pair
#
# Output: results/perquery/graph_test/*.npz

#SBATCH --job-name="ev_04_perquery_graph"
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=all_usr_prod
#SBATCH --account=cvcs2026
# Excludes the Blackwell nodes (sm_120): 04_eval_gnn.sh is called with `bash`, so this script's header applies.
#SBATCH --constraint="gpu_RTX5000_16G|gpu_2080_11G|gpu_2080Ti_11G|gpu_P100_16G|gpu_RTX_A5000_24G|gpu_A40_45G|gpu_L40S_45G"

set -uo pipefail
umask 002

PROJECT_DIR="/work/cvcs2026/ai_interior_design/CVCS-AI-Assisted-ID"
cd "$PROJECT_DIR" || exit 1
mkdir -p logs

# Read by scripts/graph/04_eval_gnn.sh and turned into `--perquery-out`.
# Env override: the split comes from configs/graph_retrieval.yaml (`valid`), so use a dedicated folder for it.
export PERQUERY_OUT="${PERQUERY_OUT:-results/perquery/graph_test}"

# Pairs <target> <variant>; `hist` is the training-free baseline (no variant). The others are the measured
# winners plus their `base`, so winner-vs-base is paired by construction (same gallery, same queries).
# `graph_sage` is the YAML basename; the registry key is `sage`.
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

mkdir -p "$PERQUERY_OUT"
echo "=== $(date) | PER-QUERY ramo graph | per-query -> $PERQUERY_OUT ==="
echo "run: ${#RUNS[@]}"
nvidia-smi

FAILED=0
for RUN in "${RUNS[@]}"; do
  echo ""
  echo "=============================================================="
  echo "[04_perquery_graph] $RUN"
  echo "=============================================================="
  # shellcheck disable=SC2086  # $RUN must split into <target> <variant>
  bash scripts/graph/04_eval_gnn.sh $RUN
  RC=$?
  [ $RC -ne 0 ] && { echo "!! FALLITA: '$RUN' (rc=$RC)" >&2; FAILED=$((FAILED + 1)); }
done

echo ""
echo "=== completato: $(date) | run fallite: $FAILED ==="
ls -la "$PERQUERY_OUT"
echo ""
echo "PROSSIMO PASSO — confronti appaiati DENTRO il ramo (nessun caveat, stessa"
echo "gallery e stesse query):"
echo "  python -m src.evaluation.significance \\"
echo "      --a $PERQUERY_OUT/<gcn_tau02>.npz --b $PERQUERY_OUT/<gcn_base>.npz --k 10"
echo ""
echo "Con la gallery condivisa (B.3) il confronto vision<->graph NON richiede"
echo "--allow-gallery-mismatch: se lo chiede, i rami usano gallery diverse."

[ $FAILED -eq 0 ] || exit 1
