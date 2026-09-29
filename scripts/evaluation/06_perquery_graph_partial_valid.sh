#!/bin/bash
# scripts/evaluation/06_perquery_graph_partial_valid.sh
# FASE C.0 — curva di degrado sotto masking sul ramo GRAPH, sul VALID, con
# persistenza per-query. Gemello di 05_perquery_vision_partial_valid.sh.
#
# Perche' esiste: senza partial il criterio A.5 (status.md § 23, AUC della curva
# self-recovery MRR su f in {0.25, 0.5, 0.75}) non si applica al ramo graph, e la
# sua config congelata (B.7) resta provvisoria.
#
# NON duplica il ponte YAML->flag: delega a scripts/graph/04_eval_gnn.sh (come
# 04_perquery_graph.sh), passandogli PERQUERY_OUT e GRAPH_PARTIAL_FLAGS.
# Split e gallery_names arrivano da configs/graph_retrieval.yaml.
#
# ⚠️ Il pairing con il vision richiede la gallery condivisa (B.3):
#    `gallery_names: results/shared_gallery.json` in ENTRAMBI i YAML. Il seed per
#    query e' `seed + qi`, e solo con l'ordine canonico il `qi` coincide.
# ⚠️ graph_evaluate riscrive embeddings.npy/names.json del checkpoint con la
#    gallery ristretta e riordinata (sempre allineati per riga).
#
# Uso:
#   sbatch scripts/evaluation/06_perquery_graph_partial_valid.sh            # baseline + le 6 varianti di 04
#   sbatch scripts/evaluation/06_perquery_graph_partial_valid.sh gcn tau02  # una sola coppia
#
# Output: results/perquery/graph_partial_valid/graph_<enc_variante>_partial-random-f<frac>_valid.npz

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
# Esclude i nodi Blackwell (sm_120): stessa whitelist di 05.
#SBATCH --constraint="gpu_RTX5000_16G|gpu_2080_11G|gpu_2080Ti_11G|gpu_P100_16G|gpu_RTX_A5000_24G|gpu_A40_45G|gpu_L40S_45G"

set -uo pipefail
umask 002

PROJECT_DIR="/work/cvcs2026/ai_interior_design/CVCS-AI-Assisted-ID"
cd "$PROJECT_DIR" || exit 1

export PERQUERY_OUT="${PERQUERY_OUT:-results/perquery/graph_partial_valid}"
# Solo `random`: e' l'unica strategia che entra nel criterio § 23.
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
  # shellcheck disable=SC2086  # $RUN va splittato in <target> <variante>
  bash scripts/graph/04_eval_gnn.sh $RUN
  RC=$?
  [ $RC -ne 0 ] && { echo "!! FALLITA: '$RUN' (rc=$RC)" >&2; FAILED=$((FAILED + 1)); }
done

echo ""
echo "=== completato: $(date) | run fallite: $FAILED ==="
ls -la "$PERQUERY_OUT"

[ $FAILED -eq 0 ] || exit 1
