#!/bin/bash
# scripts/evaluation/04_perquery_graph.sh
# FASE A.3 sul ramo graph — "strada breve": si rilanciano le valutazioni gia'
# note, con la persistenza per-query attiva.
#
# Perche' qui la strada breve E' quella giusta: nel ramo graph la variante non e'
# scelta sul test ma sul valid, dalla `RetrievalProbe` dentro il training
# (src/graph/training/train_gnn.py:174-181). Il rilievo A1 non lo tocca, quindi
# non serve nessuna griglia su valid: bastano i per-query delle varianti gia'
# selezionate, e da li' i confronti appaiati diventano possibili.
#
# NON duplica il ponte YAML->flag: delega a scripts/graph/04_eval_gnn.sh, che e'
# l'unico posto in cui i flag architetturali vengono derivati dal YAML del
# training (se divergessero, il checkpoint non si ricaricherebbe — rilievo B5).
# L'unica cosa che questo script aggiunge e' la variabile PERQUERY_OUT.
#
# ⚠️ Lo split arriva da configs/graph_retrieval.yaml (dal 10 set: `valid`).
#    Per il TEST del protocollo B usare 07_perquery_graph_test.sh.
#
# Uso:
#   sbatch scripts/evaluation/04_perquery_graph.sh            # baseline + le 3 vincenti + le 3 base
#   sbatch scripts/evaluation/04_perquery_graph.sh gcn tau02  # una sola coppia encoder/variante
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
# Esclude i nodi Blackwell (sm_120): 04_eval_gnn.sh e' chiamato con `bash`, quindi
# vale l'header di QUESTO script (stessa whitelist di 05/06).
#SBATCH --constraint="gpu_RTX5000_16G|gpu_2080_11G|gpu_2080Ti_11G|gpu_P100_16G|gpu_RTX_A5000_24G|gpu_A40_45G|gpu_L40S_45G"

set -uo pipefail
umask 002

PROJECT_DIR="/work/cvcs2026/ai_interior_design/CVCS-AI-Assisted-ID"
cd "$PROJECT_DIR" || exit 1
mkdir -p logs

# Letta da scripts/graph/04_eval_gnn.sh e trasformata in `--perquery-out`.
# Override via env (10 set 2026): lo split arriva da configs/graph_retrieval.yaml
# (oggi `valid`), quindi per il full sul valid si usa una cartella dedicata.
export PERQUERY_OUT="${PERQUERY_OUT:-results/perquery/graph_test}"

# Coppie <target> <variante>. `hist` e' la baseline training-free (nessuna
# variante). Le altre sono le vincenti misurate piu' la rispettiva `base`: il
# confronto vincente-vs-base e' appaiato per costruzione (stessa gallery, stesse
# query) ed e' quello che dice se il guadagno di ogni ablation e' reale.
# ⚠️ `graph_sage` e' il basename del YAML; la chiave del registry e' `sage`.
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
  # shellcheck disable=SC2086  # $RUN va splittato in <target> <variante>
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
