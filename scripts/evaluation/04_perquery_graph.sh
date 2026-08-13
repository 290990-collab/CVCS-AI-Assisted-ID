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
# ⚠️ Valuta sul TEST (e' cio' che fa 04_eval_gnn.sh, `split` dal config
#    condiviso): sono le stesse righe gia' pubblicate, non una nuova selezione.
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

set -uo pipefail
umask 002

PROJECT_DIR="/work/cvcs2026/ai_interior_design/CVCS-AI-Assisted-ID"
cd "$PROJECT_DIR" || exit 1
mkdir -p logs results/perquery/graph_test

# Letta da scripts/graph/04_eval_gnn.sh e trasformata in `--perquery-out`.
export PERQUERY_OUT="results/perquery/graph_test"

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
echo "Il confronto vision<->graph richiede --allow-gallery-mismatch (67.405 vs"
echo "67.453) e resta non appaiato in senso stretto finche' non si restringe la"
echo "gallery all'inner join (fase B.3)."

[ $FAILED -eq 0 ] || exit 1
