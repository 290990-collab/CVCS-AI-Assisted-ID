#!/bin/bash
# scripts/evaluation/07_perquery_graph_test.sh
# Il TEST del ramo graph, per UNA sola configurazione gia' scelta sul valid.
# Gemello di 03_perquery_vision_test.sh (protocollo B, 11 set 2026).
#
# Perche' uno script a parte: lo split del ramo graph arriva da
# configs/graph_retrieval.yaml (`split: valid`, dove si sceglie). Qui lo si
# forza a `test` via GRAPH_EXTRA_FLAGS, senza toccare il YAML: cosi' nessuna run
# sul valid puo' finire per errore sul test, e viceversa.
#
# NON duplica il ponte YAML->flag: delega a scripts/graph/04_eval_gnn.sh (come
# 04 e 06 di questa cartella). La gallery condivisa arriva dal YAML: a parita' di
# seed e split le query sono le STESSE del vision (appaiamento cross-ramo).
#
# Argomenti OBBLIGATORI: niente modalita' "tutti" (sul test non si sceglie).
#
# Uso:
#   sbatch scripts/evaluation/07_perquery_graph_test.sh <target> <variante> [full|partial|both]
#   sbatch scripts/evaluation/07_perquery_graph_test.sh gat asym            # full + partial
#   sbatch scripts/evaluation/07_perquery_graph_test.sh hist - full          # baseline training-free
#   ⚠️ <target> = basename del YAML (gcn | gat | graph_sage | hist), non la chiave `sage`.
#   Env: PERQUERY_OUT (default results/perquery/graph_test_B)
#
# Output: $PERQUERY_OUT/graph_<enc>_<var>_full_test.npz
#         $PERQUERY_OUT/graph_<enc>_<var>_partial-random-f<frac>_test.npz

#SBATCH --job-name="ev_07_perquery_graph_test"
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=all_usr_prod
#SBATCH --account=cvcs2026
# Esclude i nodi Blackwell (sm_120): stessa whitelist di 05/06.
#SBATCH --constraint="gpu_RTX5000_16G|gpu_2080_11G|gpu_2080Ti_11G|gpu_P100_16G|gpu_RTX_A5000_24G|gpu_A40_45G|gpu_L40S_45G"

set -uo pipefail
umask 002

PROJECT_DIR="/work/cvcs2026/ai_interior_design/CVCS-AI-Assisted-ID"
cd "$PROJECT_DIR" || exit 1

TARGET="${1:-}"
VARIANT="${2:-}"
MODE="${3:-both}"

if [ -z "$TARGET" ] || [ -z "$VARIANT" ]; then
  echo "!! ERRORE: servono <target> e <variante> (per la baseline: 'hist -')." >&2
  echo "   uso: sbatch $0 <gcn|gat|graph_sage|hist> <variante|-> [full|partial|both]" >&2
  echo "   La configurazione va scelta PRIMA sul valid." >&2
  exit 1
fi
case "$MODE" in
  full|partial|both) ;;
  *) echo "!! ERRORE: modalita' '$MODE' non riconosciuta (attese: full | partial | both)" >&2
     exit 1 ;;
esac

# Argomenti per 04_eval_gnn.sh: la baseline non ha variante.
if [ "$TARGET" = "hist" ]; then RUN_ARGS=("hist"); else RUN_ARGS=("$TARGET" "$VARIANT"); fi

export PERQUERY_OUT="${PERQUERY_OUT:-results/perquery/graph_test_B}"
export GRAPH_EXTRA_FLAGS="--split test"
mkdir -p logs "$PERQUERY_OUT"

echo "=== $(date) | TEST (una volta sola) ramo graph | ${RUN_ARGS[*]} | $MODE ==="
echo "per-query -> $PERQUERY_OUT"
grep -n "^gallery_names" configs/graph_retrieval.yaml
echo "⚠️  Se questa non e' la configurazione scelta (e pre-registrata) sul valid,"
echo "    fermati: stai selezionando sul test."
nvidia-smi

FAILED=0

if [ "$MODE" = "full" ] || [ "$MODE" = "both" ]; then
  echo ""
  echo "########  TEST graph FULL: ${RUN_ARGS[*]}  ########"
  GRAPH_PARTIAL_FLAGS="" bash scripts/graph/04_eval_gnn.sh "${RUN_ARGS[@]}"
  RC=$?; [ $RC -ne 0 ] && { echo "!! FALLITA: full (rc=$RC)" >&2; FAILED=$((FAILED + 1)); }
fi

if [ "$MODE" = "partial" ] || [ "$MODE" = "both" ]; then
  echo ""
  echo "########  TEST graph PARTIAL: ${RUN_ARGS[*]}  ########"
  # Solo `random`, come 06: e' l'unica strategia del criterio di robustezza.
  GRAPH_PARTIAL_FLAGS="--partial --partial-strategies random --partial-fractions 0.0 0.25 0.5 0.75" \
    bash scripts/graph/04_eval_gnn.sh "${RUN_ARGS[@]}"
  RC=$?; [ $RC -ne 0 ] && { echo "!! FALLITA: partial (rc=$RC)" >&2; FAILED=$((FAILED + 1)); }
fi

echo ""
echo "=== completato: $(date) | run fallite: $FAILED ==="
ls -la "$PERQUERY_OUT"
echo ""
echo "Confronto di robustezza col vision (stesse query, stesse stanze tolte):"
echo "  python -m src.evaluation.robustness_auc compare --split test \\"
echo "      --a results/perquery/vision_test_B/vision_<enc>_<pool>_<tag> --b $PERQUERY_OUT/graph_<enc>_<var>"

[ $FAILED -eq 0 ] || exit 1
