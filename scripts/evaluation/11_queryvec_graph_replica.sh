#!/bin/bash
# Ensemble-effect control: the graph replica `gat/asymrobrep`, per-query files + damaged-query vectors
# (qvec/1) from the same forward. Twin of 09 (same recipe, same removed rooms `random`
# f = 0.0 0.25 0.5 0.75), VALID only (interpretive control, the test is not consumed).
#
# Delegates to scripts/graph/04_eval_gnn.sh (asymrobrep is in its KNOWN_VARIANTS, no architecture flag);
# the three env variables read by 04 are fixed here.
# Output folders (fixed, whitelisted):
#   results/perquery/fusion_branches_valid   (per-query of this job, used by C3)
#   results/queryvec/valid                   (qvec/1)
#
# Warning: graph_evaluate rewrites embeddings/graph/gat/asymrobrep/{embeddings.npy,names.json} (not asymrob's):
# no re-evaluation of gat/asymrobrep before 12_late_fusion_graphgraph.sh.
# Warning: 04 exits 0 even when python fails: the files are checked at the end.
#
# Usage:
#   sbatch scripts/evaluation/11_queryvec_graph_replica.sh
#   FORCE=1 sbatch scripts/evaluation/11_queryvec_graph_replica.sh   # overwrite files already there
#
# Output (4 + 4 files):
#   results/perquery/fusion_branches_valid/graph_gat_asymrobrep_partial-random-f<f>_valid.npz
#   results/queryvec/valid/graph_gat_asymrobrep_partial-random-f<f>_valid.npz

#SBATCH --job-name="ev_11_queryvec_graph_replica"
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
# Excludes the Blackwell nodes (sm_120): same whitelist as 05/06/07.
#SBATCH --constraint="gpu_RTX5000_16G|gpu_2080_11G|gpu_2080Ti_11G|gpu_P100_16G|gpu_RTX_A5000_24G|gpu_A40_45G|gpu_L40S_45G"

set -uo pipefail
umask 002

PROJECT_DIR="/work/cvcs2026/ai_interior_design/CVCS-AI-Assisted-ID"
cd "$PROJECT_DIR" || exit 1

# valid only: no split argument
if [ $# -gt 0 ]; then
  echo "!! ERRORE: nessun argomento ammesso (solo valid, status.md §50), ricevuto '$*'" >&2
  exit 1
fi
SPLIT=valid

# Fixed config.
TARGET=gat
VARIANT=asymrobrep
TAG="graph_${TARGET}_${VARIANT}"
FRACTIONS="0.0 0.25 0.5 0.75"

PERQUERY_OUT="results/perquery/fusion_branches_${SPLIT}"
QVEC_DIR="results/queryvec/${SPLIT}"
for D in "$PERQUERY_OUT" "$QVEC_DIR"; do
  case "$D" in
    results/perquery/fusion_branches_valid|results/queryvec/valid) ;;
    *) echo "!! ERRORE: cartella di output non ammessa: '$D' (solo fusion_branches_* e queryvec/*)" >&2
       exit 1 ;;
  esac
done

EXISTING=$(ls "$QVEC_DIR"/${TAG}_partial-random-f*_"${SPLIT}".npz \
              "$PERQUERY_OUT"/${TAG}_partial-random-f*_"${SPLIT}".npz 2>/dev/null | wc -l)
if [ "$EXISTING" -gt 0 ] && [ "${FORCE:-0}" != "1" ]; then
  echo "!! ERRORE: $EXISTING file qvec/per-query gia' presenti per $TAG: rilancia con FORCE=1 se voluto" >&2
  exit 1
fi

# The three variables read by 04_eval_gnn.sh, all fixed here (env values overwritten).
# `--split` explicit also on valid: argparse keeps the last one, after the YAML's.
export PERQUERY_OUT
export GRAPH_PARTIAL_FLAGS="--partial --partial-strategies random --partial-fractions $FRACTIONS"
export GRAPH_EXTRA_FLAGS="--split $SPLIT --query-vectors-out $QVEC_DIR"
mkdir -p logs "$PERQUERY_OUT" "$QVEC_DIR"

echo "=== $(date) | ENSEMBLE CONTROL — graph REPLICA query vectors | split $SPLIT | $TAG ==="
echo "per-query -> $PERQUERY_OUT | qvec -> $QVEC_DIR"
echo "partial: $GRAPH_PARTIAL_FLAGS | extra: $GRAPH_EXTRA_FLAGS"
grep -n "^gallery_names" configs/graph_retrieval.yaml

START_MARK=$(mktemp)
bash scripts/graph/04_eval_gnn.sh "$TARGET" "$VARIANT"
RC=$?

MISSING=0
for F in $FRACTIONS; do
  for P in "$PERQUERY_OUT/${TAG}_partial-random-f${F}_${SPLIT}.npz" \
           "$QVEC_DIR/${TAG}_partial-random-f${F}_${SPLIT}.npz"; do
    if [ ! -f "$P" ] || [ ! "$P" -nt "$START_MARK" ]; then
      echo "!! MANCANTE o vecchio: $P" >&2
      MISSING=$((MISSING + 1))
    fi
  done
done
rm -f "$START_MARK"

echo ""
echo "=== completato: $(date) | rc=$RC | file mancanti: $MISSING ==="
ls -la "$QVEC_DIR"
[ $RC -eq 0 ] && [ $MISSING -eq 0 ] || exit 1
echo "PROSSIMO PASSO: sbatch scripts/evaluation/12_late_fusion_graphgraph.sh"
