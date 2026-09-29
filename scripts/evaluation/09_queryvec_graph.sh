#!/bin/bash
# scripts/evaluation/09_queryvec_graph.sh
# LATE FUSION (status.md §49) — graph branch: per-query files + vectors of the
# damaged queries (contract qvec/1), from the SAME forward. Config fixed by the
# pre-registration: gat / asymrob, damage `random` f = 0.0 0.25 0.5 0.75
# (the same rooms as the vision `nowalls_random`: same rng `seed + qi`).
#
# Delegates to scripts/graph/04_eval_gnn.sh (the YAML->flag bridge stays in ONE
# place, as 06/07) and does NOT touch 06/07: the three env variables read by 04
# are set here to fixed values, so a stale PERQUERY_OUT / GRAPH_EXTRA_FLAGS in
# the user's environment cannot redirect the output. 07 is not reused because
# it hard-codes GRAPH_EXTRA_FLAGS="--split test" (the qvec flag would be lost).
# Output folders (whitelisted below):
#   results/perquery/fusion_branches_<split>   (per-query of this job, used by C3)
#   results/queryvec/<split>                   (qvec/1)
#
# ⚠️ graph_evaluate REWRITES embeddings/graph/gat/asymrob/{embeddings.npy,names.json}.
#    The qvec meta pins their sha1: do not run another gat/asymrob evaluation
#    between this job and 10_late_fusion.sh, or the fusion stops (explicitly).
# ⚠️ 04 exits 0 even when python fails: the files are checked at the end.
#
# Use:
#   sbatch scripts/evaluation/09_queryvec_graph.sh valid
#   sbatch scripts/evaluation/09_queryvec_graph.sh test       # ONLY in the pre-registered test group
#   FORCE=1 sbatch ...                                        # overwrite files already there
#
# Output (4 + 4 files):
#   results/perquery/fusion_branches_<split>/graph_gat_asymrob_partial-random-f<f>_<split>.npz
#   results/queryvec/<split>/graph_gat_asymrob_partial-random-f<f>_<split>.npz

#SBATCH --job-name="ev_09_queryvec_graph"
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

SPLIT="${1:-}"
case "$SPLIT" in
  valid|test) ;;
  *) echo "!! ERRORE: serve lo split come primo argomento (valid | test), ricevuto '$SPLIT'" >&2
     exit 1 ;;
esac

# Fixed config of the pre-registration (§49).
TARGET=gat
VARIANT=asymrob
TAG="graph_${TARGET}_${VARIANT}"
FRACTIONS="0.0 0.25 0.5 0.75"

PERQUERY_OUT="results/perquery/fusion_branches_${SPLIT}"
QVEC_DIR="results/queryvec/${SPLIT}"
for D in "$PERQUERY_OUT" "$QVEC_DIR"; do
  case "$D" in
    results/perquery/fusion_branches_valid|results/perquery/fusion_branches_test|results/queryvec/valid|results/queryvec/test) ;;
    *) echo "!! ERRORE: cartella di output non ammessa: '$D' (solo fusion_branches_* e queryvec/*)" >&2
       exit 1 ;;
  esac
done

EXISTING=$(ls "$QVEC_DIR"/${TAG}_partial-random-f*_"${SPLIT}".npz 2>/dev/null | wc -l)
if [ "$EXISTING" -gt 0 ] && [ "${FORCE:-0}" != "1" ]; then
  echo "!! ERRORE: $EXISTING file qvec gia' presenti in $QVEC_DIR per $TAG: rilancia con FORCE=1 se voluto" >&2
  exit 1
fi

# The three variables read by 04_eval_gnn.sh, ALL fixed here (env values overwritten).
# `--split` explicit also on valid: argparse keeps the last one, after the YAML's.
export PERQUERY_OUT
export GRAPH_PARTIAL_FLAGS="--partial --partial-strategies random --partial-fractions $FRACTIONS"
export GRAPH_EXTRA_FLAGS="--split $SPLIT --query-vectors-out $QVEC_DIR"
mkdir -p logs "$PERQUERY_OUT" "$QVEC_DIR"

echo "=== $(date) | LATE FUSION — graph query vectors | split $SPLIT | $TAG ==="
echo "per-query -> $PERQUERY_OUT | qvec -> $QVEC_DIR"
echo "partial: $GRAPH_PARTIAL_FLAGS | extra: $GRAPH_EXTRA_FLAGS"
grep -n "^gallery_names" configs/graph_retrieval.yaml
[ "$SPLIT" = "test" ] && echo "⚠️  TEST: solo nel gruppo di job pre-registrato, con alpha* gia' fissato sul valid."

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
echo "PROSSIMO PASSO (a 08 e 09 finiti): sbatch scripts/evaluation/10_late_fusion.sh $SPLIT"
