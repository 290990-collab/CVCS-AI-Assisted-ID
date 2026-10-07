#!/bin/bash
# Ensemble-effect control, CPU job: fuses two trainings of the same graph, gat/asymrob (beta=1, from 09) +
# gat/asymrobrep (beta=0, from 11), for beta in {0, 0.1, ..., 1}, damaged plans (f = 0.0 0.25 0.5 0.75) and
# full plan. Same method as 10 (vision + graph), no whitening. No GPU. VALID only.
#
# Inputs and output are fixed (no env overrides, no ALPHAS_FROM):
#   results/queryvec/valid/graph_gat_asymrob_*      (09)
#   results/queryvec/valid/graph_gat_asymrobrep_*   (11)
#   gallery vectors: the paths pinned (sha1) in the qvec meta
#   -> results/perquery/fusion_graphgraph_valid/fusion_a<beta>_{partial-random-f<f>,full}_valid.npz
# late_fusion refuses a folder that already holds vision + graph fused files.
#
# Usage:
#   sbatch scripts/evaluation/12_late_fusion_graphgraph.sh
#   FORCE=1 sbatch scripts/evaluation/12_late_fusion_graphgraph.sh   # overwrite fused files already there

#SBATCH --job-name="ev_12_late_fusion_graphgraph"
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=all_usr_prod
#SBATCH --account=cvcs2026

set -uo pipefail
umask 002

PROJECT_DIR="/work/cvcs2026/ai_interior_design/CVCS-AI-Assisted-ID"
source ~/floorplan-env/bin/activate
cd "$PROJECT_DIR" || exit 1

if [ $# -gt 0 ]; then
  echo "!! ERRORE: nessun argomento ammesso (solo valid, status.md §50), ricevuto '$*'" >&2
  exit 1
fi
if [ -n "${ALPHAS_FROM:-}" ]; then
  echo "!! ERRORE: il controllo usa la griglia intera di beta: togli ALPHAS_FROM dall'ambiente" >&2
  exit 1
fi

SPLIT=valid
GRAPH_QVEC="results/queryvec/valid/graph_gat_asymrob"
GRAPH2_QVEC="results/queryvec/valid/graph_gat_asymrobrep"
OUT="results/perquery/fusion_graphgraph_valid"
case "$OUT" in
  results/perquery/fusion_graphgraph_valid) ;;
  *) echo "!! ERRORE: cartella di output non ammessa: '$OUT'" >&2; exit 1 ;;
esac

EXISTING=$(ls "$OUT"/fusion_a*_"${SPLIT}".npz 2>/dev/null | wc -l)
if [ "$EXISTING" -gt 0 ] && [ "${FORCE:-0}" != "1" ]; then
  echo "!! ERRORE: $EXISTING file fusi gia' presenti in $OUT: rilancia con FORCE=1 se voluto" >&2
  exit 1
fi
mkdir -p logs "$OUT"

# FAISS/BLAS threads = the CPUs of the job.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "=== $(date) | ENSEMBLE CONTROL graph + graph | split $SPLIT | out $OUT | beta: griglia 0..1 passo 0.1 ==="
python -m src.evaluation.late_fusion run --pair graph-graph --split "$SPLIT" \
  --gallery-names results/shared_gallery.json \
  --graph-qvec "$GRAPH_QVEC" --graph2-qvec "$GRAPH2_QVEC" \
  --fractions 0.0 0.25 0.5 0.75 \
  --out "$OUT"
RC=$?

echo ""
echo "=== completato: $(date) | rc=$RC ==="
ls "$OUT" | wc -l
[ $RC -eq 0 ] || exit 1
echo ""
echo "PROSSIMO PASSO (CPU): controlli -> selezione di beta -> complementarita' (comandi in COMANDI.md,"
echo "  sezione «Controllo effetto d'insieme (graph + graph)»)"
