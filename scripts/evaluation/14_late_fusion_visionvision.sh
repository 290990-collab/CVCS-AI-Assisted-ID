#!/bin/bash
# Control "different models, same information", CPU job: fuses two vision encoders on the same image,
# pespatial/gem/whiten (gamma=1, from 08) + radio/natural/whiten (gamma=0, from 13), each whitened with its own
# job's parameters, for gamma in {0, 0.1, ..., 1}, damaged plans (f = 0.0 0.25 0.5 0.75) and full plan.
# Same method as 10. No GPU. VALID only.
#
# Inputs and output are fixed (no env overrides, no ALPHAS_FROM):
#   results/queryvec/valid/vision_pespatial_gem_whiten-train_*   (08)
#   results/queryvec/valid/vision_radio_natural_whiten-train_*    (13)
#   gallery vectors: the paths pinned (sha1) in the qvec meta
#   -> results/perquery/fusion_visionvision_valid/fusion_a<gamma>_{partial-nowalls-random-f<f>,full}_valid.npz
# late_fusion refuses a folder that already holds fused files of another pair.
#
# Usage:
#   sbatch scripts/evaluation/14_late_fusion_visionvision.sh
#   FORCE=1 sbatch scripts/evaluation/14_late_fusion_visionvision.sh   # overwrite fused files already there

#SBATCH --job-name="ev_14_late_fusion_visionvision"
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
  echo "!! ERRORE: nessun argomento ammesso (solo valid, status.md §51), ricevuto '$*'" >&2
  exit 1
fi
if [ -n "${ALPHAS_FROM:-}" ]; then
  echo "!! ERRORE: il controllo usa la griglia intera di gamma: togli ALPHAS_FROM dall'ambiente" >&2
  exit 1
fi

SPLIT=valid
VISION_QVEC="results/queryvec/valid/vision_pespatial_gem_whiten-train"
VISION2_QVEC="results/queryvec/valid/vision_radio_natural_whiten-train"
OUT="results/perquery/fusion_visionvision_valid"
case "$OUT" in
  results/perquery/fusion_visionvision_valid) ;;
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

echo "=== $(date) | CONTROL §51 vision + vision | split $SPLIT | out $OUT | gamma: griglia 0..1 passo 0.1 ==="
python -m src.evaluation.late_fusion run --pair vision-vision --split "$SPLIT" \
  --gallery-names results/shared_gallery.json \
  --vision-qvec "$VISION_QVEC" --vision2-qvec "$VISION2_QVEC" \
  --fractions 0.0 0.25 0.5 0.75 \
  --out "$OUT"
RC=$?

echo ""
echo "=== completato: $(date) | rc=$RC ==="
ls "$OUT" | wc -l
[ $RC -eq 0 ] || exit 1
echo ""
echo "PROSSIMO PASSO (CPU): controlli -> selezione di gamma -> complementarita' (comandi in COMANDI.md,"
echo "  sezione «Controllo modelli diversi, stessa informazione (vision + vision)»)"
