#!/bin/bash
# Exploratory: fusion gain as a function of graph strength. Each of the 37 graphs (seed 42, valid) fused
# with the frozen vision (tasks 0-36) and with the vision with head (tasks 37-73). Same fusion code as 03/07
# (`late_fusion.run`), grid alpha 0..1 step 0.2, removed rooms only (f 0.25/0.5/0.75). CPU only. Valid only.
#
# Inputs (read only): graph qvec/per-query of the 37 configs (02_eval_graph.sh), vision qvec/per-query of the
# main fusion and of the fusion with head.
# Writes (new) results/final_pipeline/curve/<frozen|head>/<enc>_<cfg>/fusion_valid/.
# Refuses a non-empty output folder.
#
# Usage:
#   sbatch --array=0-73%24 scripts/final_pipeline/09_fusion_curve.sh
# Then: python -m src.evaluation.fusion_curve summary

#SBATCH --job-name="rg_09_curve"
#SBATCH --output=logs/%x_%A_%a.log
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=02:00:00
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
mkdir -p logs

I="${SLURM_ARRAY_TASK_ID:-${1:-}}"
[[ "$I" =~ ^[0-9]+$ ]] || { echo "!! indice mancante: usa sbatch --array=0-73" >&2; exit 1; }
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export CUDA_VISIBLE_DEVICES=""

echo "=== $(date) | curva ESPLORATIVA §59.E | task $I ==="
python -m src.evaluation.fusion_curve run --index "$I"
RC=$?
echo "=== completato: $(date) | rc=$RC ==="
exit $RC
