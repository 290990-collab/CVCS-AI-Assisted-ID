#!/bin/bash
# Multi-seed round: the curve "fusion gain vs graph strength" on another training seed S (damage seed S).
# Same 74 tasks and code as 09_fusion_curve.sh (seed 42): 37 graphs x {frozen, with head}, alpha grid 0..1
# step 0.2, removed rooms only. W comes from the multi-seed evaluation (12_eval_graph_multiseed.sh valid,
# tasks 1-3): launch after it. CPU only, valid only.
# Writes (new) results/final_pipeline/curve/s<S>/<frozen|head>/<enc>_<cfg>/fusion_valid/.
#
# Usage:
#   sbatch --array=0-73%24 scripts/final_pipeline/14_fusion_curve_multiseed.sh 100042
# Then: python -m src.evaluation.fusion_curve summary --seed 100042

#SBATCH --job-name="rg_14_curve_r2"
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

S="${1:-}"
I="${SLURM_ARRAY_TASK_ID:-${2:-}}"
case "$S" in 100042|200042|300042) ;; *) echo "!! uso: 14_fusion_curve_multiseed.sh <100042|200042|300042> (indice da --array)" >&2; exit 1 ;; esac
[[ "$I" =~ ^[0-9]+$ ]] || { echo "!! indice mancante: usa sbatch --array=0-73" >&2; exit 1; }
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export CUDA_VISIBLE_DEVICES=""

echo "=== $(date) | curva, secondo giro §64 D | seed $S | task $I ==="
python -m src.evaluation.fusion_curve run --index "$I" --seed "$S"
RC=$?
echo "=== completato: $(date) | rc=$RC ==="
exit $RC
