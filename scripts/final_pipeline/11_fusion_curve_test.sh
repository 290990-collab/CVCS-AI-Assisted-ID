#!/bin/bash
# Reduced fusion curve on the test, seed 42. Five graphs (gcn/ref, gat/t05, gat/ref, gat/t01, W) fused with
# the frozen vision (tasks 0-4) and with the vision with head (tasks 5-9), weight fixed from the valid curve
# ({0, w, 1}), removed rooms f 0.25/0.5/0.75. Same fusion code as 09_fusion_curve.sh. CPU only.
# Gate: TEST_PREREGISTERED (graph_config_select test-gate).
#
# Writes (new) results/final_pipeline/curve_test/<frozen|head>/<enc>_<cfg>/fusion_test/.
#
# Usage:
#   sbatch --array=0-9 scripts/final_pipeline/11_fusion_curve_test.sh
# Then (login node): python -m src.evaluation.fusion_curve_test summary

#SBATCH --job-name="rg_11_curve_test"
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
[[ "$I" =~ ^[0-9]+$ ]] || { echo "!! indice mancante: usa sbatch --array=0-9" >&2; exit 1; }
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export CUDA_VISIBLE_DEVICES=""

echo "=== $(date) | curva ridotta sul TEST (§62 E) | task $I ==="
python -m src.evaluation.fusion_curve_test run --index "$I"
RC=$?
echo "=== completato: $(date) | rc=$RC ==="
exit $RC
