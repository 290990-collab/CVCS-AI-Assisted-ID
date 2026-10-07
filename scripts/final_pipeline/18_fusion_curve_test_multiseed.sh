#!/bin/bash
# Multi-seed test: the reduced curve on the test on another training seed S (damage seed S). Same 10 tasks and
# code as 11_fusion_curve_test.sh (seed 42): gcn/ref, gat/t05, gat/ref, gat/t01, W, each fused with the frozen
# vision (tasks 0-4) and with the vision with head (tasks 5-9), weight fixed from the valid curve of the same
# seed ({0, w, 1}). Graph inputs from the multi-seed test evaluations (12_eval_graph_multiseed.sh test): launch
# after them. CPU only. Gate: TEST_PREREGISTERED_R2.
# Writes (new) results/final_pipeline/curve_test/s<S>/<frozen|head>/<enc>_<cfg>/fusion_test/.
#
# Usage:
#   sbatch --array=0-9 scripts/final_pipeline/18_fusion_curve_test_multiseed.sh 100042      # then 200042 300042
# Then: python -m src.evaluation.fusion_curve_test summary --seed 100042

#SBATCH --job-name="rg_18_curve_test_r2"
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
case "$S" in 100042|200042|300042) ;; *) echo "!! uso: 18_fusion_curve_test_multiseed.sh <100042|200042|300042> (indice da --array)" >&2; exit 1 ;; esac
[[ "$I" =~ ^[0-9]+$ ]] || { echo "!! indice mancante: usa sbatch --array=0-9" >&2; exit 1; }
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export CUDA_VISIBLE_DEVICES=""

echo "=== $(date) | curva ridotta sul TEST, secondo giro §65 D | seed $S | task $I ==="
python -m src.evaluation.fusion_curve_test run --index "$I" --seed "$S"
RC=$?
echo "=== completato: $(date) | rc=$RC ==="
exit $RC
