#!/bin/bash
# LayoutGKN on the test split: one GPU job, no training.
# Uses the checkpoint of the full comparison (epoch 297, sha1 pinned in src/competitors/layoutgkn/test_eval.py)
# on the 2000 test queries of W seed 42 with the same removed rooms (checked query by query),
# f 0/0.25/0.5/0.75 + whole plan; paired with W.
#
# Writes (new) results/perquery/competitor_layoutgkn/asym_test/, results/competitors/layoutgkn/test/ (test.json);
# refuses to start if any exists or if results/final_pipeline/TEST_PREREGISTERED is missing.
#
# Usage:
#   sbatch scripts/competitors/03_layoutgkn_test.sh
# Rollback: rm -r results/perquery/competitor_layoutgkn/asym_test results/competitors/layoutgkn/test

#SBATCH --job-name="lgkn_03_test"
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=all_usr_prod
#SBATCH --account=cvcs2026
#SBATCH --constraint="gpu_RTX5000_16G|gpu_2080_11G|gpu_2080Ti_11G|gpu_P100_16G|gpu_RTX_A5000_24G|gpu_A40_45G|gpu_L40S_45G"

set -uo pipefail
umask 002

PROJECT_DIR="/work/cvcs2026/ai_interior_design/CVCS-AI-Assisted-ID"
source ~/floorplan-env/bin/activate
cd "$PROJECT_DIR" || exit 1
mkdir -p logs

[ -e results/final_pipeline/TEST_PREREGISTERED ] || { echo "!! manca results/final_pipeline/TEST_PREREGISTERED" >&2; exit 1; }
for D in results/perquery/competitor_layoutgkn/asym_test results/competitors/layoutgkn/test; do
  if [ -e "$D" ]; then echo "!! $D esiste già: niente viene sovrascritto" >&2; exit 1; fi
done
python -c "import torch_scatter, grakel, shapely" || { echo "!! mancano torch_scatter/grakel/shapely in ~/floorplan-env" >&2; exit 1; }

echo "=== inizio: $(date) | nodo $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) ==="
python -m src.competitors.layoutgkn.test_eval
RC=$?
echo "=== completato: $(date) | rc=$RC ==="
exit $RC
