#!/bin/bash
# LayoutGKN full comparison: one GPU job, <= 24 h.
# Trains GKN-asym on the whole train split (<= 300 epochs or 20 h; checkpoint = best epoch on the robustness
# probe), evaluates on our protocol (full gallery, 2000 valid queries, f 0/0.25/0.5/0.75 + whole plan), also with
# random weights, and compares paired with W seed 42 (gat/comb).
#
# Writes (new) embeddings/competitors/layoutgkn/asym/, results/perquery/competitor_layoutgkn/asym/,
# results/competitors/layoutgkn/full/ (full.json); refuses to start if any exists.
# Needs torch_scatter, grakel, shapely in ~/floorplan-env and the upstream code (commit 395dc92).
#
# Usage:
#   sbatch scripts/competitors/02_layoutgkn_full.sh
# Rollback: rm -r embeddings/competitors/layoutgkn/asym results/perquery/competitor_layoutgkn/asym results/competitors/layoutgkn/full

#SBATCH --job-name="lgkn_02_full"
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=24:00:00
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

for D in embeddings/competitors/layoutgkn/asym results/perquery/competitor_layoutgkn/asym results/competitors/layoutgkn/full; do
  if [ -e "$D" ]; then echo "!! $D esiste già: niente viene sovrascritto" >&2; exit 1; fi
done
python -c "import torch_scatter, grakel, shapely" || { echo "!! mancano torch_scatter/grakel/shapely in ~/floorplan-env" >&2; exit 1; }

echo "=== inizio: $(date) | nodo $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) ==="
python -m src.competitors.layoutgkn.full
RC=$?
echo "=== completato: $(date) | rc=$RC ==="
exit $RC
