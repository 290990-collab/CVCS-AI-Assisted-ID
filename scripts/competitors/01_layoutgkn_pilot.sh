#!/bin/bash
# LayoutGKN feasibility pilot: one GPU job, <= 8 h.
# Trains on 10,000 train plans (<= 100 epochs or ~6.5 h), evaluates on a 5,000-plan sub-gallery,
# plus references (random weights, hist, W seed 42) and costs on the full gallery.
#
# Writes (new) embeddings/competitors/layoutgkn/pilot/ and results/competitors/layoutgkn/pilot/
# (pilot.json + per-query); refuses to start if either exists.
# Needs torch_scatter, grakel, shapely in ~/floorplan-env and the upstream code in
# /work/cvcs2026/ai_interior_design/external/LayoutGKN (commit 395dc92).
#
# Usage:
#   sbatch scripts/competitors/01_layoutgkn_pilot.sh
# Rollback: rm -r embeddings/competitors/layoutgkn/pilot results/competitors/layoutgkn/pilot

#SBATCH --job-name="lgkn_01_pilot"
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
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

for D in embeddings/competitors/layoutgkn/pilot results/competitors/layoutgkn/pilot; do
  if [ -e "$D" ]; then echo "!! $D esiste già: niente viene sovrascritto" >&2; exit 1; fi
done
python -c "import torch_scatter, grakel, shapely" || { echo "!! mancano torch_scatter/grakel/shapely in ~/floorplan-env" >&2; exit 1; }

echo "=== inizio: $(date) | nodo $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) ==="
python -m src.competitors.layoutgkn.pilot
RC=$?
echo "=== completato: $(date) | rc=$RC ==="
exit $RC
