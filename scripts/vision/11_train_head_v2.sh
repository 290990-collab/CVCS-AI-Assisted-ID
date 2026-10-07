#!/bin/bash
# Head v2: train the residual query-only head on pairs_v2.npz, epoch chosen on probe_v2.npz
# (1000 valid queries disjoint from the 2000 eval ones, mean of the three damage AUCs).
# Recipe: tau 0.07, batch 4096, lr 1e-3, wd 1e-4, seed 42, max 5000 epochs, patience 100.
# Writes only new files: embeddings/vision/pespatial/gem/head_v2.pt + head_v2_history.json (refuses if they exist).
#
# Usage:
#   sbatch scripts/vision/11_train_head_v2.sh
# Then: sbatch scripts/vision/12_eval_head_v2.sh valid   and   sbatch scripts/final_pipeline/15_queryvec_vision_head_v2.sh valid 42

#SBATCH --job-name="vx_11_head_v2"
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --partition=all_usr_prod
#SBATCH --account=cvcs2026
#SBATCH --constraint="gpu_RTX5000_16G|gpu_2080_11G|gpu_2080Ti_11G|gpu_P100_16G|gpu_RTX_A5000_24G|gpu_A40_45G|gpu_L40S_45G"

set -uo pipefail
umask 002
source ~/floorplan-env/bin/activate
cd /work/cvcs2026/ai_interior_design/CVCS-AI-Assisted-ID || exit 1
mkdir -p logs
export PYTHONUNBUFFERED=1

python -m src.evaluation.multiseed_runs gate --split valid || exit 1
D=embeddings/vision/pespatial/gem
for f in "$D/head_v2.pt" "$D/head_v2_history.json"; do
  [ -e "$f" ] && { echo "!! $f esiste gia': niente sovrascritture" >&2; exit 1; }
done
echo "=== $(date) | head v2: training ==="
nvidia-smi
python -m src.vision.training.train_head_v2
RC=$?
echo "=== completato: $(date) | rc=$RC ==="
exit $RC
