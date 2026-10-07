#!/bin/bash
# Head v2 data preparation: data only, no training.
#   1. pairs: per train+valid plan, 6 views, damages in turn rooms removed (no walls) / crop / cover,
#      fraction U[0.25, 0.75] -> embeddings/vision/pespatial/gem/pairs_v2.npz (view cache in a new folder of _render_cache)
#   2. probe: the usual 1000 valid probe queries (disjoint from the 2000 eval queries),
#      three damages x f 0.25/0.5/0.75 -> embeddings/vision/pespatial/gem/probe_v2.npz
# No overwrites: pairs.npz and probe_partial.npz of the current head are untouched (refuses if pairs_v2.npz / probe_v2.npz exist).
#
# Uso:
#   sbatch scripts/vision/10_prepare_head_v2.sh

#SBATCH --job-name="vx_10_prep_head_v2"
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=18:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=all_usr_prod
#SBATCH --account=cvcs2026
#SBATCH --constraint="gpu_RTX5000_16G|gpu_2080_11G|gpu_2080Ti_11G|gpu_P100_16G|gpu_RTX_A5000_24G|gpu_A40_45G|gpu_L40S_45G"

set -uo pipefail
set -f      # the [a,b,c] overrides are not globs
umask 002

source ~/floorplan-env/bin/activate
cd /work/cvcs2026/ai_interior_design/CVCS-AI-Assisted-ID || exit 1
mkdir -p logs
export PYTHONUNBUFFERED=1

D=embeddings/vision/pespatial/gem
for f in "$D/pairs_v2.npz" "$D/probe_v2.npz"; do
  [ -e "$f" ] && { echo "!! $f esiste gia': niente sovrascritture" >&2; exit 1; }
done

COMMON="model.name=pespatial model.variant=gem model.kwargs.pooling=gem training.damage=nowalls_random"
V2="training.damage_mix=[nowalls_random,crop,patch] training.mask_fraction=[0.25,0.75] training.views_per_plan=6"

echo "=== $(date) | head v2: preparazione dati | $D ==="
nvidia-smi
echo "########  1/2 COPPIE v2  ########"
python -m src.vision.data.projection_pairs $COMMON $V2 training.pairs_file=pairs_v2.npz || exit 1
echo "########  2/2 PROBE v2  ########"
python -m src.vision.training.retrieval_probe $COMMON \
  "training.probe.damages=[nowalls_random,crop,patch]" training.probe.file=probe_v2.npz || exit 1
echo "=== completato: $(date) ==="
