# Shared settings for the vision jobs (source from each script): model list, valid poolings per encoder, resolutions.

PROJECT_DIR="/work/cvcs2026/ai_interior_design/CVCS-AI-Assisted-ID"

# Models: argument $1 (single) or all 8.
# Also used by scripts/evaluation/02_perquery_vision_valid.sh, which runs the whole list without arguments:
# adding a model widens that grid (56 runs with the first 5 encoders, 80 with tipsv2, 92 with pecore, 104 with pespatial);
# pass the new encoder explicitly to evaluate it alone.
select_models() {
  if [ -n "$1" ]; then echo "$1"; else echo "dinov2 dinov3 siglip2 radio ijepa tipsv2 pecore pespatial"; fi
}

# Valid poolings per encoder. I-JEPA has no CLS token: its natural pooling is already the patch mean, so "mean" is excluded.
poolings_for() {
  case "$1" in
    ijepa) echo "natural gem" ;;
    *)     echo "natural gem mean" ;;
  esac
}

# Resolutions per encoder; "native" = configs/vision_models/<model>.yaml preset (one resolution, variant = pooling).
# TIPSv2 is the only encoder with native resolution != 224 (448, ViT-B/14); both are needed, so the resolution
# enters the variant name (else both would write to embeddings/vision/tipsv2/<variant>/).
resolutions_for() {
  case "$1" in
    tipsv2) echo "448 224" ;;
    *)      echo "native" ;;
  esac
}

# Effective resolutions: explicit argument (one job per resolution) or all of the model.
#   $1 = model, $2 = requested resolution (optional)
select_resolutions() {
  if [ -n "${2:-}" ]; then echo "$2"; else resolutions_for "$1"; fi
}

# Variant name = subfolder of embeddings/vision/<model>/; native resolution keeps the bare pooling name.
#   $1 = pooling, $2 = resolution
variant_for() {
  if [ "$2" = "native" ]; then echo "$1"; else echo "$1$2"; fi
}

# Python overrides for a non-native resolution (empty if native).
# Batch shrinks with resolution: without xformers attention materialises [B, heads, N, N] (memory ~ B*N^2);
# at 448 there are 1026 tokens vs 258 at 224 (~16x), and the config batch 256 OOMs on an 11G card.
#   $1 = resolution
res_flags() {
  if [ "$1" = "native" ]; then
    echo ""
  elif [ "$1" -ge 384 ] 2>/dev/null; then
    echo "model.kwargs.image_size=$1 retrieval.batch_size=32"
  else
    echo "model.kwargs.image_size=$1"
  fi
}

# Log label: "dinov2/natural" or "tipsv2/natural@448".
#   $1 = model, $2 = pooling, $3 = resolution
label_for() {
  if [ "$3" = "native" ]; then echo "$1/$2"; else echo "$1/$2@$3"; fi
}
