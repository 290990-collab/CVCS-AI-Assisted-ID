# Shared settings for the graph-branch jobs (source from each script).
# Centralises PROJECT_DIR, the encoder list and the YAML -> train_gnn.py flag bridge (cf. scripts/vision/_common.sh).

PROJECT_DIR="/work/cvcs2026/ai_interior_design/CVCS-AI-Assisted-ID"

# Encoders: argument $1 (single) or all three; names are basenames in configs/graph_models/ (graph_sage, not sage).
select_encoders() {
  if [ -n "$1" ]; then echo "$1"; else echo "gcn gat graph_sage"; fi
}

# YAML -> argparse flags for train_gnn.py: `k: v` becomes `--k v` (underscore -> dash).
# Booleans with special flags are handled apart; any other boolean is skipped.
train_flags_from_yaml() {
  python - "$1" <<'PY'
import sys
from omegaconf import OmegaConf

cfg = OmegaConf.to_container(OmegaConf.load(sys.argv[1]), resolve=True) or {}

# booleans mapped to non-standard flags
special = {
    "normalize":       lambda v: []              if v else ["--no-normalize"],
    "drop_self_loops": lambda v: []              if v else ["--keep-self-loops"],
    "raw_skip":        lambda v: ["--raw-skip"]  if v else [],
    "wandb":           lambda v: ["--wandb"]     if v else [],
}

out = []
for k, v in cfg.items():
    if k in special:
        out += special[k](bool(v))
        continue
    if isinstance(v, bool):
        continue  # no other booleans expected
    out += [f"--{k.replace('_', '-')}", str(v)]

print(" ".join(out))
PY
}
