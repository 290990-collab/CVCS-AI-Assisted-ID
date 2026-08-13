# scripts/graph/_common.sh
# Impostazioni condivise dai job del ramo grafi (da "source"-are in ogni script).
# Analogo a scripts/vision/_common.sh: centralizza PROJECT_DIR, la lista encoder e
# il "ponte" che traduce i YAML di configs/graph_models/ nei flag di argparse di
# src/graph/training/train_gnn.py.

PROJECT_DIR="/work/cvcs2026/ai_interior_design/CVCS-AI-Assisted-ID"

# Encoder da processare: l'argomento $1 (uno solo) oppure tutti e tre.
# I nomi sono i BASENAME dei file in configs/graph_models/ (graph_sage, non sage).
select_encoders() {
  if [ -n "$1" ]; then echo "$1"; else echo "gcn gat graph_sage"; fi
}

# Ponte YAML -> flag argparse.
# Legge configs/graph_models/<name>.yaml (via OmegaConf, come il ramo vision) e
# stampa la stringa di flag per train_gnn.py: ogni chiave `k: v` diventa `--k v`
# (underscore -> trattino). I tre booleani con flag speciali (store_false /
# store_true nell'argparse) sono gestiti a parte; ogni altro booleano viene saltato.
train_flags_from_yaml() {
  python - "$1" <<'PY'
import sys
from omegaconf import OmegaConf

cfg = OmegaConf.to_container(OmegaConf.load(sys.argv[1]), resolve=True) or {}

# Booleani mappati su flag non standard di train_gnn.py.
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
        continue  # nessun altro booleano previsto
    out += [f"--{k.replace('_', '-')}", str(v)]

print(" ".join(out))
PY
}
