# Config YAML file loader

import yaml


# utils/config.py — approccio manuale, fragile
import yaml, os

def load_config(env="local"):
    with open(f"configs/{env}.yaml") as f:
        return yaml.safe_load(f)

cfg = load_config(os.getenv("ENV", "local"))
print(cfg["model"]["hidden_dim"])