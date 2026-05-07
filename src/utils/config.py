from pathlib import Path

import yaml


def load_config_file(path: str | Path) -> dict:
    """Optional helper for scripts not using Hydra."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)