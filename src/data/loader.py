from pathlib import Path
from typing import Any


def load_processed_dataset(processed_root: str | Path) -> list[Any]:
    """Load processed floor-plan graphs.

    This is a stub so the project is runnable before the real dataset arrives.
    """
    root = Path(processed_root)
    if not root.exists():
        print(f"Processed dataset path does not exist yet: {root}")
        return []

    # TODO: read serialized graph objects and convert to PyG Data instances
    return []
