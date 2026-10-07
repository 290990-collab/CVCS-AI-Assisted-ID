"""Shared gallery for the two branches: inner join of gallery names.

The frozen intersection (JSON with provenance and sha1) is computed once; runs read it.
Canonical order is alphabetical so both branches get the same `gallery_sha1`; canonical name is the file stem.
Restricting the gallery changes the numbers of both branches (48 plans fewer).

Usage:

    python -m src.evaluation.gallery_join \\
        --vision embeddings/vision/<model>/<pool>/image_paths.json \\
        --graph  embeddings/graph/<run>/names.json \\
        --out    results/shared_gallery.json

Then point `eval.gallery_names` at that file in both YAMLs.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from src.evaluation.perquery import gallery_sha1

SCHEMA_VERSION = 1


# --- names ---

def canonical_names(entries: list[str]) -> list[str]:
    """Canonical names (file stems) from PNG paths or bare names."""
    return [Path(e).stem for e in entries]


def read_gallery_names(path: str | Path) -> list[str]:
    """Canonical names from `image_paths.json` / `names.json`; ValueError on non-list or duplicates."""
    entries = json.loads(Path(path).read_text())
    if not isinstance(entries, list) or not all(isinstance(e, str) for e in entries):
        raise ValueError(
            f"{path}: atteso un JSON con una lista di stringhe "
            f"(image_paths.json o names.json)"
        )
    names = canonical_names(entries)
    if len(set(names)) != len(names):
        dupes = {n for n in names if names.count(n) > 1}
        raise ValueError(
            f"{path}: {len(names) - len(set(names))} nomi duplicati "
            f"(es. {sorted(dupes)[:3]}): il join per nome sarebbe ambiguo"
        )
    return names


# --- intersection ---

def compute_shared_names(vision_names: list[str], graph_names: list[str]) -> list[str]:
    """Names in both branches, alphabetical (makes `gallery_sha1` identical)."""
    shared = sorted(set(vision_names) & set(graph_names))
    if not shared:
        raise ValueError(
            "intersezione vuota: le due gallery non hanno nessun nome in comune. "
            "Controlla di aver passato i file dei due rami e non due volte lo stesso."
        )
    return shared


def restrict_rows(names: list[str], shared: list[str]) -> list[int]:
    """Row indices of `names` to keep, in the order of `shared`; KeyError if a shared name is missing."""
    row_of = {n: i for i, n in enumerate(names)}
    missing = [n for n in shared if n not in row_of]
    if missing:
        raise KeyError(
            f"{len(missing)} nomi della gallery condivisa non sono in questa "
            f"gallery (es. {missing[:3]}): file dell'inner join non compatibile"
        )
    return [row_of[n] for n in shared]


# --- artifact on disk ---

def write_shared_gallery(out_path: str | Path, shared: list[str],
                         sources: dict[str, dict]) -> dict:
    """Write the inner-join JSON with provenance; return the payload."""
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created": datetime.now().isoformat(timespec="seconds"),
        "n_shared": len(shared),
        "sha1": gallery_sha1(shared),
        "sources": sources,
        "names": shared,
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload))
    return payload


def load_shared_names(path: str | Path) -> list[str]:
    """Load inner-join names; ValueError if the stored sha1 does not match."""
    payload = json.loads(Path(path).read_text())
    names = payload["names"]
    expected = payload.get("sha1")
    if expected and gallery_sha1(names) != expected:
        raise ValueError(
            f"{path}: sha1 non corrispondente ({gallery_sha1(names)} != {expected}). "
            "Il file e' stato modificato: rigeneralo con python -m src.evaluation.gallery_join"
        )
    return names


# --- CLI ---

def parse_args():
    p = argparse.ArgumentParser(
        description="Inner join dei nomi fra le gallery dei due rami (fase B.3)."
    )
    p.add_argument("--vision", required=True,
                   help="image_paths.json del ramo vision")
    p.add_argument("--graph", required=True,
                   help="names.json del ramo graph")
    p.add_argument("--out", default="results/shared_gallery.json",
                   help="dove scrivere l'inner join (default: %(default)s)")
    return p.parse_args()


def main():
    args = parse_args()
    vision_names = read_gallery_names(args.vision)
    graph_names = read_gallery_names(args.graph)
    shared = compute_shared_names(vision_names, graph_names)

    payload = write_shared_gallery(
        args.out, shared,
        sources={
            "vision": {"path": str(args.vision), "n": len(vision_names)},
            "graph": {"path": str(args.graph), "n": len(graph_names)},
        },
    )

    only_vision = len(vision_names) - len(shared)
    only_graph = len(graph_names) - len(shared)
    print(f"[gallery_join] vision {len(vision_names)} · graph {len(graph_names)} "
          f"-> condivisi {len(shared)}")
    print(f"[gallery_join] esclusi: {only_vision} solo-vision, {only_graph} solo-graph")
    print(f"[gallery_join] sha1 {payload['sha1'][:12]} -> {args.out}")
    print("[gallery_join] ora punta `eval.gallery_names` a questo file nei due YAML: "
          "le due gallery diventano identiche riga per riga.")


if __name__ == "__main__":
    main()
