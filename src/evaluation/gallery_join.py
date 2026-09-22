# src/evaluation/gallery_join.py

"""
Gallery condivisa fra i due rami: l'inner join dei nomi (fase **B.3**).

Il problema che risolve
-----------------------
Le due gallery non coincidono: il ramo vision indicizza 67.453 PNG, il ramo
graph 67.405 grafi. Finche' le due liste sono diverse, un confronto
vision<->graph non e' appaiato — sistemi diversi valutati su corpus diversi — e
`significance.py` lo accetta solo con `--allow-gallery-mismatch`, cioe'
dichiarando che il confronto e' zoppo.

La soluzione, e perche' e' fatta cosi'
--------------------------------------
1. **Un artefatto congelato, non un ricalcolo.** L'intersezione si calcola UNA
   volta con la CLI di questo modulo e finisce in un JSON con la sua provenienza
   (quante righe aveva ogni ramo, quante ne restano, lo sha1). Le run leggono
   quel file. Se l'intersezione si ricalcolasse a ogni run, due run lanciate a
   giorni di distanza potrebbero usare gallery diverse senza che nulla lo dica.
2. **Ordine canonico = alfabetico.** `gallery_sha1` dipende dall'ORDINE delle
   righe (`perquery.py`), e i due rami enumerano in ordini diversi. Riordinando
   entrambi sull'ordine del file condiviso, i due rami producono lo **stesso**
   sha1: e' quello che rende il confronto appaiato per costruzione, invece che
   per promessa.
3. **Nome canonico = lo stem.** Vision salva path di PNG, graph nomi nudi: lo
   stem e' l'unica forma comune (stessa convenzione di `random_floor.py`).

⚠️ Restringere la gallery **cambia i numeri** di entrambi i rami: 48 piante in
meno nel corpus di ricerca. Il vincolo DURO 2 (gallery = intero
`snapshot_train/`) resta rispettato nello spirito — non si sta selezionando un
sottoinsieme comodo, si sta togliendo cio' che un ramo non puo' vedere — ma va
dichiarato in ogni tabella prodotta cosi'.

Uso (CPU, secondi):

    python -m src.evaluation.gallery_join \\
        --vision embeddings/vision/<model>/<pool>/image_paths.json \\
        --graph  embeddings/graph/<run>/names.json \\
        --out    results/shared_gallery.json

Poi si punta `eval.gallery_names` a quel file nei due YAML.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from src.evaluation.perquery import gallery_sha1

SCHEMA_VERSION = 1


# ----------------------------------------------------------------------
# Nomi.
# ----------------------------------------------------------------------

def canonical_names(entries: list[str]) -> list[str]:
    """Righe -> nomi canonici (lo stem del file).

    Accetta indifferentemente path di PNG (`.../123.png` -> `123`) e nomi gia'
    nudi (`123` -> `123`), cioe' i due formati che i rami salvano.
    """
    return [Path(e).stem for e in entries]


def read_gallery_names(path: str | Path) -> list[str]:
    """Legge `image_paths.json` (vision) o `names.json` (graph) -> nomi canonici.

    Raises:
        ValueError: se il JSON non e' una lista di stringhe, o se contiene
            duplicati (un nome due volte renderebbe ambiguo il join).
    """
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


# ----------------------------------------------------------------------
# Intersezione.
# ----------------------------------------------------------------------

def compute_shared_names(vision_names: list[str], graph_names: list[str]) -> list[str]:
    """Nomi presenti in ENTRAMBI i rami, in ordine alfabetico (canonico).

    L'ordinamento non e' estetica: e' cio' che rende identico il `gallery_sha1`
    dei due rami, quindi confrontabile riga per riga il loro per-query.
    """
    shared = sorted(set(vision_names) & set(graph_names))
    if not shared:
        raise ValueError(
            "intersezione vuota: le due gallery non hanno nessun nome in comune. "
            "Controlla di aver passato i file dei due rami e non due volte lo stesso."
        )
    return shared


def restrict_rows(names: list[str], shared: list[str]) -> list[int]:
    """Righe di `names` da tenere, NELL'ORDINE di `shared`.

    Restituisce indici, non nomi: il chiamante li usa per riordinare embedding e
    lista dei nomi con la stessa permutazione, senza mai disallinearli.

    Raises:
        KeyError: se un nome condiviso non esiste in `names`. Vuol dire che il
            file dell'intersezione non appartiene a questa gallery — meglio
            fermarsi che valutare su un corpus diverso da quello dichiarato.
    """
    row_of = {n: i for i, n in enumerate(names)}
    missing = [n for n in shared if n not in row_of]
    if missing:
        raise KeyError(
            f"{len(missing)} nomi della gallery condivisa non sono in questa "
            f"gallery (es. {missing[:3]}): file dell'inner join non compatibile"
        )
    return [row_of[n] for n in shared]


# ----------------------------------------------------------------------
# Artefatto su disco.
# ----------------------------------------------------------------------

def write_shared_gallery(out_path: str | Path, shared: list[str],
                         sources: dict[str, dict]) -> dict:
    """Scrive il JSON dell'inner join con la sua provenienza. Ritorna il payload."""
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
    """Rilegge i nomi dell'inner join, verificando lo sha1 registrato.

    Raises:
        ValueError: se lo sha1 non torna (file modificato a mano o troncato).
    """
    payload = json.loads(Path(path).read_text())
    names = payload["names"]
    expected = payload.get("sha1")
    if expected and gallery_sha1(names) != expected:
        raise ValueError(
            f"{path}: sha1 non corrispondente ({gallery_sha1(names)} != {expected}). "
            "Il file e' stato modificato: rigeneralo con python -m src.evaluation.gallery_join"
        )
    return names


# ----------------------------------------------------------------------
# CLI.
# ----------------------------------------------------------------------

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
