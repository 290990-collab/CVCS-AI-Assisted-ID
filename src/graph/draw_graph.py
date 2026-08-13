# src/graph/draw_graph.py

"""
Visualizzazione dei grafi prodotti da `graph_builder.build_graph`.

Serve a *ispezionare a occhio* che l'adapter RoomMeta -> Data faccia la cosa
giusta: nodi = stanze (colorate per tipo), archi = adiacenze. Due layout:

- "spatial" (default): ogni nodo e' piazzato al centroide reale della stanza
  (le coordinate cx, cy sono gia' dentro le node features). Cosi' il grafo
  ricalca la disposizione fisica della pianta e si puo' sovrapporre al PNG.
- "spring": layout a molle di networkx, utile quando la geometria non serve e
  si vuole solo leggere la topologia.

Uso da riga di comando:
    python -m src.graph.draw_graph 10002 --out /tmp/g.png
    python -m src.graph.draw_graph 10002 --overlay          # grafo sul PNG
    python -m src.graph.draw_graph 10002 10003 3951 --grid  # piu' piante
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend headless: siamo su nodi senza display.
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.lines import Line2D
from torch_geometric.data import Data

from src.graph.graph_builder import build_graph_from_png
from src.data.rplan_metadata import NUM_ROOM_TYPES, ROOM_TYPES

# Cartella dei PNG della gallery (per l'overlay opzionale del grafo sulla pianta).
DEFAULT_SNAPSHOT_DIR = Path(
    "/work/cvcs2026/ai_interior_design/datasets/RPLAN/Interface/static/Data/snapshot_train"
)

# Palette fissa: un colore per tipo di stanza, stabile tra figure diverse.
_PALETTE = plt.get_cmap("tab20")(np.linspace(0, 1, NUM_ROOM_TYPES))


def node_room_types(data: Data) -> list[int]:
    """Recupera il tipo di ogni nodo dalla parte one-hot delle feature.

    Le prime NUM_ROOM_TYPES colonne di `data.x` sono l'one-hot del tipo, quindi
    l'argmax di ciascuna riga da' l'id-tipo della stanza.

    Input:  data -> grafo PyG.
    Output: lista di id-tipo (int), uno per nodo, allineata a data.x.
    """
    onehot = data.x[:, :NUM_ROOM_TYPES]
    return onehot.argmax(dim=1).tolist()


def _to_networkx(data: Data) -> nx.Graph:
    """Converte il grafo PyG in un grafo networkx non diretto (per il disegno).

    Deduplica gli archi (in PyG sono simmetrizzati, quindi presenti in entrambe
    le direzioni) affidandosi al fatto che nx.Graph e' non diretto.

    Input:  data -> grafo PyG.
    Output: networkx.Graph con i soli nodi/archi (senza attributi).
    """
    g = nx.Graph()
    g.add_nodes_from(range(data.num_nodes))

    edge_index = data.edge_index.cpu().numpy()
    for src, dst in edge_index.T:
        g.add_edge(int(src), int(dst))
    return g


def _spatial_positions(data: Data) -> dict[int, tuple[float, float]]:
    """Posizioni dei nodi = centroidi reali delle stanze (coordinate immagine).

    cx, cy stanno nelle colonne geometriche (subito dopo l'one-hot) e sono gia'
    normalizzate in [0,1]. La y cresce verso il basso (convenzione immagine):
    l'asse verra' invertito in fase di disegno per allinearsi al PNG.

    Input:  data -> grafo PyG.
    Output: dict node_id -> (x, y).
    """
    cx = data.x[:, NUM_ROOM_TYPES].cpu().numpy()
    cy = data.x[:, NUM_ROOM_TYPES + 1].cpu().numpy()
    return {i: (float(cx[i]), float(cy[i])) for i in range(data.num_nodes)}


def _legend_handles(types_present: set[int]) -> list[Line2D]:
    """Costruisce le voci di legenda (pallino colorato + nome) per i tipi usati.

    Input:  types_present -> insieme di id-tipo effettivamente nel grafo.
    Output: lista di handle Line2D da passare a ax.legend.
    """
    handles = []
    for t in sorted(types_present):
        handles.append(
            Line2D(
                [0], [0], marker="o", linestyle="",
                markerfacecolor=_PALETTE[t], markeredgecolor="black",
                markersize=9, label=ROOM_TYPES[t],
            )
        )
    return handles


def draw_graph(
    data: Data,
    ax: plt.Axes | None = None,
    layout: str = "spatial",
    background_png: str | Path | None = None,
    title: str | None = None,
    show_legend: bool = True,
) -> plt.Axes:
    """Disegna un singolo grafo su un asse matplotlib.

    Input:
        data:           grafo PyG da build_graph.
        ax:             asse su cui disegnare; se None ne crea uno nuovo.
        layout:         "spatial" (centroidi reali) o "spring" (molle nx).
        background_png: path del PNG della pianta da mettere sotto il grafo
                        (solo con layout "spatial"); None per nessuno sfondo.
        title:          titolo del pannello; se None usa il nome della pianta.
        show_legend:    se True aggiunge la legenda tipi-colore.
    Output: l'asse su cui e' stato disegnato.
    Side effects: disegna su `ax`; non salva nulla.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    room_types = node_room_types(data)
    node_colors = [_PALETTE[t] for t in room_types]
    graph = _to_networkx(data)

    # --- scelta del layout ---
    if layout == "spatial":
        pos = _spatial_positions(data)
    elif layout == "spring":
        pos = nx.spring_layout(graph, seed=42)
    else:
        raise ValueError(f"layout sconosciuto: {layout!r} (usa 'spatial'|'spring')")

    # --- sfondo: la pianta reale sotto il grafo (solo spatial, coord in [0,1]) ---
    # extent [0,1,1,0] mappa la y verso il basso -> allinea PNG e centroidi.
    if background_png is not None and layout == "spatial":
        img = plt.imread(str(background_png))
        ax.imshow(img, extent=(0, 1, 1, 0), aspect="auto")

    # --- disegno di archi e nodi ---
    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color="#555555", width=1.5)
    nx.draw_networkx_nodes(
        graph, pos, ax=ax, node_color=node_colors,
        edgecolors="black", node_size=350,
    )
    # etichetta = indice della stanza (per incrociare con room_types se serve).
    nx.draw_networkx_labels(graph, pos, ax=ax, font_size=8, font_color="white")

    # Senza sfondo, l'asse va invertito a mano per avere la y verso il basso
    # (con lo sfondo ci pensa gia' l'extent dell'imshow).
    if layout == "spatial" and background_png is None:
        ax.invert_yaxis()

    ax.set_title(title if title is not None else str(data.name), fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

    if show_legend:
        ax.legend(
            handles=_legend_handles(set(room_types)),
            loc="center left", bbox_to_anchor=(1.0, 0.5),
            fontsize=7, frameon=False,
        )
    return ax


def save_graph(
    data: Data,
    out_path: str | Path,
    **kwargs,
) -> Path:
    """Disegna un grafo e salva la figura su file.

    Input:
        data:     grafo PyG.
        out_path: percorso del file immagine di uscita (PNG).
        **kwargs: inoltrati a `draw_graph` (layout, background_png, ...).
    Output: il Path del file salvato.
    Side effects: crea la cartella di destinazione e scrive l'immagine.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    draw_graph(data, ax=ax, **kwargs)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def draw_graph_grid(
    graphs: list[Data],
    out_path: str | Path,
    ncols: int = 4,
    layout: str = "spatial",
) -> Path:
    """Disegna piu' grafi in una griglia e salva un'unica figura.

    Utile per controllare a colpo d'occhio un campione di piante. La legenda
    viene omessa nei singoli pannelli per non affollarli.

    Input:
        graphs:   lista di grafi PyG.
        out_path: file immagine di uscita.
        ncols:    numero di colonne della griglia.
        layout:   layout passato a draw_graph.
    Output: il Path del file salvato.
    Side effects: scrive l'immagine su disco.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    nrows = (len(graphs) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, data in zip(axes, graphs):
        draw_graph(data, ax=ax, layout=layout, show_legend=False)
    # spegne gli assi in eccesso quando i grafi non riempiono la griglia.
    for ax in axes[len(graphs):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _png_path(name: str, snapshot_dir: Path) -> Path:
    """Risolve lo stem/nome di una pianta nel path del suo PNG."""
    stem = Path(name).stem
    return snapshot_dir / f"{stem}.png"


def main() -> None:
    """CLI: visualizza uno o piu' grafi RPLAN dati i loro nomi/stem."""
    parser = argparse.ArgumentParser(description="Visualizza grafi RPLAN.")
    parser.add_argument("names", nargs="+", help="stem/nome delle piante (es. 10002)")
    parser.add_argument("--out", default="results/graph_viz/graph.png",
                        help="file immagine di uscita")
    parser.add_argument("--layout", default="spatial", choices=["spatial", "spring"])
    parser.add_argument("--overlay", action="store_true",
                        help="sovrappone il grafo al PNG della pianta (solo layout spatial)")
    parser.add_argument("--grid", action="store_true",
                        help="dispone piu' piante in una griglia")
    parser.add_argument("--snapshot-dir", default=str(DEFAULT_SNAPSHOT_DIR))
    args = parser.parse_args()

    snapshot_dir = Path(args.snapshot_dir)

    # Costruisce i grafi, saltando i nomi senza record .mat.
    graphs = []
    for name in args.names:
        png = _png_path(name, snapshot_dir)
        data = build_graph_from_png(png)
        if data is None:
            print(f"[skip] {name}: nessun record .mat collegabile")
            continue
        graphs.append(data)

    if not graphs:
        print("nessun grafo da disegnare.")
        return

    if args.grid or len(graphs) > 1:
        out = draw_graph_grid(graphs, args.out, layout=args.layout)
        print(f"griglia di {len(graphs)} grafi salvata in {out}")
    else:
        data = graphs[0]
        bg = _png_path(data.name, snapshot_dir) if args.overlay else None
        out = save_graph(data, args.out, layout=args.layout, background_png=bg)
        print(f"grafo salvato in {out}")


if __name__ == "__main__":
    main()
