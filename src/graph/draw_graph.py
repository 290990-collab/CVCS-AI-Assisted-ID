"""
Draw the graphs produced by `graph_builder.build_graph`.

Nodes = rooms colored by type, edges = adjacencies. Layouts: "spatial" (default; nodes at the room centroids
stored in the node features, so the graph can be overlaid on the plan PNG) and "spring" (networkx).

Usage:
    python -m src.graph.draw_graph 10002 --out /tmp/g.png
    python -m src.graph.draw_graph 10002 --overlay          # graph on the PNG
    python -m src.graph.draw_graph 10002 10003 3951 --grid  # several plans
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.lines import Line2D
from torch_geometric.data import Data

from src.graph.graph_builder import build_graph_from_png
from src.data.rplan_metadata import NUM_ROOM_TYPES, ROOM_TYPES

# gallery PNGs, for the overlay
DEFAULT_SNAPSHOT_DIR = Path(
    "/work/cvcs2026/ai_interior_design/datasets/RPLAN/Interface/static/Data/snapshot_train"
)

# one fixed color per room type
_PALETTE = plt.get_cmap("tab20")(np.linspace(0, 1, NUM_ROOM_TYPES))


def node_room_types(data: Data) -> list[int]:
    """Room-type id per node: argmax of the one-hot in the first NUM_ROOM_TYPES columns of `data.x`."""
    onehot = data.x[:, :NUM_ROOM_TYPES]
    return onehot.argmax(dim=1).tolist()


def _to_networkx(data: Data) -> nx.Graph:
    """Undirected networkx graph (nodes and edges only); duplicate directions collapse."""
    g = nx.Graph()
    g.add_nodes_from(range(data.num_nodes))

    edge_index = data.edge_index.cpu().numpy()
    for src, dst in edge_index.T:
        g.add_edge(int(src), int(dst))
    return g


def _spatial_positions(data: Data) -> dict[int, tuple[float, float]]:
    """Node -> (cx, cy) room centroid in [0,1], image convention (y grows downward)."""
    cx = data.x[:, NUM_ROOM_TYPES].cpu().numpy()
    cy = data.x[:, NUM_ROOM_TYPES + 1].cpu().numpy()
    return {i: (float(cx[i]), float(cy[i])) for i in range(data.num_nodes)}


def _legend_handles(types_present: set[int]) -> list[Line2D]:
    """Legend handles (colored dot + name) for the room types present."""
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
    """Draw one graph on `ax` (new axes if None) and return it; nothing is saved.

    `background_png` (spatial layout only) is drawn under the graph.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    room_types = node_room_types(data)
    node_colors = [_PALETTE[t] for t in room_types]
    graph = _to_networkx(data)

    if layout == "spatial":
        pos = _spatial_positions(data)
    elif layout == "spring":
        pos = nx.spring_layout(graph, seed=42)
    else:
        raise ValueError(f"layout sconosciuto: {layout!r} (usa 'spatial'|'spring')")

    # extent [0,1,1,0]: y downward, aligns PNG and centroids
    if background_png is not None and layout == "spatial":
        img = plt.imread(str(background_png))
        ax.imshow(img, extent=(0, 1, 1, 0), aspect="auto")

    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color="#555555", width=1.5)
    nx.draw_networkx_nodes(
        graph, pos, ax=ax, node_color=node_colors,
        edgecolors="black", node_size=350,
    )
    # label = room index
    nx.draw_networkx_labels(graph, pos, ax=ax, font_size=8, font_color="white")

    # without background the y axis is inverted by hand (imshow extent does it otherwise)
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
    """Draw one graph and save it to `out_path`; `kwargs` go to `draw_graph`."""
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
    """Draw several graphs in a grid (no per-panel legend) and save one figure."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    nrows = (len(graphs) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, data in zip(axes, graphs):
        draw_graph(data, ax=ax, layout=layout, show_legend=False)
    # hide unused axes
    for ax in axes[len(graphs):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _png_path(name: str, snapshot_dir: Path) -> Path:
    """PNG path of a plan name/stem."""
    stem = Path(name).stem
    return snapshot_dir / f"{stem}.png"


def main() -> None:
    """CLI: draw one or more RPLAN graphs by name/stem."""
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

    # skip names without a .mat record
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
