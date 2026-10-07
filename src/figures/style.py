"""
Shared look of the report figures (two-column conference layout).

Figures are drawn at their final width (one column or text width), never rescaled by LaTeX; font sizes match an 8 pt caption.
Okabe-Ito palette, plus a distinct dash pattern and marker per system so figures read in black and white.

Every figure writes in `figures/`: `<name>.pdf` (vector), `<name>.png` (400 dpi), `<name>.sources.txt` (files read, command, key numbers).
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")          # cluster: no display, write files and exit

import matplotlib.pyplot as plt   # noqa: E402  (must follow matplotlib.use)
import numpy as np                # noqa: E402

from src.evaluation.significance import (  # noqa: E402
    BOOTSTRAP_B,
    BOOTSTRAP_SEED,
    bootstrap_ci,
)

FIGURES_DIR = Path("figures")

# CVPR/IEEE column geometry, inches
COLUMN_WIDTH_IN = 3.25
TEXT_WIDTH_IN = 6.875

BASE_FONT_PT = 8
SMALL_FONT_PT = 7

# Okabe-Ito qualitative palette.
OKABE_ITO = {
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#000000",
    "grey": "#8C8C8C",
}

# one look per system, shared across figures
SYSTEM_STYLE = {
    "fusion": {"color": OKABE_ITO["green"], "linestyle": "-", "marker": "o"},
    "graph": {"color": OKABE_ITO["blue"], "linestyle": "--", "marker": "s"},
    "vision": {"color": OKABE_ITO["vermillion"], "linestyle": "-.", "marker": "^"},
    "baseline": {"color": OKABE_ITO["grey"], "linestyle": ":", "marker": "x"},
}

# one look per fused pair
PAIR_STYLE = {
    "vision-graph": {"color": OKABE_ITO["green"], "linestyle": "-", "marker": "o"},
    "vision-vision": {"color": OKABE_ITO["orange"], "linestyle": "-.", "marker": "^"},
    "graph-graph": {"color": OKABE_ITO["blue"], "linestyle": "--", "marker": "s"},
}


def apply_style() -> None:
    """Set the rcParams shared by every report figure. Call it once, first."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        # Type 42: embedded TrueType, selectable text
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.size": BASE_FONT_PT,
        "axes.titlesize": BASE_FONT_PT,
        "axes.labelsize": BASE_FONT_PT,
        "xtick.labelsize": SMALL_FONT_PT,
        "ytick.labelsize": SMALL_FONT_PT,
        "legend.fontsize": SMALL_FONT_PT,
        "legend.frameon": False,
        "legend.handlelength": 2.2,
        "legend.borderaxespad": 0.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.30,
        "lines.linewidth": 1.4,
        "lines.markersize": 4.0,
        "errorbar.capsize": 2.0,
    })


def new_figure(width: str = "column", height_in: float = 2.4):
    """(fig, ax) sized for the report; `width` is "column" or "text", `height_in` in inches."""
    if width not in ("column", "text"):
        raise ValueError(f"width sconosciuta: {width!r} (usa 'column' o 'text')")
    width_in = COLUMN_WIDTH_IN if width == "column" else TEXT_WIDTH_IN
    apply_style()
    return plt.subplots(figsize=(width_in, height_in))


def mean_ci(values: np.ndarray, b: int = BOOTSTRAP_B,
            seed: int = BOOTSTRAP_SEED) -> tuple[float, float, float]:
    """(mean, lo, hi) of a per-query [P] vector, 95% bootstrap CI as `significance.bootstrap_ci`; NaNs if empty."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    lo, hi = bootstrap_ci(values, b=b, seed=seed)
    return float(values.mean()), lo, hi


def _command_line() -> str:
    """Command that produced the figure; `python -m` uses the module name from `__main__.__spec__`."""
    main = sys.modules.get("__main__")
    spec = getattr(main, "__spec__", None)
    head = f"python -m {spec.name}" if spec else f"python {sys.argv[0]}"
    return " ".join([head, *sys.argv[1:]])


def save_figure(fig, name: str, sources, notes=(), out_dir: Path = FIGURES_DIR, svg: bool = False) -> dict:
    """Write <name>.pdf, .png, .sources.txt (and .svg if `svg`); `notes` are the key numbers; returns the paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{name}.pdf"
    png = out_dir / f"{name}.png"
    txt = out_dir / f"{name}.sources.txt"

    fig.savefig(pdf)
    fig.savefig(png)
    if svg:
        fig.savefig(out_dir / f"{name}.svg")

    lines = [
        f"# {name} — figura del report",
        f"data:     {dt.datetime.now():%Y-%m-%d %H:%M}",
        f"comando:  {_command_line()}",
        "",
        "numeri della figura:",
        *[f"  {line}" for line in notes],
        "",
        "file letti:",
        *[f"  {Path(s)}" for s in sources],
        "",
    ]
    txt.write_text("\n".join(lines), encoding="utf-8")
    plt.close(fig)
    print(f"scritto: {pdf}\n         {png}\n         {txt}")
    return {"pdf": pdf, "png": png, "sources": txt}
