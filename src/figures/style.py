# src/figures/style.py

"""
Shared look of the report figures — TWO-COLUMN conference layout (18 Sep 2026).

Why a module and not three lines copied in every figure: in a two-column paper
a figure is drawn at its FINAL width (one column or the full text width) and is
never rescaled by LaTeX. Rescaling is what makes the labels of one figure
bigger than the body text and those of the next one smaller; the font sizes
below are chosen so that text inside a figure matches an 8 pt caption at 100%.

Colours are the Okabe-Ito palette (safe for colour-blind readers) and every
system also carries its own dash pattern and marker, so a figure still reads
when the paper is printed in black and white.

Every figure writes three files in `figures/`:

    <name>.pdf          vector, this is what goes in the report
    <name>.png          400 dpi raster, to look at it quickly
    <name>.sources.txt  the files it read, the command, the key numbers

The third one is the point: a figure whose numbers cannot be traced back to the
per-query files it came from cannot be remade, and a number that cannot be
remade cannot go in a paper.
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

# Two-column conference layout (CVPR/IEEE geometry, inches).
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

# One look per system, reused across figures: the reader learns the colours once.
SYSTEM_STYLE = {
    "fusion": {"color": OKABE_ITO["green"], "linestyle": "-", "marker": "o"},
    "graph": {"color": OKABE_ITO["blue"], "linestyle": "--", "marker": "s"},
    "vision": {"color": OKABE_ITO["vermillion"], "linestyle": "-.", "marker": "^"},
    "baseline": {"color": OKABE_ITO["grey"], "linestyle": ":", "marker": "x"},
}

# One look per fused pair (the three rungs of the gain ladder, status.md §49-§51).
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
        # Type 42 = embedded TrueType: the text in the PDF stays selectable and
        # searchable instead of being turned into curves.
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
    """Create a figure already sized for the report.

    Args:
        width: "column" (one column) or "text" (both columns).
        height_in: height in inches; ~2.2-2.6 keeps a single-column figure
            close to the golden ratio without eating half a page.

    Returns:
        (fig, ax) with the shared style applied.
    """
    if width not in ("column", "text"):
        raise ValueError(f"width sconosciuta: {width!r} (usa 'column' o 'text')")
    width_in = COLUMN_WIDTH_IN if width == "column" else TEXT_WIDTH_IN
    apply_style()
    return plt.subplots(figsize=(width_in, height_in))


def mean_ci(values: np.ndarray, b: int = BOOTSTRAP_B,
            seed: int = BOOTSTRAP_SEED) -> tuple[float, float, float]:
    """Mean of a per-query vector with its 95% bootstrap CI.

    Same resampling as `significance.bootstrap_ci` (same B and seed as every
    delta already in the report), applied to the values themselves instead of
    to a paired difference: here the band says how precise the mean is.

    Args:
        values: [P] one value per query.

    Returns:
        (mean, lo, hi); NaNs if `values` is empty.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    lo, hi = bootstrap_ci(values, b=b, seed=seed)
    return float(values.mean()), lo, hi


def _command_line() -> str:
    """The command that produced the figure, as one would retype it.

    With `python -m pkg.mod` argv[0] is the file path, which is not re-runnable
    as written: the module name comes from `__main__.__spec__`.
    """
    main = sys.modules.get("__main__")
    spec = getattr(main, "__spec__", None)
    head = f"python -m {spec.name}" if spec else f"python {sys.argv[0]}"
    return " ".join([head, *sys.argv[1:]])


def save_figure(fig, name: str, sources, notes=(), out_dir: Path = FIGURES_DIR) -> dict:
    """Write <name>.pdf, <name>.png and <name>.sources.txt.

    Args:
        fig: the figure to save.
        name: base name, no extension (e.g. "f_damage_test").
        sources: paths actually read to produce the numbers.
        notes: lines with the key numbers of the figure, so the caption can be
            written (and checked) without rerunning anything.
        out_dir: destination directory, created if missing.

    Returns:
        dict with the three paths written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{name}.pdf"
    png = out_dir / f"{name}.png"
    txt = out_dir / f"{name}.sources.txt"

    fig.savefig(pdf)
    fig.savefig(png)

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
