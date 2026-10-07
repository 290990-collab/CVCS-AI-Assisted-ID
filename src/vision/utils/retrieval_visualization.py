# Render top-k retrieval panels (query + results with per-axis relevance).

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from src.data.rplan_metadata import ROOM_TYPES, load_metadata
from src.vision.data.vision_damage import damaged_query, make_patch_context, resolve_patch_size
from src.vision.evaluation.evaluate import (
    _transform_image_size,
    build_gallery_axes,
    partial_runs,
    whitening_fit_rows,
)
from src.evaluation.metrics import average_precision_at_k, ndcg_at_k, recall_at_k
from src.evaluation.relevance import AXES, DISCRETE_AXES
from src.vision.models.projection_head import load_head
from src.vision.models.retrieval_model import VisionRetrievalPipeline
from src.vision.models.vision_model_manager import VisionModelManager
from src.vision.utils.config import load_vision_config, transform_tag

DEFAULT_CONFIG = "configs/vision_retrieval.yaml"
OUT_DIR   = "results/visualizations"

THUMB_SIZE = 512                   # native snapshots ~1167x875
TOP_K      = 5

# UI palette
COLOR_BG       = (245, 245, 248)
COLOR_QUERY    = (59, 130, 246)    # blue
COLOR_SOURCE   = (245, 170, 30)    # orange: recovered source plan (partial)
COLOR_HEADER   = (30,  30,  40)
COLOR_LABEL    = (50,  50,  60)
COLOR_SCORE_BG = (220, 220, 225)
COLOR_LEGEND   = (150, 160, 175)

# --- UI geometry ---
# base values tuned for THUMB_SIZE=256; `_apply_scale()` rescales them proportionally
_BASE_THUMB = 256
SCALE       = 1.0

SCORE_BAR_H = 10
LABEL_H     = 66                   # 3 lines: visual score, relevance, id
MARGIN      = 14
BORDER      = 3
FONT_LABEL  = 11
FONT_TITLE  = 13
FONT_SUB    = 11
FONT_BADGE  = 12
BADGE_R     = 14
LINE_GAP    = 3

# fixed legend; ASCII markers only (LiberationSans lacks glyphs like a check mark)
LEGEND_TEXT = ("vis = score visivo FAISS   |   C/T/G = rilevanza .mat "
               "(composizione / topologia / geometria)   |   "
               "exact:C/T = classe di equivalenza esatta (Recall/mAP)")


def _apply_scale(thumb: int) -> None:
    """Rescale THUMB_SIZE and the UI constants (fonts, margins, bars) proportionally."""
    global THUMB_SIZE, SCALE, SCORE_BAR_H, LABEL_H, MARGIN, BORDER
    global FONT_LABEL, FONT_TITLE, FONT_SUB, FONT_BADGE, BADGE_R, LINE_GAP
    THUMB_SIZE = thumb
    SCALE = thumb / _BASE_THUMB
    SCORE_BAR_H = max(2, round(10 * SCALE))
    LABEL_H     = round(66 * SCALE)
    MARGIN      = max(4, round(14 * SCALE))
    BORDER      = max(1, round(3 * SCALE))
    FONT_LABEL  = max(9, round(11 * SCALE))
    FONT_TITLE  = max(11, round(13 * SCALE))
    FONT_SUB    = max(9, round(11 * SCALE))
    FONT_BADGE  = max(10, round(12 * SCALE))
    BADGE_R     = max(10, round(14 * SCALE))
    LINE_GAP    = max(2, round(3 * SCALE))


# --- drawing primitives ---

def _load_font(size: int) -> ImageFont.ImageFont:
    for name in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _score_color(score: float) -> tuple:
    """Map a score in [0, 1] to a colour: green (high) to orange/red (low); used for the cell border."""
    t = max(0.0, min(1.0, score))
    r = int(20  + (235 - 20)  * (1 - t))
    g = int(180 * t)
    b = int(30)
    return (r, g, b)


def _load_thumb(img_path: str, size: int) -> Image.Image:
    """Load and resize to `size x size`; unreadable file -> grey placeholder."""
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception:
        img = Image.new("RGB", (size, size), color=(200, 200, 200))
    return img.resize((size, size), Image.LANCZOS)


def _paste_framed(
    canvas: Image.Image,
    img: Image.Image,
    x: int,
    y: int,
    border: int,
    border_color: tuple,
) -> None:
    """Paste `img` on `canvas` at (x, y) with a `border` px frame."""
    w, h   = img.size
    framed = Image.new("RGB", (w + border * 2, h + border * 2), color=border_color)
    framed.paste(img, (border, border))
    canvas.paste(framed, (x, y))


def _draw_score_bar(
    draw: ImageDraw.Draw,
    x: int,
    y: int,
    width: int,
    height: int,
    fg_color: tuple,
    score: float | None,
) -> None:
    """Score bar under the thumbnail: full `fg_color` if `score is None` (query), else filled to `score` in [0, 1]."""
    if score is None:
        draw.rectangle([x, y, x + width, y + height], fill=fg_color)
        return

    fill_w = int(width * max(0.0, min(1.0, score)))
    draw.rectangle([x, y, x + width,  y + height], fill=COLOR_SCORE_BG)
    if fill_w > 0:
        draw.rectangle([x, y, x + fill_w, y + height], fill=fg_color)


def _draw_labels(
    draw: ImageDraw.Draw,
    lines: list[str],
    x: int,
    y: int,
    font: ImageFont.ImageFont,
) -> None:
    """Left-aligned text lines, stacked."""
    cur_y = y
    for line in lines:
        draw.text((x, cur_y), line, fill=COLOR_LABEL, font=font)
        bbox  = draw.textbbox((0, 0), line, font=font)
        cur_y += (bbox[3] - bbox[1]) + LINE_GAP


def _draw_rank_badge(
    draw: ImageDraw.Draw,
    x: int,
    y: int,
    rank: int,
    font: ImageFont.ImageFont,
) -> None:
    """White circle with the rank number, top-left of the cell."""
    r       = BADGE_R
    cx, cy  = x + MARGIN + BORDER + r + 2, y + MARGIN + BORDER + r + 2
    draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                 fill=(255, 255, 255), outline=(80, 80, 80), width=1)

    text     = str(rank)
    bbox     = draw.textbbox((0, 0), text, font=font)
    tw, th   = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text((cx - tw // 2, cy - th // 2), text, fill=(30, 30, 30), font=font)


# --- main rendering ---

def visualize_query(
    query_path: str,
    results: list[dict],
    out_path: str,
    query_index: int = 0,
    summary_line: str = "",
    query_image: Image.Image | None = None,
    query_label: list[str] | None = None,
) -> None:
    """
    PNG panel with the query on the left and the top-k results on the right.

    Per result: visual score (FAISS) as the bar under the thumbnail; per-axis relevance from the .mat
    metadata (composition/topology/geometry, C/T/G in [0,1]) in the label, border colour = their mean.

    results: list of {"path", "score", "rel_axes"}, rel_axes = {"composition","topology","geometry"} or None
    (no .mat metadata). summary_line: per-query metrics line (header). query_image: thumbnail used instead of
    `query_path` (partial mode: the degraded plan). query_label: query cell label (default ["QUERY", stem]).
    """
    font_label = _load_font(FONT_LABEL)
    font_title = _load_font(FONT_TITLE)
    font_sub   = _load_font(FONT_SUB)
    font_badge = _load_font(FONT_BADGE)

    # one cell for the query + one per result
    n_cols     = len(results) + 1
    bordered   = THUMB_SIZE + BORDER * 2
    cell_w     = THUMB_SIZE + MARGIN * 2

    # header: title, optional metrics summary, fixed legend
    pad     = round(8 * SCALE)
    title_h = FONT_TITLE + round(8 * SCALE)
    sub_h   = FONT_SUB + round(8 * SCALE)
    header_h = pad + title_h + (sub_h if summary_line else 0) + sub_h + pad
    cell_h   = MARGIN + bordered + SCORE_BAR_H + LABEL_H + MARGIN
    total_w  = cell_w * n_cols
    total_h   = header_h + cell_h

    canvas = Image.new("RGB", (total_w, total_h), color=COLOR_BG)
    draw   = ImageDraw.Draw(canvas)

    draw.rectangle([0, 0, total_w, header_h], fill=COLOR_HEADER)
    header_text = (
        f"Query #{query_index:02d} — {Path(query_path).name}    |    "
        f"top-{len(results)} risultati"
    )
    y_line = pad
    draw.text((MARGIN, y_line), header_text, fill=(220, 220, 235), font=font_title)
    y_line += title_h
    if summary_line:
        draw.text((MARGIN, y_line), summary_line, fill=(150, 200, 160), font=font_sub)
        y_line += sub_h
    draw.text((MARGIN, y_line), LEGEND_TEXT, fill=COLOR_LEGEND, font=font_sub)

    def paste_cell(
        img_path: str,
        col: int,
        label_lines: list[str],
        border_color: tuple,
        score: float | None = None,
        rank: int | None = None,
        img_obj: Image.Image | None = None,
    ) -> None:
        """One cell in column `col` (0 = query, 1..k = results); `score=None` = query cell; `img_obj` replaces `img_path`."""
        x0    = cell_w * col + MARGIN
        y0    = header_h + MARGIN
        bar_y = y0 + bordered

        thumb = (
            img_obj.convert("RGB").resize((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)
            if img_obj is not None else _load_thumb(img_path, THUMB_SIZE)
        )
        _paste_framed(canvas, thumb, x0, y0, BORDER, border_color)
        _draw_score_bar(draw, x0, bar_y, bordered, SCORE_BAR_H,
                        fg_color=border_color, score=score)
        _draw_labels(draw, label_lines, x0, bar_y + SCORE_BAR_H + 5, font_label)

        if rank is not None:
            _draw_rank_badge(draw, x0 - MARGIN, y0 - MARGIN, rank, font_badge)

    # query cell (col 0); in partial `query_image` is the degraded plan
    paste_cell(
        img_path=query_path,
        col=0,
        label_lines=query_label or ["QUERY", Path(query_path).stem],
        border_color=COLOR_QUERY,
        img_obj=query_image,
    )

    # result cells (col 1..k); partial: the recovered original (is_source) is tagged [orig]
    for i, r in enumerate(results):
        rel_axes = r.get("rel_axes")
        if rel_axes is None:
            rel_line     = "rel  n/a"
            border_color = _score_color(r["score"])   # fallback: visual score
        else:
            c, t, g      = rel_axes["composition"], rel_axes["topology"], rel_axes["geometry"]
            exact        = r.get("exact") or {}
            hit          = [lbl for lbl, ax in (("C", "composition"), ("T", "topology"))
                            if exact.get(ax)]
            marks        = f"   exact:{'/'.join(hit)}" if hit else ""
            rel_line     = f"C {c:.2f} T {t:.2f} G {g:.2f}{marks}"
            border_color = _score_color((c + t + g) / 3.0)
        is_source = r.get("is_source", False)
        id_line   = ("[orig] " if is_source else "") + Path(r["path"]).stem
        paste_cell(
            img_path=r["path"],
            col=i + 1,
            label_lines=[f"vis   {r['score']:.4f}", rel_line, id_line],
            border_color=COLOR_SOURCE if is_source else border_color,
            score=r["score"],
            rank=i + 1,
        )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


# --- per-axis relevance on results (full and partial) ---

def _decorate_with_axes(results, qi, axes, stem2row) -> dict:
    """Annotate results with graded C/T/G relevance and exact-class membership vs query `qi`;
    returns per-axis similarities over the whole gallery."""
    sims = {ax: axes.sim(ax, qi) for ax in AXES}
    exact_masks = {ax: axes.relevant(ax, qi) for ax in DISCRETE_AXES}
    for r in results:
        row = stem2row.get(Path(r["path"]).stem)
        if row is not None:
            r["rel_axes"] = {ax: float(sims[ax][row]) for ax in AXES}
            r["exact"]    = {ax: bool(exact_masks[ax][row]) for ax in DISCRETE_AXES}
        else:
            r["rel_axes"] = None
            r["exact"]    = None
    return sims


def _metrics_summary(sims, results, qi, axes, stem2row, exclude_self) -> str:
    """Per-query summary: nDCG@k (3 axes, IDCG over the whole gallery), Recall@k and mAP@k (discrete axes,
    relevant = exact class); `n/a` when the query is a singleton on that axis."""
    k = len(results)
    ret_rows = [stem2row[Path(r["path"]).stem]
                for r in results if Path(r["path"]).stem in stem2row]
    if exclude_self:
        # self removed from the results too, else it would count as an error
        ret_rows = [r for r in ret_rows if r != qi]
    not_self = np.ones(len(axes), dtype=bool)
    if exclude_self:
        not_self[qi] = False

    ndcg = "  ".join(
        f"{ax[0].upper()} {ndcg_at_k(sims[ax][ret_rows], sims[ax][not_self], k):.3f}"
        for ax in AXES
    )

    def _fmt(v) -> str:
        return "n/a" if v is None else f"{v:.3f}"

    rec, ap = [], []
    for ax in DISCRETE_AXES:
        mask = axes.relevant(ax, qi).copy()
        if exclude_self:
            mask[qi] = False
        n_rel = int(mask.sum())
        rr = mask[ret_rows]
        rec.append(f"{ax[0].upper()} {_fmt(recall_at_k(rr, n_rel, k))}")
        ap.append(f"{ax[0].upper()} {_fmt(average_precision_at_k(rr, n_rel, k))}")

    return (f"nDCG@{k} {ndcg}   |   R@{k} {'  '.join(rec)}   |   "
            f"mAP@{k} {'  '.join(ap)}")


# --- CLI ---

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualizza i risultati del retrieval visivo RPLAN")
    p.add_argument("--config",    default=DEFAULT_CONFIG, help="Config YAML (encoder + data_dir + save_dir)")
    p.add_argument("--data-dir",  default=None, help="Override della cartella PNG (default: config.retrieval.data_dir)")
    p.add_argument("--save-dir",  default=None, help="Override degli embedding salvati (default: config.retrieval.save_dir)")
    p.add_argument("--out-dir",   default=OUT_DIR,   help="Cartella di output per le visualizzazioni")
    p.add_argument("--top-k",     type=int, default=TOP_K, help="Numero di risultati per query")
    p.add_argument("--n-queries", type=int, default=5, help="Quante query eseguire")
    p.add_argument("--thumb",     type=int, default=THUMB_SIZE, help="Dimensione thumbnail (px)")
    p.add_argument("--partial",   action="store_true",
                   help="forza la modalità partial (override di partial.enabled)")
    # unknown args are dotlist overrides (e.g. model.variant=gem)
    return p.parse_known_args()


def _run_full_viz(pipeline, axes, stem2row, queries, args) -> None:
    """Full-image queries, self-match excluded."""
    for i, query_path in enumerate(tqdm(queries, desc="Query")):
        qp = str(query_path)

        # top_k + 1 to drop the query itself (self-match score ~1.0)
        results = pipeline.query(pipeline.load_query(qp), top_k=args.top_k + 1)
        results = [r for r in results
                   if not (r["path"] == qp or r["score"] >= 0.9999)][:args.top_k]

        qi = stem2row.get(Path(qp).stem)
        if qi is not None and bool(axes.valid[qi]) and results:
            sims = _decorate_with_axes(results, qi, axes, stem2row)
            summary_line = _metrics_summary(sims, results, qi, axes, stem2row, exclude_self=True)
        else:
            for r in results:
                r["rel_axes"] = None
            summary_line = "metadati .mat assenti per la query — solo score visivo"

        out_path = f"{args.out_dir}/query_{i:02d}_{query_path.stem}.png"
        visualize_query(qp, results, out_path, query_index=i, summary_line=summary_line)
        tqdm.write(f"[Viz] {out_path}")


def _run_partial_viz(pipeline, axes, stem2row, queries, pcfg, args, patch_ctx=None) -> None:
    """Degraded-plan queries: shows the masked input, marks the recovered original (self-recovery) and per-axis relevance.

    Damage goes through `vision_damage.damaged_query` (as in evaluation) and the panel shows the encoded image
    (resized canvas for patch). `patch_ctx` is needed only by `patch` runs (built in `main`)."""
    seed = int(pcfg.get("seed", 42))
    open_boundary = bool(pcfg.get("open_boundary", True))
    runs = partial_runs(pcfg)
    print(f"[Viz] PARTIAL | {len(runs)} run | open_boundary={open_boundary}")

    for label, strat, params in runs:
        safe = label.replace(" ", "_").replace("=", "")
        for i, query_path in enumerate(tqdm(queries, desc=f"[{label}]")):
            qp = str(query_path)
            qi = stem2row.get(Path(qp).stem)
            if qi is None or not bool(axes.valid[qi]):
                continue
            meta = load_metadata(qp)
            if meta is None:
                continue

            rng = random.Random(seed + qi)
            tensor, dinfo, img = damaged_query(qp, meta, strat, params, rng, open_boundary,
                                               pipeline.transform, patch_ctx, return_image=True)

            # self not excluded: recovering the original is the goal
            results = pipeline.query(tensor, top_k=args.top_k)[:args.top_k]

            rank = None
            for j, r in enumerate(results, start=1):
                if stem2row.get(Path(r["path"]).stem) == qi:
                    r["is_source"] = True
                    rank = rank or j

            sims = _decorate_with_axes(results, qi, axes, stem2row)
            # metrics exclude the self (as the report tables); the source marker comes from the grid
            metrics_line = _metrics_summary(sims, results, qi, axes, stem2row, exclude_self=True)
            sr = f"self-rec rank={rank}" if rank else "self-rec: miss"
            area = f"area tolta {dinfo['area_removed']:.2f}"
            summary_line = f"{label}  |  {area}  |  {sr}  |  {metrics_line}"

            removed = dinfo.get("removed")
            if removed is not None:        # rooms / nowalls: removed rooms
                removed_types = ", ".join(ROOM_TYPES[meta.room_types[k]] for k in removed) or "nessuna"
                what = f"-{len(removed)}: {removed_types}"
            else:                          # crop / patch: no whole rooms removed
                what = f"{strat}: {area}"
            query_label = ["QUERY (partial)", what, Path(qp).stem]

            out_path = f"{args.out_dir}/{safe}/query_{i:02d}_{query_path.stem}.png"
            visualize_query(
                qp, results, out_path, query_index=i, summary_line=summary_line,
                query_image=img, query_label=query_label,
            )
            tqdm.write(f"[Viz] {out_path}")


def main():
    args, overrides = parse_args()

    # UI constants are module globals read by paste_cell
    _apply_scale(args.thumb)

    config   = load_vision_config(args.config, overrides)
    manager  = VisionModelManager(config)
    pipeline = VisionRetrievalPipeline(
        encoder=manager.encoder,
        transform=manager.transform,
        device=manager.device,
    )

    data_dir = args.data_dir or config.retrieval.data_dir
    save_dir = args.save_dir or config.retrieval.save_dir
    pipeline.load(save_dir)

    # optional head + whitening, as in evaluate
    head = None
    hcfg = getattr(config, "head", None)
    if hcfg is not None and hcfg.get("enabled"):
        head = load_head(save_dir, pipeline.raw_embeddings.shape[1],
                         hcfg.hidden_dim, hcfg.out_dim, manager.device,
                         filename=hcfg.get("file") or "head.pt")
    pipeline.prepare_index(
        head=head,
        whiten=config.whitening.enabled,
        eps=config.whitening.eps,
        whiten_dim=config.whitening.get("dim"),
        fit_rows=whitening_fit_rows(config, pipeline.image_paths),
    )

    # per-axis relevance over the gallery; rows aligned to pipeline.image_paths, lookup by stem
    axes, stem2row = build_gallery_axes(pipeline.image_paths)

    all_images = sorted(Path(data_dir).glob("*.png"))
    queries    = all_images[:args.n_queries]
    if not queries:
        print(f"[Viz] Nessuna immagine trovata in {data_dir}")
        return

    pcfg = getattr(config, "partial", None)
    partial_on = args.partial or (pcfg is not None and bool(pcfg.get("enabled", False)))
    if partial_on and pcfg is None:
        raise SystemExit("--partial richiesto ma manca il blocco `partial` nel config")

    # output: <out_dir>/<model>_<variant>_<transform tag>_<full|partial>/
    mode = "partial" if partial_on else "full"
    args.out_dir = (f"{args.out_dir}/{config.model.name}_{config.model.variant}_"
                    f"{transform_tag(config)}_{mode}")

    if partial_on:
        patch_ctx = None
        if any(strat == "patch" for _, strat, _ in partial_runs(pcfg)):
            # same resolution as evaluate.main: a wrong preset fails here
            patch_size = resolve_patch_size(
                pipeline.encoder, config.model.kwargs.get("patch_size"),
                _transform_image_size(pipeline.transform),
            )
            patch_ctx = make_patch_context(pipeline.transform, patch_size)
        _run_partial_viz(pipeline, axes, stem2row, queries, pcfg, args, patch_ctx)
    else:
        _run_full_viz(pipeline, axes, stem2row, queries, args)


if __name__ == "__main__":
    main()
