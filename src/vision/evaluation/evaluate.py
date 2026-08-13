# src/vision/evaluation/evaluate.py

import argparse
import random
import re
from pathlib import Path

import numpy as np

from src.data.rplan_metadata import get_split, load_metadata
from src.vision.data.vision_partial_query import make_partial_query
from src.evaluation.metrics import (
    average_precision_at_k,
    ndcg_at_k,
    recall_at_k,
)
from src.evaluation.perquery import PerQueryRecorder, gallery_sha1
from src.evaluation.relevance import AXES, DISCRETE_AXES, GalleryAxes
from src.vision.models.projection_head import load_head
from src.vision.models.retrieval_model import VisionRetrievalPipeline
from src.vision.models.vision_model_manager import VisionModelManager
from src.vision.utils.config import load_vision_config, transform_tag

DEFAULT_CONFIG = "configs/vision_retrieval.yaml"
DEFAULT_K_VALUES = (1, 5, 10, 100)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="config YAML del retrieval vision (sceglie encoder + parametri)",
    )
    parser.add_argument(
        "--partial",
        action="store_true",
        help="forza la modalità partial (override di partial.enabled; le "
             "strategie restano quelle del config)",
    )
    # gli argomenti non riconosciuti sono override dotlist (es. model.variant=gem)
    return parser.parse_known_args()


def load_pipeline(config) -> VisionRetrievalPipeline:
    """Costruisce l'encoder dal config, ricarica gli embedding RAW e prepara
    l'indice applicando i contributi attivi (head opzionale + whitening), così
    si valutano raw/whiten/head/head+whiten riusando la stessa cache raw."""
    manager = VisionModelManager(config)
    pipeline = VisionRetrievalPipeline(
        encoder=manager.encoder,
        transform=manager.transform,
        device=manager.device,
    )
    pipeline.load(config.retrieval.save_dir)

    head = None
    hcfg = getattr(config, "head", None)
    if hcfg is not None and hcfg.get("enabled"):
        in_dim = pipeline.raw_embeddings.shape[1]
        head = load_head(config.retrieval.save_dir, in_dim,
                         hcfg.hidden_dim, hcfg.out_dim, manager.device)

    pipeline.prepare_index(
        head=head,
        whiten=config.whitening.enabled,
        eps=config.whitening.eps,
        whiten_dim=config.whitening.get("dim"),
    )
    return pipeline


def build_gallery_axes(image_paths: list[str]) -> tuple[GalleryAxes, dict[str, int]]:
    """Allinea i metadati .mat all'ordine della gallery FAISS e costruisce le
    feature per-asse.

    Returns:
        (axes, stem2row) dove stem2row mappa lo stem del PNG -> indice di riga.
    """
    print(f"[evaluate] Costruzione feature per-asse su {len(image_paths)} piante...")
    metas = [load_metadata(p) for p in image_paths]
    axes = GalleryAxes(metas)
    stem2row = {Path(p).stem: i for i, p in enumerate(image_paths)}
    n_valid = int(axes.valid.sum())
    print(f"[evaluate] Metadati .mat presenti: {n_valid}/{len(image_paths)}")
    return axes, stem2row


def sample_query_rows(
    image_paths: list[str], num_queries: int, seed: int, split: str | None = None
) -> list[int]:
    """Campiona righe-query in modo riproducibile. La gallery resta intera; con
    `split` (valid|test) le query sono ristrette alle piante di quello split
    ufficiale RPLAN (le righe sono comunque cercate sull'intera gallery)."""
    pool = list(range(len(image_paths)))
    if split:
        pool = [i for i in pool if get_split(image_paths[i]) == split]
        print(f"[evaluate] query ristrette allo split '{split}': {len(pool)} candidate")
    n = min(num_queries, len(pool))
    rng = random.Random(seed)
    rows = rng.sample(pool, n)
    print(f"[evaluate] {n} query campionate (gallery = query pool)\n")
    return rows


def _rows_from_results(results, qi, stem2row, max_k, exclude_self) -> list[int]:
    """Mappa i risultati FAISS su righe della gallery, opzionalmente escludendo
    il self. Ritorna fino a max_k righe in ordine di rank."""
    rows = []
    for r in results:
        row = stem2row.get(Path(r["path"]).stem)
        if row is None or (exclude_self and row == qi):
            continue
        rows.append(row)
    return rows[:max_k]


def retrieved_rows_for_query(
    pipeline, query_path, qi, stem2row, max_k
) -> list[int]:
    """Retrieval con query = immagine completa; esclude il self-match."""
    results = pipeline.query(pipeline.load_query(query_path), top_k=max_k + 1)
    return _rows_from_results(results, qi, stem2row, max_k, exclude_self=True)


# ----------------------------------------------------------------------
# Accumulo metriche per-asse (condiviso tra full e partial).
# ----------------------------------------------------------------------

def _new_metrics(k_values):
    return {
        ax: {"ndcg": {k: [] for k in k_values},
             "recall": {k: [] for k in k_values},
             "map": {k: [] for k in k_values}}
        for ax in AXES
    }


def _accumulate_axes(metrics, skipped, axes, qi, ret_rows, k_values, exclude_self):
    """Aggiorna `metrics`/`skipped` con le metriche per-asse di una query.

    exclude_self=True (full): self escluso da gallery, rilevanti e risultati.
    exclude_self=False (partial): la query è degradata, ritrovare l'originale è
    un successo legittimo -> il self resta nei rilevanti e nell'IDCG.
    """
    ret_rows = np.asarray(ret_rows, dtype=int)
    not_self = np.ones(len(axes), dtype=bool)
    if exclude_self:
        not_self[qi] = False

    for ax in AXES:
        sims = axes.sim(ax, qi)
        gallery_gains = sims[not_self]
        retrieved_gains = sims[ret_rows] if len(ret_rows) else np.empty(0)
        for k in k_values:
            metrics[ax]["ndcg"][k].append(ndcg_at_k(retrieved_gains, gallery_gains, k))

        if ax in DISCRETE_AXES:
            rel = axes.relevant(ax, qi)
            if exclude_self:
                rel[qi] = False
            num_rel = int(rel.sum())
            if num_rel == 0:
                skipped[ax] += 1
                continue
            retrieved_rel = rel[ret_rows] if len(ret_rows) else np.empty(0, dtype=bool)
            for k in k_values:
                metrics[ax]["recall"][k].append(
                    recall_at_k(retrieved_rel, num_rel, k)
                )
                metrics[ax]["map"][k].append(
                    average_precision_at_k(retrieved_rel, num_rel, k)
                )


# ----------------------------------------------------------------------
# Persistenza dei valori per-query (opzionale).
# ----------------------------------------------------------------------

def perquery_context(config, image_paths) -> dict | None:
    """Contesto per il salvataggio dei valori per-query, o None se disattivato.

    Governato da UNA sola chiave di config, `eval.perquery_dir`: null (default)
    = nessun file scritto e comportamento identico a prima; un path = una
    matrice per-query per ogni run di valutazione.
    """
    out_dir = config.eval.get("perquery_dir")
    if not out_dir:
        return None
    names = [Path(p).stem for p in image_paths]
    return {
        "dir": Path(out_dir),
        "tag": f"{config.model.name}_{config.model.variant}_{transform_tag(config)}",
        "split": config.eval.get("split") or "all",
        "seed": int(config.eval.seed),
        "gallery": {
            "n": len(names),
            "sha1": gallery_sha1(names),
            "source": str(config.retrieval.data_dir),
        },
    }


def _label_slug(label: str) -> str:
    """Etichetta di run -> frammento di nome file ('random f=0.5' -> 'random-f0.5')."""
    return re.sub(r"[^0-9A-Za-z.]+", "-", label.replace("=", "")).strip("-")


def _write_perquery(recorder, ctx, partial_label: str | None = None) -> None:
    """Scrive il .npz per-query di UNA run.

    Nome file: `vision_<tag>_<mode>_<split>.npz`, con `tag` =
    encoder_variante_contributo (la stessa convenzione di namespacing delle
    visualizzazioni); nel partial la modalità include la strategia, altrimenti
    le run della curva di masking si sovrascriverebbero.

    Side effects: crea la cartella e scrive il file.
    """
    mode = "full" if partial_label is None else "partial"
    slug = mode if partial_label is None else f"partial-{_label_slug(partial_label)}"
    path = ctx["dir"] / f"vision_{ctx['tag']}_{slug}_{ctx['split']}.npz"
    recorder.write(
        path,
        meta={
            "branch": "vision",
            "run_tag": ctx["tag"],
            "mode": mode,
            "partial_label": partial_label,
            "split": ctx["split"],
            "exclude_self": partial_label is None,
            "query_seed": ctx["seed"],
            "gallery": ctx["gallery"],
        },
    )
    print(f"[evaluate] valori per-query salvati in {path}")


# ----------------------------------------------------------------------
# Valutazione full (query = immagine completa).
# ----------------------------------------------------------------------

def evaluate(pipeline, axes, stem2row, query_rows, image_paths, k_values,
             perquery=None) -> None:
    max_k = max(k_values)
    metrics = _new_metrics(k_values)
    skipped = {ax: 0 for ax in DISCRETE_AXES}
    n_query_valid = 0
    recorder = PerQueryRecorder(k_values, max_k) if perquery else None

    for qi in query_rows:
        if not axes.valid[qi]:
            continue
        n_query_valid += 1
        ret_rows = retrieved_rows_for_query(
            pipeline, image_paths[qi], qi, stem2row, max_k
        )
        skipped_before = dict(skipped)   # delta = assi su cui questa query è saltata
        _accumulate_axes(metrics, skipped, axes, qi, ret_rows, k_values, exclude_self=True)
        if recorder is not None:
            recorder.add(
                name=Path(image_paths[qi]).stem, qi=qi, ret_rows=ret_rows,
                metrics=metrics, skipped_before=skipped_before, skipped=skipped,
                axes=axes, exclude_self=True,
            )

    print(f"[evaluate] query valutate: {n_query_valid}")
    for ax in DISCRETE_AXES:
        if skipped[ax]:
            print(
                f"[evaluate] {ax}: {skipped[ax]} query escluse da Recall/mAP "
                "(classe di equivalenza singleton, nessun rilevante)"
            )
    print()
    _print_axis_tables(metrics, k_values)
    if recorder is not None:
        _write_perquery(recorder, perquery)


# ----------------------------------------------------------------------
# Valutazione partial (query = pianta degradata).
# ----------------------------------------------------------------------

def partial_runs(pcfg) -> list[tuple[str, str, dict]]:
    """Espande la config in una lista di (label, strategia, params).
    Il random si espande in una run per frazione (curva del masking-level)."""
    runs = []
    s = pcfg.strategies
    if s.get("random", {}).get("enabled", False):
        for f in s.random.fractions:
            runs.append((f"random f={f}", "random", {"fraction": float(f)}))
    if s.get("semantic", {}).get("enabled", False):
        runs.append(("semantic", "semantic", {"keep_types": list(s.semantic.keep_types)}))
    if s.get("topology", {}).get("enabled", False):
        runs.append(("topology", "topology", {"max_degree": int(s.topology.get("max_degree", 1))}))
    return runs


def evaluate_partial(pipeline, axes, stem2row, query_rows, image_paths, k_values, pcfg,
                     perquery=None) -> None:
    """Per ogni strategia/livello: query parziale -> retrieval -> metriche.

    Due viste: self-recovery (ritrova la pianta originale?) e per-asse rispetto
    alla pianta COMPLETA (degrado graduale su composizione/topologia/geometria).
    """
    max_k = max(k_values)
    seed = int(pcfg.get("seed", 42))
    open_boundary = bool(pcfg.get("open_boundary", True))
    runs = partial_runs(pcfg)

    print(f"[evaluate] PARTIAL | {len(runs)} run | open_boundary={open_boundary}\n")

    for label, strat, params in runs:
        metrics = _new_metrics(k_values)
        skipped = {ax: 0 for ax in DISCRETE_AXES}
        selfrec = {"recall": {k: [] for k in k_values}, "rr": []}
        recorder = PerQueryRecorder(k_values, max_k, with_self_rr=True) if perquery else None
        n_valid = 0
        n_empty = 0  # query senza stanze rimosse (degenerano nella query completa)

        for qi in query_rows:
            if not axes.valid[qi]:
                continue
            meta = load_metadata(image_paths[qi])
            if meta is None:
                continue
            n_valid += 1

            rng = random.Random(seed + qi)
            img, removed = make_partial_query(
                image_paths[qi], meta, strat, params, rng, open_boundary
            )
            if not removed:
                n_empty += 1

            results = pipeline.query(pipeline.transform(img), top_k=max_k + 1)
            ret_rows = _rows_from_results(results, qi, stem2row, max_k, exclude_self=False)

            skipped_before = dict(skipped)   # delta = assi su cui questa query è saltata
            _accumulate_axes(metrics, skipped, axes, qi, ret_rows, k_values, exclude_self=False)

            ret_arr = np.asarray(ret_rows, dtype=int)
            hit = np.where(ret_arr == qi)[0]
            rank = int(hit[0]) + 1 if len(hit) else None
            selfrec["rr"].append(1.0 / rank if rank else 0.0)
            for k in k_values:
                selfrec["recall"][k].append(1.0 if (rank and rank <= k) else 0.0)

            if recorder is not None:
                recorder.add(
                    name=Path(image_paths[qi]).stem, qi=qi, ret_rows=ret_rows,
                    metrics=metrics, skipped_before=skipped_before, skipped=skipped,
                    axes=axes, exclude_self=False, self_rr=selfrec["rr"][-1],
                )

        _print_partial_report(label, metrics, skipped, selfrec, n_valid, n_empty, k_values)
        if recorder is not None:
            _write_perquery(recorder, perquery, partial_label=label)


# ----------------------------------------------------------------------
# Report.
# ----------------------------------------------------------------------

def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _coverage(metrics, ax, k) -> float:
    """Frazione di query su cui Recall/mAP sono definiti su questo asse.

    Ogni query valutata appende un nDCG; solo quelle con almeno un rilevante
    appendono Recall/mAP. Il rapporto fra le due lunghezze è quindi la copertura,
    senza toccare l'accumulo né passare parametri in più.
    """
    n_all = len(metrics[ax]["ndcg"][k])
    return len(metrics[ax]["recall"][k]) / n_all if n_all else float("nan")


def _print_axis_tables(metrics, k_values) -> None:
    """Tabelle per-asse, una per profondità k.

    Due colonne hanno un nome che va letto con attenzione:

    - `Rec/Prec`: `recall_at_k` normalizza per `min(k, #rilevanti)`, quindi sulle
      query con almeno k rilevanti quel numero È Precision@k, non recall. Sulla
      composizione le classi di equivalenza contano migliaia di piante, cioè
      siamo quasi sempre in quel regime (dettaglio in `src/evaluation/metrics.py`).
    - `cop.`: le query con classe di equivalenza singleton non hanno rilevanti ed
      escono dalla media. Sotto il 100% la media descrive un SOTTOINSIEME scelto
      dalla ground truth — le piante con struttura comune — ed è quindi
      ottimista in valore assoluto (il confronto fra sistemi resta appaiato: chi
      esce non dipende dal modello).

    La ripartizione esatta per asse e per k la stampa
    `python -m src.evaluation.metric_diagnostics` dai file per-query.
    """
    for k in k_values:
        print(f"================  K = {k}  ================")
        header = f"{'asse':<13} {'nDCG':>8} {'Rec/Prec':>9} {'mAP':>8} {'cop.':>7}"
        print(header)
        print("-" * len(header))
        for ax in AXES:
            ndcg = _mean(metrics[ax]["ndcg"][k])
            if ax in DISCRETE_AXES:
                rec = _mean(metrics[ax]["recall"][k])
                mapk = _mean(metrics[ax]["map"][k])
                print(f"{ax:<13} {ndcg:>8.3f} {rec:>9.3f} {mapk:>8.3f} "
                      f"{100*_coverage(metrics, ax, k):>6.1f}%")
            else:
                print(f"{ax:<13} {ndcg:>8.3f} {'—':>9} {'—':>8} {'—':>7}")
        print("Rec/Prec: con #rilevanti >= k è Precision@k · cop.: query su cui "
              "Recall/mAP sono definiti")
        print()


def _print_partial_report(label, metrics, skipped, selfrec, n_valid, n_empty, k_values) -> None:
    """Report di una run partial: self-recovery + tabelle per-asse.

    ⚠️ Le tabelle per-asse stampate qui NON sono confrontabili con quelle della
    modalità full: nel partial la query è degradata e ritrovare l'originale è un
    successo legittimo, quindi il self resta nella gallery, fra i rilevanti e
    nell'IDCG (`_accumulate_axes(..., exclude_self=False)`). Il self viene quasi
    sempre recuperato in cima con gain 1.0, il che alza strutturalmente l'nDCG.
    Le due modalità si leggono ciascuna al proprio interno.
    """
    print(f"################  PARTIAL [{label}]  ################")
    print(f"query valutate: {n_valid} (senza masking: {n_empty})")
    for ax in DISCRETE_AXES:
        if skipped[ax]:
            print(f"  {ax}: {skipped[ax]} query escluse da Recall/mAP (classe singleton)")
    mrr = _mean(selfrec["rr"])
    rec_line = "  ".join(f"R@{k}={_mean(selfrec['recall'][k]):.3f}" for k in k_values)
    print(f"self-recovery (pianta originale): MRR={mrr:.3f}  {rec_line}")
    print()
    _print_axis_tables(metrics, k_values)
    print("⚠️  partial: self INCLUSO in gallery/rilevanti/IDCG -> le tabelle "
          "per-asse qui sopra non sono")
    print("    confrontabili con quelle della modalità full."
          + (" Questa run non rimuove nulla: la query è la pianta intatta."
             if n_valid and n_empty == n_valid else ""))
    print()


def main():
    args, overrides = parse_args()
    config = load_vision_config(args.config, overrides)
    k_values = tuple(getattr(config.eval, "k_values", DEFAULT_K_VALUES))

    pcfg = getattr(config, "partial", None)
    partial_on = args.partial or (pcfg is not None and bool(pcfg.get("enabled", False)))
    if partial_on and pcfg is None:
        raise SystemExit("--partial richiesto ma manca il blocco `partial` nel config")
    mode = "PARTIAL" if partial_on else "FULL"

    print(
        f"[evaluate] encoder = {config.model.name} | variante = {config.model.variant} | "
        f"contributo = {transform_tag(config)} | modalità = {mode} | "
        "rilevanza per-asse dai .mat (composizione/topologia/geometria).\n"
    )

    pipeline = load_pipeline(config)
    image_paths = pipeline.image_paths
    axes, stem2row = build_gallery_axes(image_paths)
    query_rows = sample_query_rows(
        image_paths,
        config.eval.num_queries,
        config.eval.seed,
        split=getattr(config.eval, "split", None),
    )

    perquery = perquery_context(config, image_paths)

    if partial_on:
        evaluate_partial(pipeline, axes, stem2row, query_rows, image_paths, k_values, pcfg,
                         perquery=perquery)
    else:
        evaluate(pipeline, axes, stem2row, query_rows, image_paths, k_values,
                 perquery=perquery)


if __name__ == "__main__":
    main()
