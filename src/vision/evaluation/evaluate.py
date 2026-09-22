# src/vision/evaluation/evaluate.py

import argparse
import random
import re
from pathlib import Path

import numpy as np

from src.data.rplan_metadata import get_split, load_metadata, split_row_indices
from src.evaluation.gallery_join import load_shared_names, restrict_rows
from src.vision.data.vision_damage import damaged_query, make_patch_context, resolve_patch_size
from src.evaluation.metrics import (
    average_precision_at_k,
    ndcg_at_k,
    recall_at_k,
)
from src.evaluation.perquery import PerQueryRecorder, gallery_sha1
from src.evaluation.query_vectors import QueryVectorRecorder, array_sha1
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


def restrict_gallery(pipeline, config) -> None:
    """Restringe la gallery all'inner join fra i rami (fase **B.3**), in place.

    Governato da `eval.gallery_names`: null (default) = gallery intera, nessun
    cambiamento. Con un path, la gallery viene **ristretta e riordinata**
    sull'ordine canonico del file condiviso — l'ordine conta, perché
    `gallery_sha1` ci dipende: è così che i due rami arrivano allo **stesso**
    hash e il confronto diventa appaiato per costruzione.

    Va chiamata PRIMA di `prepare_index`: whitening e indice devono vedere la
    stessa gallery delle metriche.
    """
    path = config.eval.get("gallery_names")
    if not path:
        return

    shared = load_shared_names(path)
    rows = restrict_rows([Path(p).stem for p in pipeline.image_paths], shared)
    before = len(pipeline.image_paths)
    pipeline.raw_embeddings = pipeline.raw_embeddings[np.asarray(rows, dtype=np.int64)]
    pipeline.image_paths = [pipeline.image_paths[i] for i in rows]
    pipeline.embeddings = pipeline.raw_embeddings
    print(f"[evaluate] gallery ristretta all'inner join (B.3): "
          f"{before} -> {len(pipeline.image_paths)} righe, ordine canonico da {path}")


def whitening_fit_rows(config, image_paths: list[str]) -> list[int] | None:
    """Righe su cui stimare il whitening, secondo `whitening.fit_split`.

    - `train` (default dal 25 ago 2026, fase B.2): solo le piante dello split
      train, come impone il vincolo DURO 1. La gallery resta intera: si
      restringe l'insieme di STIMA, non il corpus di ricerca.
    - `all`: protocollo trasduttivo (stima su tutta la gallery). È quello di
      tutte le run fino al 24 ago 2026 e resta disponibile come termine di
      confronto appaiato — non come default.

    Returns:
        Lista di indici, oppure None quando non c'è niente da restringere
        (whitening spento, oppure `fit_split: all`).
    """
    if not config.whitening.enabled:
        return None

    fit_split = config.whitening.get("fit_split") or "all"
    if fit_split == "all":
        print("[evaluate] whitening TRASDUTTIVO: stima su tutta la gallery "
              "(whitening.fit_split=all)")
        return None

    rows = split_row_indices(image_paths, fit_split)
    print(f"[evaluate] whitening stimato sullo split '{fit_split}': "
          f"{len(rows)}/{len(image_paths)} righe della gallery")
    return rows


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
    restrict_gallery(pipeline, config)   # B.3: prima di head e whitening

    head = None
    hcfg = getattr(config, "head", None)
    if hcfg is not None and hcfg.get("enabled"):
        in_dim = pipeline.raw_embeddings.shape[1]
        head = load_head(config.retrieval.save_dir, in_dim,
                         hcfg.hidden_dim, hcfg.out_dim, manager.device,
                         filename=hcfg.get("file") or "head.pt")

    pipeline.prepare_index(
        head=head,
        whiten=config.whitening.enabled,
        eps=config.whitening.eps,
        whiten_dim=config.whitening.get("dim"),
        fit_rows=whitening_fit_rows(config, pipeline.image_paths),
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


def partial_rows(results, qi, stem2row, max_k) -> tuple[list[int], list[int]]:
    """Dalle STESSE risposte FAISS, le due liste che servono al partial (B.4).

    Nel partial la query è una pianta degradata e l'originale è ancora in
    gallery: le due viste la trattano in modo opposto e vanno separate.

    Returns:
        (self_rows, axis_rows)
        - `self_rows`: il self RESTA. È il self-recovery, dove ritrovare
          l'originale È il compito (endpoint del criterio A.5).
        - `axis_rows`: il self è TOLTO. Sono le metriche per-asse, che misurano
          il degrado rispetto alle *altre* piante e devono essere confrontabili
          col full, dove il self è sempre escluso (rilievo B3).

    Entrambe restano lunghe fino a `max_k` perché il filtro parte dalle
    `max_k + 1` risposte richieste a FAISS: togliere il self non accorcia la
    lista, la fa scorrere.
    """
    self_rows = _rows_from_results(results, qi, stem2row, max_k, exclude_self=False)
    axis_rows = _rows_from_results(results, qi, stem2row, max_k, exclude_self=True)
    return self_rows, axis_rows


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

    exclude_self=True: self escluso da gallery, rilevanti e risultati. Dal 25
    ago 2026 (B.4) è la modalità di **entrambe** le viste per-asse, full e
    partial: solo così le due curve sono confrontabili.
    exclude_self=False: il self resta nei rilevanti e nell'IDCG. Non più usato
    dal ramo vision; la vista in cui ritrovare l'originale è un successo è il
    **self-recovery**, che ha un endpoint suo (`self_rr`) e non passa di qui.
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


def query_vectors_context(config, perquery, raw_embeddings=None) -> dict | None:
    """Context for saving the damaged query vectors (`qvec/1`), or None if off.

    Governed by `eval.query_vectors_dir` (late fusion, 16 Sep 2026): null
    (default) = nothing written, behaviour identical to before. The vectors are
    tied to the per-query file of the same run, so `eval.perquery_dir` is
    required.

    Args:
        config:         vision DictConfig.
        perquery:       context from `perquery_context` (None if disabled).
        raw_embeddings: [N, D] RAW gallery AFTER `restrict_gallery` (the rows
                        the whitening and the index were built from); its sha1
                        pins the gallery the query vectors belong to.

    Raises:
        ValueError: `eval.query_vectors_dir` set without `eval.perquery_dir`.
    """
    out_dir = config.eval.get("query_vectors_dir")
    if not out_dir:
        return None
    if perquery is None:
        raise ValueError(
            "eval.query_vectors_dir requires eval.perquery_dir: the query vectors "
            "are paired with the per-query file of the same run"
        )
    hcfg = getattr(config, "head", None)
    head = None
    if hcfg is not None and hcfg.get("enabled"):
        head = {"enabled": True, "file": hcfg.get("file") or "head.pt"}
    ctx = {
        "dir": Path(out_dir),
        "whitening": {
            "enabled": bool(config.whitening.enabled),
            "fit_split": config.whitening.get("fit_split") or "all",
            "eps": float(config.whitening.eps),
            "dim": config.whitening.get("dim"),
        },
        "head": head,
        "gallery_vectors": None,
    }
    if raw_embeddings is not None:
        ctx["gallery_vectors"] = {
            "path": str(Path(config.retrieval.save_dir) / "embeddings.npy"),
            "sha1": array_sha1(raw_embeddings),
            "shape": list(raw_embeddings.shape),
        }
    return ctx


def _label_slug(label: str) -> str:
    """Etichetta di run -> frammento di nome file ('random f=0.5' -> 'random-f0.5')."""
    return re.sub(r"[^0-9A-Za-z.]+", "-", label.replace("=", "")).strip("-")


def _write_perquery(recorder, ctx, partial_label: str | None = None,
                    damage: dict | None = None) -> Path:
    """Scrive il .npz per-query di UNA run (ritorna il path scritto).

    Nome file: `vision_<tag>_<mode>_<split>.npz`, con `tag` =
    encoder_variante_contributo (la stessa convenzione di namespacing delle
    visualizzazioni); nel partial la modalità include la strategia, altrimenti
    le run della curva di masking si sovrascriverebbero.

    `damage` (solo partial, 11 set 2026): descrizione del danno applicato, finisce
    in `meta["damage"]` (chiave additiva; assente = file come prima).

    Side effects: crea la cartella e scrive il file.
    """
    mode = "full" if partial_label is None else "partial"
    slug = mode if partial_label is None else f"partial-{_label_slug(partial_label)}"
    path = ctx["dir"] / f"vision_{ctx['tag']}_{slug}_{ctx['split']}.npz"
    meta = {
        "branch": "vision",
        "run_tag": ctx["tag"],
        "mode": mode,
        "partial_label": partial_label,
        "split": ctx["split"],
        # Dal 25 ago (B.4) le metriche per-asse escludono il self anche nel
        # partial: la chiave resta nel meta perché `significance.py` la usa
        # per rifiutare il confronto con i file scritti PRIMA (dove era False).
        "exclude_self": True,
        "query_seed": ctx["seed"],
        "gallery": ctx["gallery"],
    }
    if damage is not None:
        meta["damage"] = damage
    recorder.write(path, meta=meta)
    print(f"[evaluate] valori per-query salvati in {path}")
    return path


def _write_query_vectors(qrec, qctx, perquery, perquery_path, pipeline, label, strat,
                         params, partial_seed, k_values) -> None:
    """Writes the `qvec/1` file of ONE partial run, next to its per-query file.

    Same file name as the per-query file, in `eval.query_vectors_dir`. The
    whitening parameters are the job's own (`pipeline.whiten_mean/matrix`,
    None when whitening is off), so the fusion whitens with exactly them.

    Side effects: creates the folder and writes the file.
    """
    path = qctx["dir"] / Path(perquery_path).name
    meta = {
        "branch": "vision",
        "run_tag": perquery["tag"],
        "perquery_file": str(perquery_path),
        "mode": "partial",
        "partial_label": label,
        "damage": {"strategy": strat, "params": dict(params)},
        "split": perquery["split"],
        "query_seed": perquery["seed"],
        "partial_seed": int(partial_seed),
        "k_values": list(k_values),
        "gallery": perquery["gallery"],
        "gallery_vectors": qctx["gallery_vectors"],
        "whitening": qctx["whitening"],
        "head": qctx["head"],
    }
    qrec.write(path, meta, whiten_mean=pipeline.whiten_mean,
               whiten_matrix=pipeline.whiten_matrix)
    print(f"[evaluate] vettori delle query salvati in {path}")


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
    Il random si espande in una run per frazione (curva del masking-level).

    Crop e patch (11 set 2026, `vision_damage.py`) si accodano DOPO topology e
    solo se abilitati: con le loro chiavi assenti o spente la lista è quella di
    prima, nello stesso ordine (contratto con il ramo graph, test_contracts).

    Le `nowalls_*` (11 set 2026) si accodano per ultime: stessa selezione delle
    stanze delle strategie storiche omonime, ma il rendering cancella anche i
    muri della stanza tolta. Etichette distinte = file per-query distinti, quindi
    le run storiche restano confrontabili con quelle già su disco."""
    runs = []
    s = pcfg.strategies
    if s.get("random", {}).get("enabled", False):
        for f in s.random.fractions:
            runs.append((f"random f={f}", "random", {"fraction": float(f)}))
    if s.get("semantic", {}).get("enabled", False):
        runs.append(("semantic", "semantic", {"keep_types": list(s.semantic.keep_types)}))
    if s.get("topology", {}).get("enabled", False):
        runs.append(("topology", "topology", {"max_degree": int(s.topology.get("max_degree", 1))}))
    crop = s.get("crop", {})
    if crop.get("enabled", False):
        for f in crop.fractions:
            runs.append((f"crop f={f}", "crop", {
                "fraction": float(f),
                "tolerance": float(crop.get("tolerance", 0.05)),
                "max_tries": int(crop.get("max_tries", 10000)),
            }))
    patch = s.get("patch", {})
    if patch.get("enabled", False):
        for f in patch.fractions:
            runs.append((f"patch f={f}", "patch", {
                "fraction": float(f),
                "tolerance": float(patch.get("tolerance", 0.05)),
            }))
    nw = s.get("nowalls_random", {})
    if nw.get("enabled", False):
        for f in nw.fractions:
            runs.append((f"nowalls-random f={f}", "nowalls_random", {"fraction": float(f)}))
    nw = s.get("nowalls_semantic", {})
    if nw.get("enabled", False):
        runs.append(("nowalls-semantic", "nowalls_semantic",
                     {"keep_types": list(nw.keep_types)}))
    nw = s.get("nowalls_topology", {})
    if nw.get("enabled", False):
        runs.append(("nowalls-topology", "nowalls_topology",
                     {"max_degree": int(nw.get("max_degree", 1))}))
    return runs


def _transform_image_size(transform) -> int | None:
    """Lato S del canvas del transform (ResizeWithPad in testa), None se ignoto."""
    steps = getattr(transform, "transforms", None) or []
    return int(steps[0].target_size) if steps and hasattr(steps[0], "target_size") else None


def evaluate_partial(pipeline, axes, stem2row, query_rows, image_paths, k_values, pcfg,
                     perquery=None, patch_size=None, query_vectors=None) -> None:
    """Per ogni strategia/livello: query parziale -> retrieval -> metriche.

    Due viste: self-recovery (ritrova la pianta originale?) e per-asse rispetto
    alla pianta COMPLETA (degrado graduale su composizione/topologia/geometria).

    La query degradata la costruisce `vision_damage.damaged_query`: per le
    strategie a stanze sono le stesse due chiamate di prima (query bit-identiche),
    in più si misura la frazione di pianta rimossa (`area_removed`). `patch_size`
    serve solo alle run `patch` (lo risolve `main` con `resolve_patch_size`).

    `query_vectors` (context from `query_vectors_context`, late fusion): also
    save the vectors of the damaged queries, from the same forward. Only runs
    whose damage reports `removed` rooms (room and nowalls strategies) are
    saved; crop/patch runs are skipped with one log line. None = unchanged.
    """
    if query_vectors is not None and perquery is None:
        raise ValueError("query_vectors requires perquery (same run, same file name)")
    max_k = max(k_values)
    seed = int(pcfg.get("seed", 42))
    open_boundary = bool(pcfg.get("open_boundary", True))
    runs = partial_runs(pcfg)
    patch_ctx = None
    if any(strat == "patch" for _, strat, _ in runs):
        if patch_size is None:
            raise ValueError("run `patch` richieste ma patch_size non risolto")
        patch_ctx = make_patch_context(pipeline.transform, patch_size)

    print(f"[evaluate] PARTIAL | {len(runs)} run | open_boundary={open_boundary}\n")

    for label, strat, params in runs:
        metrics = _new_metrics(k_values)
        skipped = {ax: 0 for ax in DISCRETE_AXES}
        selfrec = {"recall": {k: [] for k in k_values}, "rr": []}
        recorder = (PerQueryRecorder(k_values, max_k, with_self_rr=True, with_area_removed=True)
                    if perquery else None)
        n_valid = 0
        n_empty = 0  # query senza stanze rimosse (degenerano nella query completa)
        areas = []   # frazione di pianta rimossa, una per query valutata
        n_fallback = 0
        qrec = QueryVectorRecorder() if query_vectors is not None else None

        for qi in query_rows:
            if not axes.valid[qi]:
                continue
            meta = load_metadata(image_paths[qi])
            if meta is None:
                continue
            n_valid += 1

            rng = random.Random(seed + qi)
            query_tensor, dinfo = damaged_query(
                image_paths[qi], meta, strat, params, rng, open_boundary,
                pipeline.transform, patch_ctx,
            )
            if not dinfo["removed_any"]:
                n_empty += 1
            areas.append(dinfo["area_removed"])
            n_fallback += int(dinfo["fallback"])

            if qrec is not None and "removed" in dinfo:
                results, query_raw, query_final = pipeline.query(
                    query_tensor, top_k=max_k + 1, return_embeddings=True)
                qrec.add(name=Path(image_paths[qi]).stem, qi=qi, vector=query_raw[0],
                         removed=dinfo["removed"], vector_final=query_final[0])
            else:
                results = pipeline.query(query_tensor, top_k=max_k + 1)
            # B.4 (25 ago): due liste, due domande. Il self resta solo nella
            # vista self-recovery; le metriche per-asse lo escludono, come il full.
            self_rows, axis_rows = partial_rows(results, qi, stem2row, max_k)

            skipped_before = dict(skipped)   # delta = assi su cui questa query è saltata
            _accumulate_axes(metrics, skipped, axes, qi, axis_rows, k_values, exclude_self=True)

            ret_arr = np.asarray(self_rows, dtype=int)
            hit = np.where(ret_arr == qi)[0]
            rank = int(hit[0]) + 1 if len(hit) else None
            selfrec["rr"].append(1.0 / rank if rank else 0.0)
            for k in k_values:
                selfrec["recall"][k].append(1.0 if (rank and rank <= k) else 0.0)

            if recorder is not None:
                # righe e flag coerenti con l'accumulo per-asse (self escluso);
                # il self-recovery viaggia a parte, nello scalare `self_rr`.
                recorder.add(
                    name=Path(image_paths[qi]).stem, qi=qi, ret_rows=axis_rows,
                    metrics=metrics, skipped_before=skipped_before, skipped=skipped,
                    axes=axes, exclude_self=True, self_rr=selfrec["rr"][-1],
                    area_removed=dinfo["area_removed"],
                )

        _print_partial_report(label, metrics, skipped, selfrec, n_valid, n_empty, k_values,
                              areas=areas, n_fallback=n_fallback)
        if recorder is not None:
            damage = {
                "strategy": strat,
                "params": dict(params),
                "patch_size": patch_ctx["patch_size"] if strat == "patch" else None,
                "image_size": _transform_image_size(pipeline.transform),
                "area_space": "resized" if strat == "patch" else "native",
                "n_fallback": n_fallback,
            }
            perquery_path = _write_perquery(recorder, perquery, partial_label=label,
                                            damage=damage)
            if qrec is not None and len(qrec):
                _write_query_vectors(qrec, query_vectors, perquery, perquery_path, pipeline,
                                     label, strat, params, seed, k_values)
            elif qrec is not None:
                print(f"[evaluate] vettori delle query NON salvati per [{label}]: "
                      "il danno non toglie stanze (niente `removed`)")


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


def _print_partial_report(label, metrics, skipped, selfrec, n_valid, n_empty, k_values,
                          areas=None, n_fallback=0) -> None:
    """Report di una run partial: self-recovery + tabelle per-asse.

    Dal 25 ago 2026 (**B.4**) le due viste sono separate e le tabelle per-asse
    SONO confrontabili con quelle del full: il self è escluso da risultati,
    rilevanti e IDCG in entrambe le modalità. Ritrovare l'originale resta un
    successo, ma si misura dove è il suo compito — nel **self-recovery** qui
    sopra (MRR/Recall), non dentro le metriche per-asse.

    ⚠️ I file per-query scritti PRIMA del 25 ago hanno il self dentro
    (`exclude_self=False` nel meta): i loro numeri per-asse sono più alti e non
    vanno mischiati con questi. `significance.py` rifiuta il confronto da solo.
    """
    print(f"################  PARTIAL [{label}]  ################")
    print(f"query valutate: {n_valid} (senza masking: {n_empty})")
    if areas:
        print(f"area di pianta rimossa: media={np.mean(areas):.3f} "
              f"min={np.min(areas):.3f} max={np.max(areas):.3f} · fallback={n_fallback}")
    for ax in DISCRETE_AXES:
        if skipped[ax]:
            print(f"  {ax}: {skipped[ax]} query escluse da Recall/mAP (classe singleton)")
    mrr = _mean(selfrec["rr"])
    rec_line = "  ".join(f"R@{k}={_mean(selfrec['recall'][k]):.3f}" for k in k_values)
    print(f"self-recovery (pianta originale): MRR={mrr:.3f}  {rec_line}")
    print()
    _print_axis_tables(metrics, k_values)
    print("ℹ️  partial (B.4): self ESCLUSO da risultati/rilevanti/IDCG -> le "
          "tabelle per-asse qui sopra sono")
    print("    confrontabili con quelle della modalità full; il self si misura "
          "nel self-recovery."
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
    query_vectors = query_vectors_context(config, perquery, pipeline.raw_embeddings)
    if query_vectors is not None and not partial_on:
        print("[evaluate] eval.query_vectors_dir ignorato in modalità full "
              "(i vettori della gallery sono già su disco)")

    if partial_on:
        patch_size = None
        if any(strat == "patch" for _, strat, _ in partial_runs(pcfg)):
            # Prima della prima query: un preset sbagliato deve fallire subito.
            patch_size = resolve_patch_size(
                pipeline.encoder, config.model.kwargs.get("patch_size"),
                _transform_image_size(pipeline.transform),
            )
            print(f"[evaluate] danno patch: patch_size={patch_size}\n")
        evaluate_partial(pipeline, axes, stem2row, query_rows, image_paths, k_values, pcfg,
                         perquery=perquery, patch_size=patch_size,
                         query_vectors=query_vectors)
    else:
        evaluate(pipeline, axes, stem2row, query_rows, image_paths, k_values,
                 perquery=perquery)


if __name__ == "__main__":
    main()
