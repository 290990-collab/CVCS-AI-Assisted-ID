# src/evaluation/random_floor.py

"""
Floor delle metriche per-asse: quanto vale nDCG/Recall/mAP per un sistema che
non sa niente?

Serve a leggere i numeri dei due rami: su assi con gain graduati e classi di
equivalenza grandi lo zero teorico non e' il riferimento giusto, perche' anche
un ranking cieco porta a casa qualcosa.

**Due null diversi**, perche' rispondono a due domande diverse:

- `random` (default): ranking estratto a sorte, DIVERSO per ogni query. E' il
  "non sa niente" puro.
- `constant` (`--constant-ranking`): le STESSE righe restituite a tutte le
  query, sorteggiate una volta per seed. Su un asse saturo il sistema stupido
  piu' pericoloso non e' quello casuale ma quello che ignora la query e
  restituisce sempre le piante "generiche": se questo null va vicino ai modelli,
  la metrica su quell'asse **non discrimina**, e il punteggio alto non e' merito
  del modello. E' l'unico modo per distinguere "l'asse e' facile" da "il modello
  e' bravo".

Tre scelte che lo rendono onesto:

- **stesso codice metrico** della valutazione vera (`accumulate_axes` del ramo
  graph, importato: se lo re-implementassi qui il confronto perderebbe senso);
- ranking = `max_k` righe **distinte** (senza rimpiazzo) estratte da *tutte* le
  righe diverse dalla query, comprese le piante senza `.mat` (gain 0):
  escluderle sarebbe informazione che un sistema cieco non ha, e gonfierebbe il
  floor;
- **stesse query** del resto del progetto (stesso campionamento riproducibile),
  cosi' il floor e' appaiato alle run vere.

Il floor **non e' un numero universale**: l'IDCG dipende dalla gallery, quindi
va calcolato per ogni gallery (vision 67.453 righe, graph 67.405).

Costo: la parte pesante e' la similarita' d'asse su tutta la gallery, che dipende
solo dalla query e NON dal ranking. Il loop e' quindi unico sulle query (i seed
scorrono all'interno) e le similarita' si calcolano una volta sola per query:
tutti i seed le riusano. Resta ripetuto per seed solo il calcolo delle metriche
dentro `accumulate_axes`/`ndcg_at_k` (l'ordinamento ideale per l'IDCG), che e'
codice condiviso e non si tocca.

Uso tipico (i due null si lanciano separatamente):

    python -m src.evaluation.random_floor \
        --gallery embeddings/<...>/image_paths.json --split test \
        --num-queries 2000 --query-seed 42 --ranking-seeds 0 1 2 3 4
    python -m src.evaluation.random_floor ... --constant-ranking
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from src.data.rplan_metadata import load_metadata
from src.evaluation.perquery import PerQueryRecorder, gallery_sha1
from src.evaluation.relevance import AXES, DISCRETE_AXES, GalleryAxes
from src.graph.evaluation.axis_metrics import accumulate_axes, mean_metric, new_metrics

DEFAULT_K_VALUES = (1, 5, 10, 100)
DEFAULT_MAX_K = 100
DEFAULT_OUT_DIR = "results/random_floor"

# Etichette dei due null (finiscono in `run_tag` e nel nome del file per-query).
NULL_RANDOM = "random"
NULL_CONSTANT = "constant"

# Il login node uccide i processi oltre `ulimit -t` secondi di CPU: la tabella
# parziale ogni N query e l'avviso sotto soglia servono a non perdere il lavoro
# gia' fatto quando succede (il posto giusto resta un nodo di calcolo).
LOGIN_NODE_CPU_LIMIT = 600.0
DEFAULT_CPU_WARN_SECONDS = 480.0
DEFAULT_PROGRESS_EVERY = 250


# ----------------------------------------------------------------------
# Gallery: nomi, metadati, feature per-asse.
# ----------------------------------------------------------------------

def load_gallery_entries(gallery_path: str | Path) -> list[str]:
    """Legge la lista di righe della gallery da un JSON.

    Accetta indifferentemente `image_paths.json` (ramo vision, path completi dei
    PNG) e `names.json` (ramo graph, nomi nudi): entrambi sono una lista di
    stringhe allineata all'ordine delle righe.
    """
    entries = json.loads(Path(gallery_path).read_text())
    if not isinstance(entries, list) or not all(isinstance(e, str) for e in entries):
        raise ValueError(
            f"{gallery_path}: atteso un JSON con una lista di stringhe "
            f"(image_paths.json o names.json)"
        )
    return entries


def gallery_names(entries: list[str]) -> list[str]:
    """Nomi canonici delle righe = stem del file.

    E' la chiave di join fra i rami (vision salva path, graph salva nomi): lo
    stem e' l'unica forma comune.
    """
    return [Path(e).stem for e in entries]


def build_gallery_axes(entries: list[str]) -> GalleryAxes:
    """Feature di rilevanza per-asse dai `.mat`, allineate all'ordine della gallery."""
    print(f"[random_floor] costruzione feature per-asse su {len(entries)} piante...")
    axes = GalleryAxes([load_metadata(e) for e in entries])
    print(f"[random_floor] metadati .mat presenti: {int(axes.valid.sum())}/{len(entries)}")
    return axes


def sample_query_rows(entries: list[str], num_queries: int, seed: int, split: str) -> list[int]:
    """Righe-query, con lo STESSO campionamento della valutazione vera.

    Riusa `src.vision.evaluation.evaluate.sample_query_rows` invece di
    ricopiarne la logica: il floor deve stare sulle stesse query dei sistemi che
    deve mettere a terra, e due implementazioni "equivalenti" divergerebbero al
    primo ritocco. L'import e' locale perche' quel modulo tira dentro
    torch/faiss, inutili qui.
    """
    from src.vision.evaluation.evaluate import sample_query_rows as _sample

    return _sample(entries, num_queries, seed, split=split)


# ----------------------------------------------------------------------
# I due ranking null.
# ----------------------------------------------------------------------

def draw_rows(rng: random.Random, n_gallery: int, max_k: int) -> list[int]:
    """`max_k + 1` righe distinte sorteggiate senza rimpiazzo dall'intera gallery.

    Senza rimpiazzo perche' un retriever vero non restituisce due volte lo stesso
    risultato; `max_k + 1` per la stessa contabilita' del retrieval vero, che
    chiede un vicino in piu' a FAISS per compensare il self-match.
    """
    return rng.sample(range(n_gallery), min(max_k + 1, n_gallery))


def drop_self(rows, qi: int, max_k: int) -> list[int]:
    """Toglie il self dal ranking e tronca a `max_k` (esclusione identica nei due null)."""
    return [r for r in rows if r != qi][:max_k]


def random_ret_rows(rng: random.Random, n_gallery: int, qi: int, max_k: int) -> list[int]:
    """Null `random`: righe sorteggiate di nuovo per OGNI query."""
    return drop_self(draw_rows(rng, n_gallery, max_k), qi, max_k)


def draw_ranking(rng: random.Random, n_gallery: int, valid_rows, max_k: int,
                 constant_ranking: bool) -> list[list[int]]:
    """Tutti i `ret_rows` di UN seed, nell'ordine in cui le query verranno valutate.

    Il sorteggio si fa in anticipo (2000x100 interi: memoria trascurabile) perche'
    il loop di valutazione e' unico sulle query e attraversa tutti i seed; farlo
    qui garantisce che l'RNG di ciascun seed venga consumato nella stessa
    sequenza di prima (una `draw_rows` per query valutata, nell'ordine delle
    query; per il null costante un solo sorteggio iniziale).
    """
    if constant_ranking:
        rows = draw_rows(rng, n_gallery, max_k)
        return [drop_self(rows, qi, max_k) for qi in valid_rows]
    return [random_ret_rows(rng, n_gallery, qi, max_k) for qi in valid_rows]


# ----------------------------------------------------------------------
# Valutazione di tutti i seed in una sola passata sulle query.
# ----------------------------------------------------------------------

class QueryAxesCache:
    """Proxy di `GalleryAxes` che memorizza sim/relevant della query CORRENTE.

    Il punto: `sim(asse, qi)` e `relevant(asse, qi)` sono calcoli su TUTTA la
    gallery (67k righe, la matrice topologica e' 67k x 91) e **non dipendono dal
    ranking** — per una data query sono identici per tutti i seed. Con il loop
    esterno sui seed venivano rifatti una volta per seed; con il loop unico sulle
    query si calcolano una volta e li riusano tutti i seed.

    `accumulate_axes` non viene toccato: continua a chiamare `axes.sim(...)`,
    `axes.relevant(...)` e `len(axes)` come sempre, solo che dall'altra parte c'e'
    questo proxy. Gli array restituiti sono gli stessi oggetti che avrebbe
    prodotto `GalleryAxes`, quindi i numeri sono identici bit a bit.
    """

    def __init__(self, axes: GalleryAxes):
        self._axes = axes
        self._qi = None
        self._sim: dict[str, np.ndarray] = {}
        self._relevant: dict[str, np.ndarray] = {}

    def _switch_to(self, qi: int) -> None:
        """Invalida la cache quando cambia la query (la cache tiene una query sola)."""
        if self._qi != qi:
            self._qi = qi
            self._sim.clear()
            self._relevant.clear()

    def __len__(self) -> int:
        return len(self._axes)

    @property
    def valid(self) -> np.ndarray:
        return self._axes.valid

    def sim(self, axis: str, qi: int) -> np.ndarray:
        self._switch_to(qi)
        if axis not in self._sim:
            self._sim[axis] = self._axes.sim(axis, qi)
        return self._sim[axis]

    def relevant(self, axis: str, qi: int) -> np.ndarray:
        """Copia della maschera: il chiamante ci scrive dentro (`rel[qi] = False`)."""
        self._switch_to(qi)
        if axis not in self._relevant:
            self._relevant[axis] = self._axes.relevant(axis, qi)
        return self._relevant[axis].copy()


def run_seeds(axes: GalleryAxes, names, query_rows, k_values, max_k: int, ranking_seeds,
              constant_ranking: bool = False, recorder: PerQueryRecorder | None = None,
              recorder_seed: int | None = None, progress_every: int = 0,
              on_progress=None):
    """Valuta TUTTI i seed in una sola passata sulle query.

    L'accumulo e' quello della valutazione vera (`accumulate_axes`, un
    contenitore per seed) e, se `recorder` e' passato, i valori per-query del
    solo `recorder_seed` vengono registrati accanto all'accumulo, come nei due
    rami: il file per-query e le medie stampate vengono dallo stesso calcolo.

    Args:
        axes:             `GalleryAxes` della gallery.
        names:            nomi allineati alle righe (chiave di join del per-query).
        query_rows:       righe-query candidate (le non valide vengono saltate).
        k_values:         profondita' da valutare.
        max_k:            lunghezza del ranking.
        ranking_seeds:    seed dei sorteggi, valutati tutti sulla stessa passata.
        constant_ranking: True -> null `constant` (un solo sorteggio per seed).
        recorder:         opzionale, riempito con i valori per-query.
        recorder_seed:    quale seed finisce nel recorder (default: il primo).
        progress_every:   ogni quante query invocare `on_progress` (0 = mai).
        on_progress:      callback(n_fatte, n_totali, metrics_per_seed).

    Returns:
        (metrics_per_seed, skipped_per_seed, n_evaluated) — i primi due sono dict
        seed -> contenitore.
    """
    seeds = list(ranking_seeds)
    if recorder_seed is None:
        recorder_seed = seeds[0]

    # Query senza .mat: nessun ground truth, saltate come nella valutazione vera.
    valid_rows = [qi for qi in query_rows if axes.valid[qi]]
    rankings = {
        seed: draw_ranking(random.Random(seed), len(axes), valid_rows, max_k,
                           constant_ranking)
        for seed in seeds
    }

    metrics = {seed: new_metrics(k_values) for seed in seeds}
    skipped = {seed: {ax: 0 for ax in DISCRETE_AXES} for seed in seeds}
    cache = QueryAxesCache(axes)

    for j, qi in enumerate(valid_rows):
        for seed in seeds:
            ret_rows = rankings[seed][j]
            skipped_before = dict(skipped[seed])
            accumulate_axes(metrics[seed], skipped[seed], cache, qi, ret_rows,
                            k_values, exclude_self=True)
            if recorder is not None and seed == recorder_seed:
                recorder.add(name=names[qi], qi=qi, ret_rows=ret_rows,
                             metrics=metrics[seed], skipped_before=skipped_before,
                             skipped=skipped[seed], axes=cache, exclude_self=True)
        if progress_every and on_progress and (j + 1) % progress_every == 0:
            on_progress(j + 1, len(valid_rows), metrics)

    return metrics, skipped, len(valid_rows)


# ----------------------------------------------------------------------
# Aggregazione fra seed e stampa.
# ----------------------------------------------------------------------

def seed_means(metrics, k_values) -> dict:
    """Media sulle query di un singolo seed: (asse, metrica, k) -> float."""
    return {
        (ax, met, k): mean_metric(metrics[ax][met][k])
        for ax in AXES for met in ("ndcg", "recall", "map") for k in k_values
    }


def print_floor_table(all_means: list[dict], k_values) -> None:
    """Tabella asse x k: media fra i seed +/- `mc_std`.

    `mc_std` e' la deviazione standard delle medie dei singoli seed a QUERY
    FISSE: misura solo quanto oscilla il sorteggio del ranking (errore
    Monte-Carlo del null), NON l'incertezza della metrica sulle query e tanto
    meno una soglia di significativita'. Il nome della colonna lo dice, perche'
    un "±" in una tabella viene letto come intervallo di confidenza per abitudine.
    """
    n_seeds = len(all_means)
    for k in k_values:
        print(f"================  K = {k}  ================")
        header = (f"{'asse':<13} {'nDCG mean±mc_std':>20} {'Recall mean±mc_std':>20} "
                  f"{'mAP mean±mc_std':>20}")
        print(header)
        print("-" * len(header))
        for ax in AXES:
            cells = []
            for met in ("ndcg", "recall", "map"):
                if ax not in DISCRETE_AXES and met != "ndcg":
                    cells.append(f"{'—':>20}")
                    continue
                values = np.array([m[(ax, met, k)] for m in all_means], dtype=float)
                cells.append(f"{values.mean():>12.4f}±{values.std():.4f}")
            print(f"{ax:<13} " + " ".join(cells))
        print()
    print(f"mc_std = std delle medie fra i {n_seeds} seed di ranking, a query fisse: "
          "errore Monte-Carlo\ndel sorteggio del null, NON incertezza della metrica "
          "e NON un test di significativita'.\n")


# ----------------------------------------------------------------------
# Entrypoint.
# ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Floor delle metriche per-asse: null casuale o null costante."
    )
    p.add_argument("--gallery", required=True,
                   help="image_paths.json (vision) o names.json (graph): definisce le righe")
    p.add_argument("--split", default="test", choices=["train", "valid", "test", "all"],
                   help="split ufficiale RPLAN da cui pescare le query (la gallery resta intera)")
    p.add_argument("--num-queries", type=int, default=2000, dest="num_queries")
    p.add_argument("--query-seed", type=int, default=42, dest="query_seed",
                   help="seed del campionamento query: deve combaciare con le run vere")
    p.add_argument("--ranking-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                   dest="ranking_seeds",
                   help="seed dei sorteggi (la tabella riporta media +/- mc_std fra seed)")
    p.add_argument("--constant-ranking", action="store_true", dest="constant_ranking",
                   help="null costante: le stesse righe per tutte le query "
                        "(un solo sorteggio per seed) — dice se la metrica discrimina")
    p.add_argument("--k-values", type=int, nargs="+", default=list(DEFAULT_K_VALUES),
                   dest="k_values")
    p.add_argument("--max-k", type=int, default=DEFAULT_MAX_K, dest="max_k",
                   help="profondita' del ranking casuale (>= max dei K)")
    p.add_argument("--out", default=None,
                   help=f"path del .npz per-query del primo seed (default: {DEFAULT_OUT_DIR}/...)")
    p.add_argument("--progress-every", type=int, default=DEFAULT_PROGRESS_EVERY,
                   dest="progress_every",
                   help="ogni quante query stampare la tabella parziale (0 = mai): "
                        "un kill non butta via il lavoro gia' fatto")
    p.add_argument("--cpu-warn-seconds", type=float, default=DEFAULT_CPU_WARN_SECONDS,
                   dest="cpu_warn_seconds",
                   help="soglia di CPU cumulata oltre la quale avvisare che il login "
                        f"node sta per uccidere il processo (default {DEFAULT_CPU_WARN_SECONDS:.0f}s, "
                        f"ulimit -t tipico {LOGIN_NODE_CPU_LIMIT:.0f}s)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_k < max(args.k_values):
        raise ValueError(f"--max-k ({args.max_k}) < max(--k-values) ({max(args.k_values)})")

    null_tag = NULL_CONSTANT if args.constant_ranking else NULL_RANDOM
    entries = load_gallery_entries(args.gallery)
    names = gallery_names(entries)
    sha1 = gallery_sha1(names)
    print(f"[random_floor] null '{null_tag}' — gallery: {len(entries)} righe, "
          f"sha1 {sha1[:12]} ({args.gallery})")

    t0 = time.perf_counter()
    axes = build_gallery_axes(entries)
    t_axes = time.perf_counter() - t0

    split = None if args.split == "all" else args.split
    query_rows = sample_query_rows(entries, args.num_queries, args.query_seed, split)

    # Il per-query si salva solo per il PRIMO seed: e' un esempio di floor, non
    # una media (la dispersione fra seed resta nella tabella come mc_std).
    seed0 = args.ranking_seeds[0]
    recorder = PerQueryRecorder(args.k_values, args.max_k)
    warned = False

    def on_progress(done: int, total: int, metrics_per_seed) -> None:
        """Tabella parziale + CPU cumulata: i seed avanzano insieme, quindi anche
        una passata interrotta lascia numeri leggibili su tutte le query fatte."""
        nonlocal warned
        cpu = time.process_time()
        print(f"[random_floor] parziale: {done}/{total} query — "
              f"CPU cumulata {cpu:.0f}s")
        print_floor_table([seed_means(metrics_per_seed[s], args.k_values)
                           for s in args.ranking_seeds], args.k_values)
        if cpu > args.cpu_warn_seconds and not warned:
            warned = True
            print(f"[random_floor] ATTENZIONE: {cpu:.0f}s di CPU superano la soglia "
                  f"{args.cpu_warn_seconds:.0f}s. Su un login node con "
                  f"ulimit -t {LOGIN_NODE_CPU_LIMIT:.0f} il processo sta per essere "
                  "UCCISO: rilancia su un nodo di calcolo (srun/sbatch) o riduci "
                  "--ranking-seeds / --num-queries.")

    t1 = time.perf_counter()
    metrics, skipped, n_evaluated = run_seeds(
        axes, names, query_rows, args.k_values, args.max_k, args.ranking_seeds,
        constant_ranking=args.constant_ranking, recorder=recorder, recorder_seed=seed0,
        progress_every=args.progress_every, on_progress=on_progress,
    )
    elapsed = time.perf_counter() - t1

    for seed in args.ranking_seeds:
        skip_txt = ", ".join(f"{ax}: {skipped[seed][ax]}" for ax in DISCRETE_AXES)
        print(f"[random_floor] seed {seed}: query escluse da Recall/mAP "
              f"(singleton) {skip_txt}")

    print(f"\n[random_floor] feature per-asse costruite in {t_axes:.1f}s; "
          f"{n_evaluated} query x {len(args.ranking_seeds)} seed valutate in "
          f"{elapsed:.1f}s (CPU totale {time.process_time():.0f}s)")
    print(f"[random_floor] null '{null_tag}' su {len(args.ranking_seeds)} seed, "
          f"query-seed {args.query_seed}\n")
    all_means = [seed_means(metrics[s], args.k_values) for s in args.ranking_seeds]
    print_floor_table(all_means, args.k_values)

    out_path = Path(args.out) if args.out else Path(DEFAULT_OUT_DIR) / (
        f"{null_tag}_floor_{Path(args.gallery).parent.name}_{args.split}_seed{seed0}.npz"
    )
    recorder.write(out_path, meta={
        "branch": f"null-{null_tag}",
        "run_tag": f"{null_tag}_floor/seed{seed0}",
        "mode": "full",
        "partial_label": None,
        "split": args.split,
        "exclude_self": True,
        "query_seed": args.query_seed,
        "gallery": {"n": len(entries), "sha1": sha1, "source": str(args.gallery)},
    })
    print(f"[random_floor] per-query del seed {seed0} salvato in {out_path} "
          f"({len(recorder)} query)")


if __name__ == "__main__":
    main()
