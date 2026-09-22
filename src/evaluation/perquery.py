# src/evaluation/perquery.py

"""
Persistenza dei valori PER-QUERY delle metriche di retrieval (core condiviso).

Perché: oggi entrambi i rami stampano solo le MEDIE sulle query. Confronti
appaiati fra run, test di significatività e analisi per-asse hanno bisogno del
valore della SINGOLA query, e riottenerlo costa una valutazione intera: qui
viene salvato una volta sola, durante la valutazione che si sta già facendo.

Vincolo di progetto: questo modulo NON cambia l'accumulo delle medie. I valori
per-query vengono LETTI dai contenitori di accumulo dei due rami
(`metrics[asse][metrica][k]`: liste in cui ogni query appende un valore) subito
dopo la chiamata di accumulo — la coda della lista È il contributo di quella
query. Le medie stampate restano quindi bit-identiche a prima.

Formato del file `.npz` (contratto stabile, versionato da SCHEMA_VERSION):

    meta         stringa JSON in array 0-d (chiavi sotto)
    names        <U16 [Q]        stem della query: è LA chiave di join fra run
                                 (mai `qi`: le gallery dei due rami differiscono)
    qi           int32 [Q]       riga della query nella PROPRIA gallery
    ndcg         f32 [A,K,Q]     NaN = query non applicabile su quell'asse
    recall, map  f32 [A,K,Q]     NaN su geometry e sulle query singleton
    num_relevant int32 [A,Q]     -1 sugli assi continui, 0 = query saltata
    ret_rows     int32 [Q,max_k] righe recuperate in ordine di rank, -1 = padding
    n_ret        int16 [Q]       quante righe valide ci sono in ret_rows
    self_rr      f32 [Q]         reciprocal rank del self (solo mode="partial")
    area_removed f32 [Q]         frazione dei pixel-pianta rimossa dal danno
                                 (opzionale, solo partial vision dall'11 set 2026)

Chiavi del `meta`: schema_version, branch, run_tag, mode ("full"|"partial"),
partial_label, split, exclude_self, num_queries, query_seed, k_values, axes,
geometry_weights, gallery{n, sha1, source}, max_k, timestamp, argv.
Chiave opzionale (additiva, 11 set 2026): damage{strategy, params, patch_size,
image_size, area_space: "native"|"resized", n_fallback} nel partial vision.
I file senza `area_removed`/`damage` restano validi: stesso SCHEMA_VERSION.

Regola NaN: una query saltata su un asse resta nell'array (allineata a `names`)
con NaN e `num_relevant = 0`. Di conseguenza `np.nanmean` sulle matrici salvate
riproduce esattamente la media stampata dai due rami, che ignora le query
saltate invece di contarle come zero.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.evaluation.relevance import AXES, DISCRETE_AXES

SCHEMA_VERSION = "perquery/1"

# Pesi delle tre componenti dell'asse geometry: sono fissi in
# `relevance.geometry_sim` (media semplice di area/aspect/distribuzione), non
# configurabili. Vanno nel meta perché due run con pesi diversi non sarebbero
# confrontabili in modo appaiato.
GEOMETRY_WEIGHTS = {"area": 1 / 3, "aspect": 1 / 3, "type_area_distribution": 1 / 3}

NAME_DTYPE = "<U16"     # gli stem RPLAN sono numerici e corti
PAD_ROW = -1            # padding di ret_rows
NO_RELEVANT_SET = -1    # num_relevant sugli assi continui (nessuna classe di equivalenza)


# ----------------------------------------------------------------------
# Identità della gallery.
# ----------------------------------------------------------------------

def gallery_sha1(names) -> str:
    """SHA-1 dei nomi della gallery, nell'ORDINE DELLE RIGHE (non ordinati
    alfabeticamente): è quell'ordine a dare senso a `qi`, quindi due run con lo
    stesso hash sono confrontabili riga per riga.

    Implementazione (deterministica, da non cambiare senza alzare
    SCHEMA_VERSION): sha1 di "\\n".join(names) in UTF-8.
    """
    joined = "\n".join(str(n) for n in names)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------
# Lettura.
# ----------------------------------------------------------------------

@dataclass
class PerQueryData:
    """Contenuto di un file per-query, con gli indici per navigarlo.

    Le matrici sono [A, K, Q] (asse, profondità K, query): `axis_index` e
    `k_index` traducono nomi d'asse e valori di K nei rispettivi indici usando
    il `meta`, così il chiamante non deve ricordarne l'ordine.
    """

    meta: dict
    names: np.ndarray
    qi: np.ndarray
    ndcg: np.ndarray
    recall: np.ndarray
    map: np.ndarray
    num_relevant: np.ndarray
    ret_rows: np.ndarray | None = None
    n_ret: np.ndarray | None = None
    self_rr: np.ndarray | None = None
    area_removed: np.ndarray | None = None

    def axis_index(self, name: str) -> int:
        """Indice della riga di `name` nelle matrici [A,K,Q] (da meta["axes"])."""
        axes = [str(a) for a in self.meta["axes"]]
        if name not in axes:
            raise KeyError(f"asse '{name}' non presente nel file (assi: {axes})")
        return axes.index(name)

    def k_index(self, k: int) -> int:
        """Indice della colonna K nelle matrici [A,K,Q] (da meta["k_values"])."""
        k_values = [int(v) for v in self.meta["k_values"]]
        if int(k) not in k_values:
            raise KeyError(f"K={k} non presente nel file (K disponibili: {k_values})")
        return k_values.index(int(k))


def load_perquery(path) -> PerQueryData:
    """Rilegge un file scritto da `write_npz`.

    Args:
        path: percorso del .npz.

    Returns:
        `PerQueryData`; `ret_rows`/`n_ret`/`self_rr`/`area_removed` sono None
        se assenti dal file.
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"].item()))
        if meta.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"{path}: schema '{meta.get('schema_version')}' incompatibile "
                f"(atteso '{SCHEMA_VERSION}')"
            )
        optional = {
            key: (z[key] if key in z.files else None)
            for key in ("ret_rows", "n_ret", "self_rr", "area_removed")
        }
        return PerQueryData(
            meta=meta,
            names=z["names"],
            qi=z["qi"],
            ndcg=z["ndcg"],
            recall=z["recall"],
            map=z["map"],
            num_relevant=z["num_relevant"],
            **optional,
        )


# ----------------------------------------------------------------------
# Scrittura.
# ----------------------------------------------------------------------

def write_npz(path, *, meta: dict, names, qi, ndcg, recall, map_, num_relevant,
              ret_rows=None, n_ret=None, self_rr=None, area_removed=None) -> None:
    """Scrive il file per-query, validando forme e dtype del contratto.

    Args:
        path:         file .npz di destinazione (le cartelle mancanti vengono create).
        meta:         chiavi del contratto; deve contenere almeno `axes` e
                      `k_values` (servono a validare le forme e a `axis_index`/
                      `k_index`). `schema_version` viene sempre (ri)scritto qui;
                      `timestamp` e `argv` sono aggiunti se mancanti.
        names:        [Q] stem delle query.
        qi:           [Q] riga della query nella gallery.
        ndcg, recall, map_: [A,K,Q] float, NaN dove la query è saltata.
        num_relevant: [A,Q] int.
        ret_rows:     [Q,max_k] int, -1 = padding (opzionale).
        n_ret:        [Q] int (opzionale).
        self_rr:      [Q] float, solo per mode="partial" (opzionale).
        area_removed: [Q] float, frazione di pianta rimossa (opzionale).

    Side effects: crea/sovrascrive `path`.
    """
    path = Path(path)
    meta = dict(meta)
    meta["schema_version"] = SCHEMA_VERSION
    meta.setdefault("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S"))
    meta.setdefault("argv", list(sys.argv))

    for key in ("axes", "k_values"):
        if key not in meta:
            raise ValueError(f"meta['{key}'] mancante: serve a interpretare le matrici")
    n_axes, n_k = len(meta["axes"]), len(meta["k_values"])

    # Il controllo va fatto PRIMA del cast: numpy troncherebbe in silenzio, e un
    # nome troncato romperebbe il join per nome fra i due rami.
    max_len = int(NAME_DTYPE[2:])
    too_long = [n for n in names if len(str(n)) > max_len]
    if too_long:
        raise ValueError(f"nomi troppo lunghi per {NAME_DTYPE} (es. {too_long[0]!r})")
    names = np.asarray(names, dtype=NAME_DTYPE)
    n_query = len(names)

    arrays = {
        "names": names,
        "qi": np.asarray(qi, dtype=np.int32),
        "ndcg": np.asarray(ndcg, dtype=np.float32),
        "recall": np.asarray(recall, dtype=np.float32),
        "map": np.asarray(map_, dtype=np.float32),
        "num_relevant": np.asarray(num_relevant, dtype=np.int32),
    }
    expected = {
        "qi": (n_query,),
        "ndcg": (n_axes, n_k, n_query),
        "recall": (n_axes, n_k, n_query),
        "map": (n_axes, n_k, n_query),
        "num_relevant": (n_axes, n_query),
    }
    if ret_rows is not None:
        arrays["ret_rows"] = np.asarray(ret_rows, dtype=np.int32)
        expected["ret_rows"] = (n_query, arrays["ret_rows"].shape[-1])
    if n_ret is not None:
        arrays["n_ret"] = np.asarray(n_ret, dtype=np.int16)
        expected["n_ret"] = (n_query,)
    if self_rr is not None:
        arrays["self_rr"] = np.asarray(self_rr, dtype=np.float32)
        expected["self_rr"] = (n_query,)
    if area_removed is not None:
        arrays["area_removed"] = np.asarray(area_removed, dtype=np.float32)
        expected["area_removed"] = (n_query,)

    for key, shape in expected.items():
        if arrays[key].shape != shape:
            raise ValueError(
                f"'{key}': forma {arrays[key].shape}, attesa {shape} "
                f"(Q={n_query}, A={n_axes}, K={n_k})"
            )

    meta.setdefault("num_queries", n_query)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, meta=np.array(json.dumps(meta, sort_keys=True)), **arrays)


# ----------------------------------------------------------------------
# Raccolta durante la valutazione.
# ----------------------------------------------------------------------

def count_relevant(axes, axis: str, qi: int, exclude_self: bool) -> int:
    """Quanti rilevanti ha la query `qi` sull'asse discreto `axis`.

    Stessa definizione usata dall'accumulo (classe di equivalenza esatta, self
    escluso se `exclude_self`). È un ricalcolo deliberato: l'accumulo del ramo
    graph è condiviso con la sonda che seleziona i checkpoint e non si tocca,
    quindi il conteggio non è recuperabile da lì. Costa una scansione della
    gallery per asse discreto, e solo quando il salvataggio per-query è attivo.
    """
    rel = axes.relevant(axis, qi)
    if exclude_self:
        rel[qi] = False
    return int(rel.sum())


class PerQueryRecorder:
    """Buffer dei valori per-query, riempito ACCANTO all'accumulo delle medie.

    Non calcola nulla di nuovo sulle metriche: rilegge l'ultimo valore appeso
    dall'accumulo per la query corrente. Uso, dentro il loop sulle query:

        skipped_before = dict(skipped)
        accumulate_axes(metrics, skipped, axes, qi, ret_rows, k_values, exclude_self)
        recorder.add(name=..., qi=qi, ret_rows=ret_rows, metrics=metrics,
                     skipped_before=skipped_before, skipped=skipped,
                     axes=axes, exclude_self=exclude_self)

    Args:
        k_values: le stesse profondità passate all'accumulo (l'ordine diventa
                  l'asse K delle matrici salvate).
        max_k:    larghezza di `ret_rows` (padding a -1 sotto questa soglia).
        with_self_rr: True nel partial, dove ogni query ha il reciprocal rank
                  del self da salvare.
        with_area_removed: True nel partial vision, dove ogni query ha la
                  frazione di pianta rimossa dal danno.
    """

    def __init__(self, k_values, max_k: int, with_self_rr: bool = False,
                 with_area_removed: bool = False):
        self.k_values = tuple(k_values)
        self.max_k = int(max_k)
        self.with_self_rr = with_self_rr
        self.with_area_removed = with_area_removed
        self._names: list[str] = []
        self._qi: list[int] = []
        self._ndcg: list[np.ndarray] = []      # ognuno [A,K]
        self._recall: list[np.ndarray] = []
        self._map: list[np.ndarray] = []
        self._num_relevant: list[np.ndarray] = []   # ognuno [A]
        self._ret_rows: list[np.ndarray] = []       # ognuno [max_k]
        self._n_ret: list[int] = []
        self._self_rr: list[float] = []
        self._area_removed: list[float] = []

    def add(self, *, name, qi, ret_rows, metrics, skipped_before, skipped,
            axes, exclude_self: bool, self_rr: float | None = None,
            area_removed: float | None = None) -> None:
        """Registra la query appena accumulata.

        Args:
            name:           stem della pianta-query (chiave di join fra run).
            qi:             riga della query nella gallery.
            ret_rows:       righe recuperate, in ordine di rank.
            metrics:        contenitore dell'accumulo (letto, mai modificato).
            skipped_before: copia di `skipped` PRIMA dell'accumulo: il delta dice
                            su quali assi discreti questa query è stata saltata.
            skipped:        `skipped` dopo l'accumulo.
            axes:           `GalleryAxes` (serve solo a contare i rilevanti).
            exclude_self:   come nell'accumulo.
            self_rr:        reciprocal rank del self, richiesto se with_self_rr.
            area_removed:   frazione di pianta rimossa, richiesta se with_area_removed.
        """
        if self.with_self_rr and self_rr is None:
            raise ValueError("self_rr richiesto: il recorder è in modalità partial")
        if self.with_area_removed and area_removed is None:
            raise ValueError("area_removed richiesto: il recorder registra l'area rimossa")

        n_axes, n_k = len(AXES), len(self.k_values)
        ndcg = np.full((n_axes, n_k), np.nan, dtype=np.float32)
        recall = np.full((n_axes, n_k), np.nan, dtype=np.float32)
        map_ = np.full((n_axes, n_k), np.nan, dtype=np.float32)
        num_relevant = np.full(n_axes, NO_RELEVANT_SET, dtype=np.int32)

        for a, ax in enumerate(AXES):
            for j, k in enumerate(self.k_values):
                ndcg[a, j] = metrics[ax]["ndcg"][k][-1]
            if ax not in DISCRETE_AXES:
                continue
            if skipped[ax] > skipped_before[ax]:
                # Classe di equivalenza singleton: Recall/mAP non definiti ->
                # restano NaN e la query non entrerà in nessuna media.
                num_relevant[a] = 0
                continue
            for j, k in enumerate(self.k_values):
                recall[a, j] = metrics[ax]["recall"][k][-1]
                map_[a, j] = metrics[ax]["map"][k][-1]
            num_relevant[a] = count_relevant(axes, ax, qi, exclude_self)

        rows = np.asarray(ret_rows, dtype=np.int32).ravel()
        if len(rows) > self.max_k:
            raise ValueError(f"ret_rows più lunghi di max_k={self.max_k}: {len(rows)}")
        padded = np.full(self.max_k, PAD_ROW, dtype=np.int32)
        padded[: len(rows)] = rows

        self._names.append(str(name))
        self._qi.append(int(qi))
        self._ndcg.append(ndcg)
        self._recall.append(recall)
        self._map.append(map_)
        self._num_relevant.append(num_relevant)
        self._ret_rows.append(padded)
        self._n_ret.append(len(rows))
        if self.with_self_rr:
            self._self_rr.append(float(self_rr))
        if self.with_area_removed:
            self._area_removed.append(float(area_removed))

    def __len__(self) -> int:
        return len(self._names)

    def write(self, path, meta: dict) -> None:
        """Impila i buffer e scrive il file.

        Le chiavi strutturali del meta (`axes`, `k_values`, `max_k`,
        `geometry_weights`) le mette il recorder, che è l'unico a conoscerle con
        certezza; il chiamante fornisce quelle di run (branch, run_tag, mode,
        partial_label, split, exclude_self, query_seed, gallery).

        Side effects: crea/sovrascrive `path`.
        """
        shape = (len(AXES), len(self.k_values))
        write_npz(
            path,
            meta={
                **meta,
                "axes": list(AXES),
                "k_values": list(self.k_values),
                "max_k": self.max_k,
                "geometry_weights": GEOMETRY_WEIGHTS,
            },
            names=self._names,
            qi=np.asarray(self._qi, dtype=np.int32),
            ndcg=_stack(self._ndcg, shape),
            recall=_stack(self._recall, shape),
            map_=_stack(self._map, shape),
            num_relevant=_stack(self._num_relevant, (len(AXES),)),
            ret_rows=(np.stack(self._ret_rows) if self._ret_rows
                      else np.empty((0, self.max_k), dtype=np.int32)),
            n_ret=np.asarray(self._n_ret, dtype=np.int16),
            self_rr=(np.asarray(self._self_rr, dtype=np.float32)
                     if self.with_self_rr else None),
            area_removed=(np.asarray(self._area_removed, dtype=np.float32)
                          if self.with_area_removed else None),
        )


def _stack(rows: list[np.ndarray], shape: tuple[int, ...]) -> np.ndarray:
    """Impila i contributi per-query sull'ULTIMO asse -> [..., Q].
    `shape` è la forma di un singolo contributo, serve al caso Q=0."""
    if not rows:
        return np.empty(shape + (0,), dtype=np.float32)
    return np.stack(rows, axis=-1)
