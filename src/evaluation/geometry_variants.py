# src/evaluation/geometry_variants.py

"""
Quanto dipende la classifica dei sistemi dai PESI dell'asse geometria?

`GalleryAxes.geometry_sim` media tre componenti — area del footprint, aspect
ratio, distribuzione dell'area per tipo di stanza — con pesi (1,1,1) scelti a
priori. Se cambiando quei pesi cambiasse *l'ordine* dei sistemi, il risultato
"chi vince sulla geometria" sarebbe un artefatto della definizione del gain, non
una proprieta' dei modelli. Questa e' la domanda a cui il modulo risponde.

L'analisi e' **post-hoc e gratuita**: i pesi entrano nel *gain* (nel ground
truth), non nel ranking prodotto dal modello. Il ranking e' gia' salvato nei
`ret_rows` del file per-query, quindi basta ricalcolare nDCG con un gain
diverso: niente retrieval, niente GPU, niente FAISS.

Attenzione: il coefficiente 0.5 dentro `dist_sim` (che porta la distanza L1 fra
distribuzioni in [0,1]) NON e' un peso ed e' fuori dall'analisi.

**Criterio di decisione, fissato prima di guardare i numeri**: il claim "il
sistema A batte B sulla geometria" regge se, in TUTTE le pesature esplorate, il
segno del delta appaiato e' lo stesso E il CI 95% del delta esclude lo zero.
Altrimenti il verdetto e' NON REGGE (inversione significativa) o INCONCLUSIVO
(almeno un CI che contiene lo zero). Per questo la tabella riporta il delta
appaiato con intervallo, non solo le stime puntuali: due medie ordinate non sono
una differenza dimostrata.

**Due avvertenze di lettura, non opzionali**:

(i) Le righe della tabella NON sono confrontabili fra loro: ogni pesatura e' una
    ground truth diversa, quindi cambia la scala e la difficolta' del compito.
    Si legge solo DENTRO una riga (chi vince a parita' di GT), mai in colonna.

(ii) `dist_sim` e' funzione deterministica di `rType` + `boxes`, cioe' esattamente
    delle node feature del grafo: la pesatura `(0,0,1)` e' quindi la piu'
    **circolare** per il ramo graph (misura il modello con il suo stesso input) e
    `(1,1,0)` la meno. L'ordine dei sistemi va letto pesatura per pesatura e mai
    mediato fra pesature: una media impasterebbe l'asse piu' circolare con quello
    piu' indipendente.

Uso tipico:

    python -m src.evaluation.geometry_variants \
        --gallery embeddings/<...>/image_paths.json \
        --perquery run_A.npz run_B.npz --k 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.data.rplan_metadata import load_metadata
from src.evaluation.metrics import ndcg_at_k
from src.evaluation.perquery import load_perquery
from src.evaluation.relevance import GalleryAxes
from src.evaluation.significance import BOOTSTRAP_B, BOOTSTRAP_SEED, bootstrap_ci

# Pesature esplorate di default: la baseline, le tre componenti isolate, area +
# aspect (la "forma" pura, senza il contenuto) e il raddoppio della
# distribuzione (l'unica componente che guarda DENTRO la pianta).
DEFAULT_WEIGHTS = (
    (1.0, 1.0, 1.0),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 1.0, 0.0),
    (1.0, 1.0, 2.0),
)
BASELINE_WEIGHTS = (1.0, 1.0, 1.0)
DEFAULT_K = 10


# ----------------------------------------------------------------------
# Il gain geometrico pesato.
# ----------------------------------------------------------------------

def geometry_sim_weighted(axes: GalleryAxes, qi: int, w) -> np.ndarray:
    """`geometry_sim` con pesi arbitrari sulle tre componenti.

    Replica riga per riga `GalleryAxes.geometry_sim` (relevance.py:137-146) e
    normalizza per la somma dei pesi, cosi' il risultato resta in [0,1] e i
    valori di pesature diverse sono confrontabili. Con w=(1,1,1) e' **bitwise
    identica** al metodo originale (le moltiplicazioni per 1.0 sono esatte e
    l'ordine delle somme e' lo stesso): la sostituzione non introduce drift.

    Args:
        axes: gallery su cui calcolare le similarita'.
        qi:   riga della query.
        w:    (w_area, w_aspect, w_dist), non tutti nulli.

    Returns:
        Vettore float32 [N] di similarita' verso l'intera gallery (self incluso;
        le piante senza .mat restano a 0, come nell'originale).
    """
    w_area, w_aspect, w_dist = (float(x) for x in w)
    total = w_area + w_aspect + w_dist
    if total <= 0:
        raise ValueError(f"pesi geometria non validi (somma {total}): {w}")

    area_sim = 1.0 - np.abs(axes.area - axes.area[qi])
    aq = axes.aspect[qi]
    aspect_sim = np.minimum(axes.aspect, aq) / np.maximum(axes.aspect, aq)
    dist_sim = 1.0 - 0.5 * np.abs(axes.dist - axes.dist[qi]).sum(axis=1)

    sim = (w_area * area_sim + w_aspect * aspect_sim + w_dist * dist_sim) / total
    sim[~axes.valid] = 0.0
    return sim.astype(np.float32)


# ----------------------------------------------------------------------
# Ricalcolo di nDCG geometria da un file per-query.
# ----------------------------------------------------------------------

def perquery_ndcg_geometry(axes: GalleryAxes, data, w, k: int):
    """nDCG@k geometria PER QUERY, ricalcolato con la pesatura `w`.

    Usa i `ret_rows` salvati (il ranking del modello, che i pesi NON cambiano) e
    ricostruisce gain e IDCG con `ndcg_at_k`, la stessa funzione della
    valutazione vera. `exclude_self` e' letto dal meta, cosi' full e partial
    mantengono la loro contabilita'.

    Returns:
        (names, values) — nomi delle query e array float64 [Q] allineato: serve
        per query per il confronto appaiato fra due sistemi.
    """
    if data.ret_rows is None:
        raise ValueError("il file per-query non contiene ret_rows: ricalcolo impossibile")

    exclude_self = bool(data.meta.get("exclude_self", True))
    values = []
    for q, qi in enumerate(np.asarray(data.qi, dtype=int)):
        sims = geometry_sim_weighted(axes, int(qi), w)
        not_self = np.ones(len(axes), dtype=bool)
        if exclude_self:
            not_self[qi] = False
        rows = np.asarray(data.ret_rows[q][:int(data.n_ret[q])], dtype=int)
        retrieved_gains = sims[rows] if len(rows) else np.empty(0)
        values.append(ndcg_at_k(retrieved_gains, sims[not_self], k))
    return [str(n) for n in data.names], np.asarray(values, dtype=float)


def ndcg_geometry_weighted(axes: GalleryAxes, data, w, k: int) -> float:
    """Media sulle query di `perquery_ndcg_geometry` (stima puntuale)."""
    _, values = perquery_ndcg_geometry(axes, data, w, k)
    return float(values.mean()) if len(values) else float("nan")


def paired_delta(names_a, values_a, names_b, values_b) -> dict:
    """Delta appaiato per nome fra due sistemi + CI 95% bootstrap.

    L'appaiamento e' sui NOMI (mai sulla posizione: i due file possono avere
    query in ordine o quantita' diversi) e il CI viene da `significance.
    bootstrap_ci`, non da una seconda implementazione del bootstrap.
    """
    col_b = {name: i for i, name in enumerate(names_b)}
    pairs = [(values_a[i], values_b[col_b[name]])
             for i, name in enumerate(names_a) if name in col_b]
    if not pairs:
        return {"n_pairs": 0, "mean_a": float("nan"), "mean_b": float("nan"),
                "delta": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}

    a = np.asarray([p[0] for p in pairs], dtype=float)
    b = np.asarray([p[1] for p in pairs], dtype=float)
    lo, hi = bootstrap_ci(a - b)
    return {"n_pairs": len(pairs), "mean_a": float(a.mean()), "mean_b": float(b.mean()),
            "delta": float((a - b).mean()), "ci_lo": lo, "ci_hi": hi}


def stored_geometry_ndcg(data, k: int) -> float:
    """nDCG@k geometria come salvato nel file (controllo di coerenza del ricalcolo)."""
    row = data.ndcg[data.axis_index("geometry"), data.k_index(k)]
    return float(np.nanmean(row))


def check_gallery(axes: GalleryAxes, data, label: str) -> None:
    """Errore duro se il file per-query non e' della gallery caricata.

    I `ret_rows` sono indici di riga: su una gallery diversa indicherebbero altre
    piante, in silenzio.
    """
    n_meta = data.meta.get("gallery", {}).get("n")
    if n_meta is not None and int(n_meta) != len(axes):
        raise ValueError(
            f"{label}: gallery del file ({n_meta} righe) diversa da quella caricata "
            f"({len(axes)}): i ret_rows non sono interpretabili"
        )


# ----------------------------------------------------------------------
# Tabella pesatura x sistema + criterio di decisione sul claim.
# ----------------------------------------------------------------------

def print_table(results: dict, systems: list[str], k: int) -> None:
    """Tabella pesatura x sistema di nDCG@k geometria."""
    print(f"=== nDCG@{k} geometria, ricalcolato per pesatura (area, aspect, dist) ===")
    header = f"{'pesatura':<16}" + "".join(f"{s:>16}" for s in systems)
    print(header)
    print("-" * len(header))
    for w, scores in results.items():
        label = "(" + ",".join(f"{x:g}" for x in w) + ")"
        print(f"{label:<16}" + "".join(f"{scores[s]:>16.4f}" for s in systems))
    print()


def claim_verdict(rows: list[dict]) -> str:
    """Verdetto sul claim "A batte B sulla geometria", criterio fissato a priori.

    REGGE          se in TUTTE le pesature il segno del delta e' lo stesso e il
                   CI 95% esclude lo zero;
    NON REGGE      se almeno una pesatura ha un delta di segno OPPOSTO con CI che
                   esclude lo zero (inversione dimostrata, non rumore);
    INCONCLUSIVO   negli altri casi (almeno un CI contiene lo zero).
    """
    excludes_zero = [r["ci_lo"] > 0 or r["ci_hi"] < 0 for r in rows]
    signs = [np.sign(r["delta"]) for r in rows]
    reference = signs[0]

    if any(e and s != reference for e, s in zip(excludes_zero, signs)):
        return "NON REGGE"
    if all(excludes_zero) and len(set(signs)) == 1 and reference != 0:
        return "REGGE"
    return "INCONCLUSIVO"


def print_paired_table(rows: list[dict], system_a: str, system_b: str, k: int) -> None:
    """Delta appaiato + CI 95% per ogni pesatura, e verdetto sul claim."""
    print(f"=== delta appaiato nDCG@{k} geometria: {system_a} - {system_b} ===")
    print(f"criterio (fissato prima dei numeri): il claim regge se in TUTTE le "
          f"{len(rows)} pesature\nil segno del delta e' invariato E il CI 95% esclude lo zero.")
    header = (f"{'pesatura':<16} {'n':>6} {system_a[:10]:>10} {system_b[:10]:>10} "
              f"{'delta':>9} {'CI 95%':>21}")
    print(header)
    print("-" * len(header))
    for r in rows:
        label = "(" + ",".join(f"{x:g}" for x in r["weights"]) + ")"
        flag = "" if (r["ci_lo"] > 0 or r["ci_hi"] < 0) else "   <- CI contiene 0"
        print(f"{label:<16} {r['n_pairs']:>6} {r['mean_a']:>10.4f} {r['mean_b']:>10.4f} "
              f"{r['delta']:>+9.4f} [{r['ci_lo']:>+9.4f},{r['ci_hi']:>+9.4f}]{flag}")
    print(f"\nVERDETTO: {claim_verdict(rows)}  "
          f"(bootstrap appaiato B={BOOTSTRAP_B}, seed {BOOTSTRAP_SEED})\n")


# ----------------------------------------------------------------------
# Entrypoint.
# ----------------------------------------------------------------------

def parse_weights(values: list[str] | None):
    """Pesature da CLI, ciascuna nella forma 'wa,wb,wc'."""
    if not values:
        return list(DEFAULT_WEIGHTS)
    parsed = []
    for item in values:
        parts = item.split(",")
        if len(parts) != 3:
            raise ValueError(f"pesatura '{item}': attesi 3 valori separati da virgola")
        parsed.append(tuple(float(x) for x in parts))
    return parsed


def load_gallery_axes(gallery_path: str) -> GalleryAxes:
    """Feature per-asse dai `.mat`, allineate alle righe del JSON di gallery."""
    entries = json.loads(Path(gallery_path).read_text())
    print(f"[geometry_variants] gallery: {len(entries)} righe ({gallery_path})")
    axes = GalleryAxes([load_metadata(e) for e in entries])
    print(f"[geometry_variants] metadati .mat presenti: {int(axes.valid.sum())}/{len(entries)}")
    return axes


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sensibilita' dell'asse geometria ai pesi delle sue componenti."
    )
    p.add_argument("--gallery", required=True,
                   help="image_paths.json (vision) o names.json (graph) della gallery")
    p.add_argument("--perquery", nargs="+", required=True,
                   help="uno o piu' .npz per-query (un sistema ciascuno)")
    p.add_argument("--label", nargs="+", default=None,
                   help="etichette dei sistemi (default: run_tag del meta)")
    p.add_argument("--weights", nargs="+", default=None,
                   help="pesature 'wa,wb,wc' (default: 6 pesature esplorative)")
    p.add_argument("--k", type=int, default=DEFAULT_K)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.label and len(args.label) != len(args.perquery):
        raise ValueError("--label deve avere tanti valori quanti --perquery")

    weights = parse_weights(args.weights)
    axes = load_gallery_axes(args.gallery)

    systems, data_by_system = [], {}
    for i, path in enumerate(args.perquery):
        data = load_perquery(path)
        check_gallery(axes, data, Path(path).name)
        label = args.label[i] if args.label else str(data.meta.get("run_tag", Path(path).stem))
        systems.append(label)
        data_by_system[label] = data

    # Controllo di coerenza: con i pesi originali il ricalcolo deve riprodurre il
    # numero salvato. Se non lo fa, i ret_rows o la gallery non corrispondono.
    for label, data in data_by_system.items():
        if args.k in list(data.meta.get("k_values", [])):
            recomputed = ndcg_geometry_weighted(axes, data, BASELINE_WEIGHTS, args.k)
            stored = stored_geometry_ndcg(data, args.k)
            print(f"[geometry_variants] {label}: ricalcolo (1,1,1) {recomputed:.6f} "
                  f"vs salvato {stored:.6f} (delta {recomputed - stored:+.2e})")
    print()

    # Un ricalcolo per (pesatura, sistema), tenuto PER QUERY: le medie servono
    # alla tabella, i valori per query al confronto appaiato.
    per_query = {}
    results = {}
    for w in weights:
        key = tuple(w)
        results[key] = {}
        for label, data in data_by_system.items():
            names, values = perquery_ndcg_geometry(axes, data, w, args.k)
            per_query[(key, label)] = (names, values)
            results[key][label] = float(values.mean()) if len(values) else float("nan")

    print_table(results, systems, args.k)

    # Il criterio di decisione confronta DUE sistemi: con piu' di due la tabella
    # resta descrittiva (e l'ordine puntuale non e' un risultato).
    if len(systems) == 2:
        a_label, b_label = systems
        rows = []
        for w in weights:
            key = tuple(w)
            names_a, values_a = per_query[(key, a_label)]
            names_b, values_b = per_query[(key, b_label)]
            rows.append({"weights": key,
                         **paired_delta(names_a, values_a, names_b, values_b)})
        print_paired_table(rows, a_label, b_label, args.k)
    else:
        print(f"[geometry_variants] {len(systems)} sistemi: il criterio di decisione "
              "(delta appaiato + CI) vale su una COPPIA.\n"
              "Rilancia con due soli --perquery per avere il verdetto; la tabella "
              "sopra e' solo descrittiva.\n")

    print("I pesi cambiano il GAIN (il ground truth), non il ranking dei modelli: "
          "l'analisi e' post-hoc sui ret_rows salvati.")
    print("Righe NON confrontabili fra loro (ogni pesatura e' una GT diversa); "
          "(0,0,1) e' la pesatura piu' circolare per il ramo graph, (1,1,0) la meno.")


if __name__ == "__main__":
    main()
