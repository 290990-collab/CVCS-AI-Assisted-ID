# src/evaluation/robustness_auc.py

"""
AUC di robustezza del criterio A.5 (core condiviso, entrambi i rami).

Che cosa misura
---------------
Dal 24 ago 2026 «migliore» = **piu' robusto** (`status.md § 23`). La misura e'
l'AUC della curva self-recovery MRR sotto masking `random`, sul valid:

    AUC(q) = media di self_rr(q, f) su f in {0.25, 0.5, 0.75}   (pesi uguali)
    AUC    = media di AUC(q) sulle query

f = 0.0 e' escluso di proposito: e' il tetto dei DATI (duplicati RPLAN, MRR
~0.970 per tutte le config), non dei modelli (§ 24.2b).

Perche' un modulo
-----------------
Il verdetto di § 24 era prodotto da uno script di scratchpad: qui la ricetta
diventa riproducibile (debito trasversale n.1 della roadmap) e serve a C.0, B.6
e B.7. L'AUC e' calcolata **per query**, cosi' il confronto fra due config resta
**appaiato** (bootstrap sulle coppie, stesso `bootstrap_ci` di significance.py).

Una "config" e' il prefisso comune dei suoi file per-query:

    <prefix>_partial-random-f<frazione>_<split>.npz

Uso (CPU, secondi):

    # classifica di tutte le config trovate in una o piu' cartelle
    python -m src.evaluation.robustness_auc rank \\
        --dir results/perquery/vision_partial_valid_B --top 10

    # delta appaiato fra due config (prefissi con la cartella)
    python -m src.evaluation.robustness_auc compare \\
        --a results/perquery/vision_partial_valid_B/vision_dinov3_natural_head \\
        --b results/perquery/graph_partial_valid/graph_gcn_tau02

Danni non a stanze (11 set 2026, solo vision): `--strategy crop|patch` legge i
file `<prefix>_partial-crop-f*` / `_partial-patch-f*` con la stessa ricetta.
`--strategy nowalls-random` legge la curva in cui la stanza tolta perde anche i
propri muri (`vision_damage.py`): stesse stanze del `random`, quindi appaiata.
Default `random` = comportamento di prima.

Metro di robustezza (14 set 2026, `status.md §38`, deciso PRIMA dell'ondata 2):
`--robust` = media per query delle AUC su `nowalls-random`, `crop` e `patch`
(`ROBUST_STRATEGIES`), cioe' i tre danni che tolgono davvero informazione. Il
vecchio `random` resta leggibile ma non entra: lasciava i muri della stanza tolta.

    python -m src.evaluation.robustness_auc rank --dir <cartella> --robust Sul `patch` una differenza di
`damage.patch_size`/`image_size` fra A e B viene segnalata: le griglie diverse
(P14 vs P16, 448 vs 224) rendono il danno non identico fra le due config.
"""

from __future__ import annotations

import argparse
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.evaluation.perquery import load_perquery
from src.evaluation.significance import (
    DEFAULT_MIN_PAIR_FRACTION,
    bootstrap_ci,
    wilcoxon_p,
)

# Frazioni dell'AUC, fissate in § 23 PRIMA di vedere i dati.
AUC_FRACTIONS = (0.25, 0.5, 0.75)
STRATEGY = "random"
AUC_STRATEGIES = ("random", "crop", "patch", "nowalls-random")
# Il metro pre-registrato (§38): l'ordine non conta, la media e' simmetrica.
ROBUST_STRATEGIES = ("nowalls-random", "crop", "patch")

# Campi del meta che devono coincidere fra le frazioni di una config e fra due
# config confrontate. Esclusi: `partial_label` (e' cio' che varia con f) e
# `num_queries`, che e' il numero di query SCRITTE — nel graph le query svuotate
# dal masking vengono saltate, quindi cambia con f e fra i rami. La copertura si
# controlla invece con l'appaiamento per nome (`pair_fraction`).
AUC_STRICT_KEYS = ("split", "exclude_self", "mode", "geometry_weights", "query_seed")

def _file_re(strategy: str = STRATEGY) -> re.Pattern:
    """Regex dei nomi file per-query di una strategia a frazioni."""
    return re.compile(
        r"^(?P<prefix>.+)_partial-" + re.escape(strategy)
        + r"-f(?P<frac>[0-9.]+)_(?P<split>[a-z]+)\.npz$"
    )


_FILE_RE = _file_re(STRATEGY)


@dataclass
class AucData:
    """AUC per query di una config, con i valori per frazione che la compongono."""

    prefix: str
    names: np.ndarray            # [P] query valide in TUTTE le frazioni
    auc: np.ndarray              # [P] AUC per query
    per_fraction: dict           # f -> [P] self_rr allineato a `names`
    meta: dict                   # meta della prima frazione (gallery, split, ...)
    n_ref: int                   # query nel file (prima del filtro NaN)
    strategy: str = STRATEGY     # danno della curva (random | crop | patch | a+b+c)
    components: dict = field(default_factory=dict)   # solo --robust: danno -> AucData

    @property
    def mean(self) -> float:
        return float(self.auc.mean()) if len(self.auc) else float("nan")


def fraction_path(prefix: str | Path, frac: float, split: str,
                  strategy: str = STRATEGY) -> Path:
    """Path del file per-query di `prefix` alla frazione `frac`.

    Il formato della frazione e' quello di Python (`0.25`, `0.5`, `0.75`), lo
    stesso che producono i due rami nel nome della run.
    """
    prefix = Path(prefix)
    return prefix.parent / f"{prefix.name}_partial-{strategy}-f{float(frac)}_{split}.npz"


def _check_same_protocol(meta_a: dict, meta_b: dict, what: str) -> None:
    for key in AUC_STRICT_KEYS:
        va, vb = meta_a.get(key), meta_b.get(key)
        if isinstance(va, list):
            va = tuple(va)
        if isinstance(vb, list):
            vb = tuple(vb)
        if va != vb:
            raise ValueError(f"{what}: meta incompatibili su '{key}': {va!r} vs {vb!r}")
    ga, gb = meta_a.get("gallery", {}), meta_b.get("gallery", {})
    if ga.get("sha1") != gb.get("sha1") or ga.get("n") != gb.get("n"):
        raise ValueError(
            f"{what}: gallery diverse (n={ga.get('n')} sha1={str(ga.get('sha1'))[:12]} vs "
            f"n={gb.get('n')} sha1={str(gb.get('sha1'))[:12]})"
        )


def _warn_damage_mismatch(meta_a: dict, meta_b: dict, what: str) -> None:
    """Avvisa se due run `patch` usano griglie diverse (patch_size/image_size).

    Non e' un errore di protocollo (encoder diversi hanno griglie diverse per
    costruzione), ma il danno non e' lo stesso: il delta va letto con la nota.
    """
    da, db = meta_a.get("damage") or {}, meta_b.get("damage") or {}
    if "patch" not in (da.get("strategy"), db.get("strategy")):
        return
    for key in ("patch_size", "image_size"):
        if da.get(key) != db.get(key):
            warnings.warn(f"{what}: danno patch non appaiato su '{key}': "
                          f"{da.get(key)!r} vs {db.get(key)!r}")


def load_auc(prefix: str | Path, split: str = "valid",
             fractions: tuple[float, ...] = AUC_FRACTIONS,
             strategy: str = STRATEGY) -> AucData:
    """Legge le frazioni di una config e ne calcola l'AUC per query.

    Una query entra solo se ha `self_rr` non-NaN in **tutte** le frazioni (nel
    ramo graph una query svuotata dalla strategia viene saltata): altrimenti la
    sua AUC sarebbe una media su frazioni diverse dalle altre.

    Raises:
        FileNotFoundError: se manca una frazione.
        ValueError: file senza `self_rr`, o frazioni con protocollo/gallery diversi.
    """
    datas = []
    for f in fractions:
        path = fraction_path(prefix, f, split, strategy)
        if not path.exists():
            raise FileNotFoundError(f"frazione f={f} mancante: {path}")
        d = load_perquery(path)
        if d.self_rr is None:
            raise ValueError(f"{path}: nessun `self_rr` (non e' una run partial)")
        datas.append(d)

    ref = datas[0]
    for f, d in zip(fractions[1:], datas[1:]):
        _check_same_protocol(ref.meta, d.meta, f"{Path(prefix).name} f={fractions[0]} vs f={f}")

    maps = [{str(n): float(v) for n, v in zip(d.names, d.self_rr)} for d in datas]
    names = [str(n) for n in ref.names
             if all(str(n) in m and not np.isnan(m[str(n)]) for m in maps)]
    per_fraction = {f: np.asarray([m[n] for n in names], dtype=float)
                    for f, m in zip(fractions, maps)}
    auc = np.mean(np.stack([per_fraction[f] for f in fractions]), axis=0) \
        if names else np.zeros(0)
    return AucData(prefix=str(prefix), names=np.asarray(names), auc=auc,
                   per_fraction=per_fraction, meta=ref.meta, n_ref=len(ref.names),
                   strategy=strategy)


def load_robust_auc(prefix: str | Path, split: str = "valid",
                    strategies: tuple[str, ...] = ROBUST_STRATEGIES) -> AucData:
    """Metro di §38: AUC per query = media delle AUC per query dei danni dati.

    Entrano solo le query presenti in TUTTI i danni (stessa regola di `load_auc`
    fra le frazioni). `per_fraction` resta vuoto: le colonne della classifica
    sono le AUC dei singoli danni, in `components`.

    Raises:
        FileNotFoundError: se a un danno manca una frazione.
        ValueError: danni con protocollo/gallery diversi.
    """
    parts = {s: load_auc(prefix, split, strategy=s) for s in strategies}
    ref = parts[strategies[0]]
    for s in strategies[1:]:
        _check_same_protocol(ref.meta, parts[s].meta,
                             f"{Path(prefix).name} {strategies[0]} vs {s}")
    index = {s: {n: i for i, n in enumerate(d.names)} for s, d in parts.items()}
    names = [n for n in ref.names if all(n in index[s] for s in strategies)]
    auc = (np.mean(np.stack([parts[s].auc[[index[s][n] for n in names]]
                             for s in strategies]), axis=0)
           if names else np.zeros(0))
    return AucData(prefix=str(prefix), names=np.asarray(names), auc=auc,
                   per_fraction={}, meta=ref.meta,
                   n_ref=min(d.n_ref for d in parts.values()),
                   strategy="+".join(strategies), components=parts)


def compare_auc(a: AucData, b: AucData,
                min_pair_fraction: float = DEFAULT_MIN_PAIR_FRACTION) -> dict:
    """Delta appaiato dell'AUC (A − B) con CI bootstrap 95% e Wilcoxon.

    L'appaiamento e' per nome di query. Il protocollo (split, seed, exclude_self,
    gallery) deve coincidere: nessun override, perche' l'AUC e' l'endpoint che
    sceglie le config congelate e un confronto zoppo qui sceglierebbe male.
    """
    _check_same_protocol(a.meta, b.meta, f"{Path(a.prefix).name} vs {Path(b.prefix).name}")
    _warn_damage_mismatch(a.meta, b.meta, f"{Path(a.prefix).name} vs {Path(b.prefix).name}")
    for s in set(a.components) & set(b.components):      # --robust: la patch e' dentro
        _warn_damage_mismatch(a.components[s].meta, b.components[s].meta,
                              f"{Path(a.prefix).name} vs {Path(b.prefix).name} [{s}]")
    col_b = {n: i for i, n in enumerate(b.names)}
    ia, ib = [], []
    for i, n in enumerate(a.names):
        j = col_b.get(n)
        if j is not None:
            ia.append(i)
            ib.append(j)
    xa, xb = a.auc[ia], b.auc[ib]
    diff = xa - xb
    lo, hi = bootstrap_ci(diff)
    p, note = wilcoxon_p(xa, xb)
    n_ref = min(a.n_ref, b.n_ref)
    fraction = len(diff) / n_ref if n_ref else 0.0
    return {
        "a": Path(a.prefix).name, "b": Path(b.prefix).name,
        "n_pairs": len(diff), "n_ref": n_ref, "pair_fraction": fraction,
        "reliable": fraction >= min_pair_fraction,
        "mean_a": float(xa.mean()) if len(xa) else float("nan"),
        "mean_b": float(xb.mean()) if len(xb) else float("nan"),
        "delta": float(diff.mean()) if len(diff) else float("nan"),
        "ci_lo": lo, "ci_hi": hi, "p": p, "p_note": note,
        "tie": bool(lo <= 0.0 <= hi),
    }


def discover_prefixes(directories, split: str = "valid",
                      fractions: tuple[float, ...] = AUC_FRACTIONS,
                      strategy: str = STRATEGY) -> list[Path]:
    """Prefissi che hanno TUTTE le frazioni dell'AUC nelle cartelle date."""
    file_re = _file_re(strategy)
    found: dict[Path, set] = {}
    for directory in directories:
        for path in Path(directory).glob(f"*_partial-{strategy}-f*_{split}.npz"):
            m = file_re.match(path.name)
            if m is None or m["split"] != split:
                continue
            found.setdefault(path.parent / m["prefix"], set()).add(float(m["frac"]))
    need = {float(f) for f in fractions}
    return sorted(p for p, fr in found.items() if need <= fr)


def discover_robust_prefixes(directories, split: str = "valid",
                             strategies: tuple[str, ...] = ROBUST_STRATEGIES) -> list[Path]:
    """Prefissi completi in TUTTI i danni del metro (intersezione)."""
    sets = [set(discover_prefixes(directories, split, strategy=s)) for s in strategies]
    return sorted(set.intersection(*sets))


# ----------------------------------------------------------------------
# CLI.
# ----------------------------------------------------------------------

def _print_compare(row: dict) -> None:
    flag = "" if row["reliable"] else f"  ⚠️ appaiate {row['pair_fraction']:.0%}"
    tie = "  (CI contiene lo zero → spareggio § 23)" if row["tie"] else ""
    print(f"  {row['a']} − {row['b']}: {row['delta']:+.4f} "
          f"[{row['ci_lo']:+.4f}, {row['ci_hi']:+.4f}]  "
          f"AUC {row['mean_a']:.4f} vs {row['mean_b']:.4f}  n={row['n_pairs']}"
          f"  p={row['p']:.2g}{flag}{tie}")


def _cmd_rank(args) -> None:
    if args.robust:
        prefixes = discover_robust_prefixes(args.dir, args.split)
        if not prefixes:
            raise SystemExit(f"nessuna config completa in {ROBUST_STRATEGIES} in {args.dir}")
        datas = [load_robust_auc(p, args.split) for p in prefixes]
    else:
        prefixes = discover_prefixes(args.dir, args.split, strategy=args.strategy)
        if not prefixes:
            raise SystemExit(f"nessuna config con tutte le frazioni {AUC_FRACTIONS} in {args.dir}")
        datas = [load_auc(p, args.split, strategy=args.strategy) for p in prefixes]
    datas.sort(key=lambda d: d.mean, reverse=True)
    ref = datas[0]
    # Una classifica ha senso solo sullo stesso protocollo: si verifica subito.
    for d in datas[1:]:
        _check_same_protocol(ref.meta, d.meta, f"{Path(ref.prefix).name} vs {Path(d.prefix).name}")
    g = ref.meta.get("gallery", {})
    print(f"[robustness_auc] {len(datas)} config · split={args.split} · "
          f"gallery n={g.get('n')} sha1={str(g.get('sha1'))[:12]} · "
          f"exclude_self={ref.meta.get('exclude_self')} · AUC su f={AUC_FRACTIONS}"
          + (f" · METRO §38 = media di {'+'.join(ROBUST_STRATEGIES)}" if args.robust
             else "" if args.strategy == STRATEGY else f" · danno={args.strategy}"))
    cols = ROBUST_STRATEGIES if args.robust else [f"f={f}" for f in AUC_FRACTIONS]
    head = " | ".join(cols)
    print(f"{'#':>3}  {'config':<52} {'AUC':>7} | {head} | n")
    for i, d in enumerate(datas[: args.top] if args.top else datas, 1):
        if args.robust:
            cells = " | ".join(f"{d.components[s].mean:>{len(s)}.4f}" for s in ROBUST_STRATEGIES)
        else:
            cells = " | ".join(f"{d.per_fraction[f].mean():.4f}" for f in AUC_FRACTIONS)
        print(f"{i:>3}  {Path(d.prefix).name:<52} {d.mean:>7.4f} | {cells} | {len(d.auc)}")
    if args.compare_top > 1:
        print(f"\nDelta appaiati del #1 contro i successivi (bootstrap B=10000, seed 0):")
        for d in datas[1: args.compare_top]:
            _print_compare(compare_auc(ref, d))


def _cmd_compare(args) -> None:
    if args.robust:
        a, b = load_robust_auc(args.a, args.split), load_robust_auc(args.b, args.split)
    else:
        a = load_auc(args.a, args.split, strategy=args.strategy)
        b = load_auc(args.b, args.split, strategy=args.strategy)
    _print_compare(compare_auc(a, b, args.min_pair_fraction))


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AUC di robustezza (criterio A.5, § 23)")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rank", help="classifica delle config trovate nelle cartelle")
    r.add_argument("--dir", nargs="+", required=True)
    r.add_argument("--top", type=int, default=0, help="righe da stampare (0 = tutte)")
    r.add_argument("--compare-top", type=int, default=2, dest="compare_top",
                   help="delta appaiati del #1 contro i successivi fino a questo rango")
    c = sub.add_parser("compare", help="delta appaiato A − B")
    c.add_argument("--a", required=True, help="prefisso della config A (con cartella)")
    c.add_argument("--b", required=True, help="prefisso della config B (con cartella)")
    c.add_argument("--min-pair-fraction", type=float, default=DEFAULT_MIN_PAIR_FRACTION,
                   dest="min_pair_fraction")
    for sp in (r, c):
        sp.add_argument("--split", default="valid")
        sp.add_argument("--strategy", choices=AUC_STRATEGIES, default=STRATEGY,
                        help="curva di danno: random (stanze, default) | crop | patch")
        sp.add_argument("--robust", action="store_true",
                        help="metro §38: media delle AUC su nowalls-random, crop e patch "
                             "(ignora --strategy)")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    {"rank": _cmd_rank, "compare": _cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    main()
