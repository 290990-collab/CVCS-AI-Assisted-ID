# src/vision/training/retrieval_probe.py

"""
Probe di retrieval PARTIAL per scegliere il checkpoint della head (fase **B.6**).

Il problema
-----------
La head di oggi e' scelta sull'epoca di val-loss InfoNCE minima (rilievo A3), e
la val-loss non predice il retrieval (status.md). Con il criterio A.5 la head e'
diventata la config congelata del vision: il suo numero e' un **limite
inferiore**, perche' e' scelta con un criterio sbagliato (§ 24.3).

La probe
--------
Stessa misura del criterio A.5, in piccolo e dentro il training: AUC della curva
self-recovery MRR su f in {0.25, 0.5, 0.75}, masking `random`, split `valid`.

Tre scelte, tutte per non contaminare la valutazione:

1. **Query della probe DISGIUNTE da quelle di valutazione.** Si esclude il set
   che `evaluate.py` campiona (`eval.num_queries`, `eval.seed`, `eval.split`,
   sulla stessa gallery): altrimenti la head sarebbe scelta sulle stesse query
   su cui poi si confronta con le config frozen, e il confronto sul valid
   sarebbe ottimista per costruzione. Il test resta intoccato (pool = valid).
2. **Seed di masking diverso** (`training.probe.seed + riga`), stesse funzioni
   di degrado di `evaluate.py` (`make_partial_query`, `open_boundary` dal config;
   dal 15 set `training.damage=nowalls_random` usa `make_wiped_query`, status.md § 46).
3. **Backbone frozen ⇒ feature RAW delle query degradate calcolate UNA volta**
   (`probe_partial.npz`, CLI di questo modulo). A ogni epoca si applica solo la
   head a gallery e query: pochi secondi su GPU.

Il punteggio rispecchia `evaluate.py` con `head.enabled=true whitening.enabled=false`
(la config congelata, § 24.4): coseno sugli output L2 della head, rank del self
sull'intera gallery (quella condivisa, se `eval.gallery_names` e' impostato), e
reciprocal rank 0 oltre `max(eval.k_values)`, come `partial_rows` di
`evaluate.py` (la lista col self e' troncata a max_k).

Uso (GPU, minuti — un encoder/variante per volta):

    python -m src.vision.training.retrieval_probe model.name=dinov3 model.variant=natural \\
        model.kwargs.pooling=natural
    python -m src.vision.training.train_projection model.name=dinov3 ... \\
        training.selection=probe_partial head.file=head_probe.pt
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from src.data.rplan_metadata import get_split, load_metadata
from src.evaluation.perquery import gallery_sha1
from src.vision.data.vision_damage import check_head_damage, room_damage_image
from src.vision.utils.config import load_vision_config

PROBE_FILE = "probe_partial.npz"
# Default di `training.damage` (storico). Dal 15 set la probe usa il danno del
# config, lo stesso delle coppie: `train_projection` rifiuta i due diversi.
PROBE_STRATEGY = "random"


# ----------------------------------------------------------------------
# Costruzione (una volta per encoder/variante).
# ----------------------------------------------------------------------

def probe_rows(image_paths: list[str], config) -> list[int]:
    """Righe-query della probe: split valid, DISGIUNTE dalle query di valutazione.

    Le query di valutazione si ricalcolano con la stessa funzione e gli stessi
    parametri di `evaluate.py`, sulla stessa gallery (gia' ristretta), quindi
    l'esclusione e' esatta, non approssimata.
    """
    from src.vision.evaluation.evaluate import sample_query_rows

    ecfg = config.eval
    eval_rows = set(sample_query_rows(image_paths, int(ecfg.num_queries),
                                      int(ecfg.seed), ecfg.get("split")))
    pool = [i for i, p in enumerate(image_paths)
            if get_split(p) == "valid" and i not in eval_rows]
    pcfg = config.training.probe
    n = min(int(pcfg.num_queries), len(pool))
    rows = random.Random(int(pcfg.seed)).sample(pool, n)
    print(f"[probe] {n} query valid (pool {len(pool)}, escluse {len(eval_rows)} "
          f"query di valutazione)")
    return rows


@torch.no_grad()
def _encode(pipeline, images, batch_size: int = 64) -> np.ndarray:
    out = []
    for i in range(0, len(images), batch_size):
        x = torch.stack([pipeline.transform(img) for img in images[i:i + batch_size]])
        out.append(pipeline.encoder(x.to(pipeline.device)).float().cpu().numpy())
    return np.concatenate(out).astype("float32")


def build_probe(config) -> Path:
    """Degrada le query della probe e ne salva le feature RAW in `probe_partial.npz`.

    Returns:
        Il path del file scritto (in `retrieval.save_dir`).
    """
    from src.vision.evaluation.evaluate import load_pipeline

    # Solo RAW: head e whitening si applicano dopo, nel training.
    raw_cfg = OmegaConf.merge(config, OmegaConf.create(
        {"head": {"enabled": False}, "whitening": {"enabled": False}}))
    pipeline = load_pipeline(raw_cfg)
    paths = pipeline.image_paths

    rows = probe_rows(paths, config)
    metas = {qi: load_metadata(paths[qi]) for qi in rows}
    rows = [qi for qi in rows if metas[qi] is not None]

    fractions = [float(f) for f in config.training.probe.fractions]
    seed = int(config.training.probe.seed)
    open_boundary = bool(config.partial.get("open_boundary", True))
    strategy = check_head_damage(config.training.get("damage", PROBE_STRATEGY))
    print(f"[probe] danno: {strategy}")

    feats, n_empty = [], []
    for f in fractions:
        images, empty = [], 0
        for qi in rows:
            img, removed = room_damage_image(paths[qi], metas[qi], strategy,
                                             {"fraction": f}, random.Random(seed + qi),
                                             open_boundary)
            empty += int(not removed)
            images.append(img)
        feats.append(_encode(pipeline, images))
        n_empty.append(empty)
        print(f"[probe] f={f}: {len(rows)} query degradate ({empty} senza stanze rimosse)")

    out = Path(config.retrieval.save_dir) / PROBE_FILE
    names = [Path(p).stem for p in paths]
    meta = {
        "fractions": fractions, "seed": seed, "strategy": strategy,
        "open_boundary": open_boundary, "n_empty": n_empty,
        "gallery_n": len(names), "gallery_sha1": gallery_sha1(names),
        "gallery_names": config.eval.get("gallery_names"),
        "eval_excluded": {"num_queries": int(config.eval.num_queries),
                          "seed": int(config.eval.seed), "split": config.eval.get("split")},
        "max_k": int(max(config.eval.k_values)),
    }
    # `gallery_rows`: righe di embeddings.npy (ordine su disco) nell'ordine della
    # gallery usata qui — cosi' il training ricostruisce la stessa gallery.
    disk_row = {Path(p).stem: i for i, p in
                enumerate(json.loads((Path(config.retrieval.save_dir) / "image_paths.json").read_text()))}
    gallery_rows = np.asarray([disk_row[n] for n in names], dtype=np.int64)
    np.savez(out, q_raw=np.stack(feats), q_rows=np.asarray(rows, dtype=np.int64),
             gallery_rows=gallery_rows, meta=np.array(json.dumps(meta)))
    print(f"[probe] salvato {out}: q_raw {np.stack(feats).shape}")
    return out


# ----------------------------------------------------------------------
# Uso nel training.
# ----------------------------------------------------------------------

def load_probe(save_dir: str | Path, device: str) -> dict:
    """Rilegge la probe e la gallery RAW nello stesso ordine della costruzione.

    Raises:
        FileNotFoundError: se la probe non e' stata costruita (messaggio con il comando).
        ValueError: se la gallery su disco non ricostruisce lo sha1 della probe.
    """
    save_dir = Path(save_dir)
    path = save_dir / PROBE_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"{path} mancante: costruiscila prima con "
            "`python -m src.vision.training.retrieval_probe <override del modello>`")
    with np.load(path, allow_pickle=False) as z:
        q_raw, q_rows, g_rows = z["q_raw"], z["q_rows"], z["gallery_rows"]
        meta = json.loads(str(z["meta"].item()))
    paths = json.loads((save_dir / "image_paths.json").read_text())
    names = [Path(paths[i]).stem for i in g_rows]
    if gallery_sha1(names) != meta["gallery_sha1"]:
        raise ValueError(f"{path}: la gallery su disco non corrisponde a quella della probe")
    gallery = np.load(save_dir / "embeddings.npy")[g_rows]
    return {
        "gallery": torch.from_numpy(gallery).float().to(device),     # [N, D]
        "q_raw": torch.from_numpy(q_raw).float().to(device),         # [F, P, D]
        "q_rows": torch.from_numpy(q_rows).long().to(device),        # [P]
        "fractions": [float(f) for f in meta["fractions"]],
        "max_k": int(meta["max_k"]),
        "meta": meta,
    }


@torch.no_grad()
def self_reciprocal_ranks(q: torch.Tensor, gallery: torch.Tensor, self_rows: torch.Tensor,
                          max_rank: int, chunk: int = 256) -> torch.Tensor:
    """Reciprocal rank del self per ogni query; 0 oltre `max_rank` (come il top-k).

    `q` [P, d] e `gallery` [N, d] sono gia' L2-normalizzati: il prodotto scalare
    e' il coseno, lo stesso punteggio di FAISS IndexFlatIP. Rank = 1 + numero di
    righe con punteggio STRETTAMENTE maggiore di quello del self.
    """
    out = []
    for i in range(0, q.shape[0], chunk):
        s = q[i:i + chunk] @ gallery.T                                # [c, N]
        rows = self_rows[i:i + chunk]
        own = s.gather(1, rows[:, None])                              # [c, 1]
        rank = 1 + (s > own).sum(dim=1)
        rr = torch.where(rank <= max_rank, 1.0 / rank.float(), torch.zeros_like(rank, dtype=torch.float))
        out.append(rr)
    return torch.cat(out)


@torch.no_grad()
def probe_auc(head, probe: dict, chunk: int = 8192) -> tuple[float, dict]:
    """AUC della probe (media su f del MRR) per la head data.

    Returns:
        (auc, {f: mrr}) — stessa ricetta di `src.evaluation.robustness_auc`.
    """
    head.eval()
    g = probe["gallery"]
    z = torch.cat([head(g[i:i + chunk]) for i in range(0, g.shape[0], chunk)])
    z = torch.nn.functional.normalize(z, dim=-1)
    per_f = {}
    for fi, f in enumerate(probe["fractions"]):
        qz = torch.nn.functional.normalize(head(probe["q_raw"][fi]), dim=-1)
        rr = self_reciprocal_ranks(qz, z, probe["q_rows"], probe["max_k"])
        per_f[f] = float(rr.mean())
    return float(np.mean(list(per_f.values()))), per_f


def main():
    parser = argparse.ArgumentParser(description="Costruisce la probe partial della fase B.6")
    parser.add_argument("--config", default="configs/vision_retrieval.yaml")
    args, overrides = parser.parse_known_args()
    build_probe(load_vision_config(args.config, overrides))


if __name__ == "__main__":
    main()
