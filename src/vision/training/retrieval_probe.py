"""Partial-retrieval probe to select the head checkpoint.

Self-recovery MRR AUC over f in {0.25, 0.5, 0.75} (masking `random`, split `valid`), computed
inside training on a held-out query set:
  - queries are disjoint from the evaluation queries (same `sample_query_rows` params, same gallery);
  - masking seed is `training.probe.seed + row`, with the same damage functions as `evaluate.py`;
  - backbone frozen: degraded-query RAW features are computed once (`probe_partial.npz`),
    each epoch only applies the head to gallery and queries.
Score mirrors `evaluate.py` with `head.enabled=true whitening.enabled=false`: cosine on L2 head
outputs, self-rank over the whole gallery, reciprocal rank 0 beyond `max(eval.k_values)`.

Usage (GPU, one encoder/variant at a time):

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
from src.vision.data.vision_damage import (
    check_head_damage, damaged_query, make_patch_context, resolve_patch_size, room_damage_image,
)
from src.vision.utils.config import load_vision_config

PROBE_FILE = "probe_partial.npz"
# default of `training.damage`; train_projection rejects a probe damage differing from the pairs'
PROBE_STRATEGY = "random"


# --- build (once per encoder/variant) ---

def probe_rows(image_paths: list[str], config) -> list[int]:
    """Probe query rows: split valid, excluding the evaluation queries (recomputed as in evaluate.py)."""
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
    """Degrade the probe queries and save their RAW features to `probe_partial.npz` (returns the path)."""
    from src.vision.evaluation.evaluate import load_pipeline

    # RAW only: head and whitening are applied later, in training
    raw_cfg = OmegaConf.merge(config, OmegaConf.create(
        {"head": {"enabled": False}, "whitening": {"enabled": False}}))
    pipeline = load_pipeline(raw_cfg)
    paths = pipeline.image_paths

    rows = probe_rows(paths, config)
    metas = {qi: load_metadata(paths[qi]) for qi in rows}
    rows = [qi for qi in rows if metas[qi] is not None]
    if config.training.probe.get("damages"):
        return _build_probe_mixed(config, pipeline, paths, rows, metas)

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
    # `gallery_rows`: rows of embeddings.npy in this gallery's order, so training rebuilds the same gallery
    disk_row = {Path(p).stem: i for i, p in
                enumerate(json.loads((Path(config.retrieval.save_dir) / "image_paths.json").read_text()))}
    gallery_rows = np.asarray([disk_row[n] for n in names], dtype=np.int64)
    np.savez(out, q_raw=np.stack(feats), q_rows=np.asarray(rows, dtype=np.int64),
             gallery_rows=gallery_rows, meta=np.array(json.dumps(meta)))
    print(f"[probe] salvato {out}: q_raw {np.stack(feats).shape}")
    return out


def _build_probe_mixed(config, pipeline, paths, rows, metas) -> Path:
    """Probe over several damages with the evaluation's `damaged_query`: one run per (damage, f).

    Mean over runs = mean over damages of their AUC. Written to `training.probe.file`
    (never `probe_partial.npz`), never overwritten."""
    from src.vision.data.projection_pairs import check_damage_mix
    from src.vision.evaluation.evaluate import _transform_image_size

    pcfg = config.training.probe
    damages = check_damage_mix(pcfg.damages)
    out = Path(config.retrieval.save_dir) / str(pcfg.get("file") or "")
    if out.name in ("", PROBE_FILE):
        raise ValueError("training.probe.damages needs training.probe.file different from "
                         f"{PROBE_FILE} (the probe of the current head)")
    if out.exists():
        raise FileExistsError(f"{out} exists already: nothing is overwritten")
    fractions = [float(f) for f in pcfg.fractions]
    seed = int(pcfg.seed)
    strat = config.partial.strategies
    params = {"nowalls_random": {}, "crop": {"tolerance": float(strat.crop.tolerance),
                                            "max_tries": int(strat.crop.max_tries)},
              "patch": {"tolerance": float(strat.patch.tolerance)}}
    patch_ctx = None
    if "patch" in damages:
        p = resolve_patch_size(pipeline.encoder, config.model.kwargs.get("patch_size"),
                               _transform_image_size(pipeline.transform))
        patch_ctx = make_patch_context(pipeline.transform, p)
    open_boundary = bool(config.partial.get("open_boundary", True))

    feats, runs, n_empty = [], [], []
    for d in damages:
        for f in fractions:
            tensors, empty = [], 0
            for qi in rows:
                x, info = damaged_query(paths[qi], metas[qi], d, {"fraction": f, **params[d]},
                                        random.Random(seed + qi), open_boundary,
                                        pipeline.transform, patch_ctx)
                empty += int(not info["removed_any"])
                tensors.append(x)
            with torch.no_grad():
                emb = [pipeline.encoder(torch.stack(tensors[i:i + 64]).to(pipeline.device)).float().cpu().numpy()
                       for i in range(0, len(tensors), 64)]
            feats.append(np.concatenate(emb).astype("float32"))
            runs.append([d, f])
            n_empty.append(empty)
            print(f"[probe] {d} f={f}: {len(rows)} query degradate ({empty} senza danno)")

    names = [Path(p).stem for p in paths]
    meta = {
        "fractions": [f for _, f in runs], "runs": runs, "damages": list(damages),
        "seed": seed, "strategy": f"mix:{'+'.join(damages)}", "open_boundary": open_boundary,
        "n_empty": n_empty, "gallery_n": len(names), "gallery_sha1": gallery_sha1(names),
        "gallery_names": config.eval.get("gallery_names"),
        "eval_excluded": {"num_queries": int(config.eval.num_queries),
                          "seed": int(config.eval.seed), "split": config.eval.get("split")},
        "max_k": int(max(config.eval.k_values)),
    }
    disk_row = {Path(p).stem: i for i, p in
                enumerate(json.loads((Path(config.retrieval.save_dir) / "image_paths.json").read_text()))}
    gallery_rows = np.asarray([disk_row[n] for n in names], dtype=np.int64)
    np.savez(out, q_raw=np.stack(feats), q_rows=np.asarray(rows, dtype=np.int64),
             gallery_rows=gallery_rows, meta=np.array(json.dumps(meta)))
    print(f"[probe] salvato {out}: q_raw {np.stack(feats).shape} ({len(runs)} run danno x f)")
    return out


# --- use in training ---

def load_probe(save_dir: str | Path, device: str) -> dict:
    """Reload the probe and the RAW gallery in build order.

    Raises FileNotFoundError if the probe is not built, ValueError if the gallery sha1 differs.
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
    """Per-query reciprocal rank of the self row; 0 beyond `max_rank`.

    `q` [P, d] and `gallery` [N, d] are L2-normalised (dot = cosine, as FAISS IndexFlatIP).
    Rank = 1 + number of rows with score strictly greater than the self's.
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
    """Probe AUC (mean over f of MRR) for `head`; returns (auc, {f: mrr}), as `robustness_auc`."""
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
