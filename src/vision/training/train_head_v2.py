"""Train the residual query-only head v2 on several damages.

    v = L2(whiten(frozen raw))     whitening = the frozen vision's (fit on train), read from its qvec file
    z = L2(v + MLP(v))             ResidualQueryHead, last layer at zero (identity at start)
    gallery = v (unchanged)        loss: symmetric InfoNCE between z(damaged view) and v(full plan)

Data: `pairs_v2.npz` (6 views per plan, damages cycling nowalls_random / crop / patch, fraction U[0.25, 0.75];
only `train` rows are fitted) and `probe_v2.npz` (1000 valid queries disjoint from the 2000 evaluation ones,
3 damages x f 0.25/0.5/0.75). Epoch selected on the probe only: mean of the 9 MRR = mean of the three damage AUCs.
Recipe: tau 0.07, batch 4096, AdamW lr 1e-3, wd 1e-4, seed 42, max 5000 epochs, patience 100.

Writes (refuses to overwrite) embeddings/vision/pespatial/gem/head_v2.pt (state + training whitening)
and head_v2_history.json.

Usage (GPU, minutes):
    python -m src.vision.training.train_head_v2
    python -m src.vision.training.train_head_v2 --smoke      # CPU, tiny, writes to a temp dir
"""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from src.evaluation.perquery import gallery_sha1
from src.vision.models.projection_head import ResidualQueryHead, info_nce
from src.vision.training.retrieval_probe import self_reciprocal_ranks

SAVE_DIR = Path("embeddings/vision/pespatial/gem")
PAIRS_FILE, PROBE_FILE = "pairs_v2.npz", "probe_v2.npz"
HEAD_FILE, HISTORY_FILE = "head_v2.pt", "head_v2_history.json"
# frozen vision whitening (`pespatial/gem` + whiten-train), carried by the replicas' qvec files
FROZEN_QVEC = "results/queryvec/valid/vision_pespatial_gem_whiten-train_partial-nowalls-random-f0.25_valid.npz"
RECIPE = {"temperature": 0.07, "batch_size": 4096, "lr": 1e-3, "weight_decay": 1e-4, "seed": 42,
          "epochs": 5000, "patience": 100, "hidden_dim": 1024, "min_improvement": 1e-4}


def whiten_l2(raw: np.ndarray, mean: np.ndarray, matrix: np.ndarray, chunk: int = 65536) -> np.ndarray:
    """The vision job's whitening (`_apply_whitening`): (x - mean) @ matrix, then L2. [..., D] -> [..., D']."""
    shape = raw.shape
    x = raw.reshape(-1, shape[-1])
    out = np.empty((x.shape[0], matrix.shape[1]), dtype=np.float32)
    for i in range(0, x.shape[0], chunk):
        y = (x[i:i + chunk].astype(np.float32) - mean) @ matrix
        out[i:i + chunk] = y / np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
    return out.reshape(*shape[:-1], matrix.shape[1])


def frozen_whitening() -> tuple[np.ndarray, np.ndarray, str]:
    """(mean, matrix, sha1) of the frozen vision's whitening (seed-42 valid job).

    Refits are not bit-identical across nodes (near-equal eigenvalues rotate within their subspace). A head
    sees coordinates, so v2 always uses this whitening: saved in the checkpoint, imposed by `evaluate.load_pipeline`.
    """
    from src.evaluation.query_vectors import load_qvec
    ref = load_qvec(FROZEN_QVEC)
    mean, matrix = ref.whiten_mean.astype(np.float32), ref.whiten_matrix.astype(np.float32)
    return mean, matrix, whitening_sha1(mean, matrix)


def whitening_sha1(mean, matrix) -> str:
    h = hashlib.sha1()
    for a in (mean, matrix):
        h.update(np.ascontiguousarray(np.asarray(a, dtype=np.float32)).tobytes())
    return h.hexdigest()


def load_probe_v2(save_dir: Path, mean, matrix, device) -> dict:
    """probe_v2.npz with the gallery rebuilt and checked by sha1, everything whitened + L2."""
    with np.load(save_dir / PROBE_FILE, allow_pickle=False) as z:
        q_raw, q_rows, g_rows = z["q_raw"], z["q_rows"], z["gallery_rows"]
        meta = json.loads(str(z["meta"].item()))
    paths = json.loads((save_dir / "image_paths.json").read_text())
    if gallery_sha1([Path(paths[i]).stem for i in g_rows]) != meta["gallery_sha1"]:
        raise ValueError(f"{save_dir / PROBE_FILE}: gallery on disk does not match the probe")
    gallery = whiten_l2(np.load(save_dir / "embeddings.npy")[g_rows], mean, matrix)
    return {"gallery": torch.from_numpy(gallery).to(device),
            "q": torch.from_numpy(whiten_l2(q_raw, mean, matrix)).to(device),     # [R, P, D]
            "q_rows": torch.from_numpy(q_rows).long().to(device),
            "runs": [tuple(r) for r in meta["runs"]], "max_k": int(meta["max_k"]), "meta": meta}


@torch.no_grad()
def probe_auc(head, probe: dict) -> tuple[float, dict]:
    """Mean over the (damage, f) runs of the self-recovery MRR = mean of the per-damage AUCs."""
    head.eval()
    per = {}
    for ri, (dmg, f) in enumerate(probe["runs"]):
        rr = self_reciprocal_ranks(head(probe["q"][ri]), probe["gallery"], probe["q_rows"], probe["max_k"])
        per[f"{dmg}@{float(f):g}"] = float(rr.mean())
    return float(np.mean(list(per.values()))), per


def train(save_dir: Path = SAVE_DIR, out_dir: Path | None = None, recipe: dict = RECIPE, device: str | None = None,
          whitening=None, max_train_rows: int | None = None, max_probe_queries: int | None = None) -> dict:
    out_dir = Path(out_dir or save_dir)
    for f in (HEAD_FILE, HISTORY_FILE):
        if (out_dir / f).exists():
            raise SystemExit(f"!! {out_dir / f} esiste gia': niente sovrascritture")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    mean, matrix, wsha = whitening or frozen_whitening()

    with np.load(save_dir / PAIRS_FILE, allow_pickle=False) as z:
        splits = z["splits"].astype(str)
        rows = np.where(splits == "train")[0]
        if max_train_rows:
            rows = rows[:max_train_rows]
        anchors = whiten_l2(z["anchors"][rows], mean, matrix)
        positives = whiten_l2(z["positives"][rows], mean, matrix)
        view_damages = [str(x) for x in z["view_damages"]] if "view_damages" in z else None
        pairs_damage = str(z["damage"]) if "damage" in z else None
    M, V, D = positives.shape
    print(f"[head_v2] coppie train {M} x {V} viste ({view_damages}) | D={D} | whitening {wsha[:12]} | {device}")
    probe = load_probe_v2(save_dir, mean, matrix, device)
    if max_probe_queries:                           # smoke test only
        probe["q"], probe["q_rows"] = probe["q"][:, :max_probe_queries], probe["q_rows"][:max_probe_queries]
    print(f"[head_v2] probe: {len(probe['runs'])} run x {probe['q'].shape[1]} query | gallery {probe['gallery'].shape[0]}")

    A = torch.from_numpy(anchors).to(device)
    P = torch.from_numpy(positives).to(device)
    torch.manual_seed(int(recipe["seed"]))
    head = ResidualQueryHead(D, int(recipe["hidden_dim"])).to(device)
    optim = torch.optim.AdamW(head.parameters(), lr=float(recipe["lr"]), weight_decay=float(recipe["weight_decay"]))
    gen = torch.Generator(device=device).manual_seed(int(recipe["seed"]))
    bs, tau = int(recipe["batch_size"]), float(recipe["temperature"])

    auc0, per0 = probe_auc(head, probe)            # epoch 0 = identity head = frozen vision
    print(f"[head_v2] epoch 0 (identita' = congelato) | probe AUC {auc0:.4f}")
    history = [{"epoch": 0, "train_loss": None, "probe_auc": auc0, "probe_mrr": per0}]
    best_auc, best_epoch, best_state, wait = auc0, 0, deepcopy(head.state_dict()), 0
    for epoch in range(1, int(recipe["epochs"]) + 1):
        head.train()
        perm = torch.randperm(M, device=device, generator=gen)
        total = 0.0
        for s in range(0, M, bs):
            b = perm[s:s + bs]
            if b.numel() < 2:
                continue
            v = torch.randint(0, V, (b.numel(),), device=device, generator=gen)
            loss = info_nce(head(P[b, v]), A[b], tau)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total += loss.item() * b.numel()
        auc, per = probe_auc(head, probe)
        history.append({"epoch": epoch, "train_loss": total / M, "probe_auc": auc, "probe_mrr": per})
        print(f"[head_v2] epoch {epoch} | loss {total / M:.4f} | probe AUC {auc:.4f}")
        if auc > best_auc + float(recipe["min_improvement"]):
            best_auc, best_epoch, best_state, wait = auc, epoch, deepcopy(head.state_dict()), 0
        else:
            wait += 1
            if wait >= int(recipe["patience"]):
                print(f"[head_v2] stop a epoch {epoch} (probe AUC massima {best_auc:.4f} a epoch {best_epoch})")
                break

    ckpt = {"state_dict": {k: v.cpu() for k, v in best_state.items()}, "dim": D, "hidden_dim": int(recipe["hidden_dim"]),
            "whiten_mean": torch.from_numpy(mean), "whiten_matrix": torch.from_numpy(matrix), "whitening_sha1": wsha,
            "best_epoch": best_epoch, "best_probe_auc": best_auc, "recipe": dict(recipe),
            "pairs": {"file": PAIRS_FILE, "damage": pairs_damage, "view_damages": view_damages, "train_rows": int(M)},
            "probe": {"file": PROBE_FILE, "runs": [list(r) for r in probe["runs"]]}}
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out_dir / HEAD_FILE)
    hist = {k: v for k, v in ckpt.items() if k not in ("state_dict", "whiten_mean", "whiten_matrix")}
    hist.update({"rule": "status.md §64.2", "written": datetime.now().isoformat(timespec="seconds"),
                 "probe_auc_epoch0_frozen": auc0, "history": history})
    (out_dir / HISTORY_FILE).write_text(json.dumps(hist, indent=1))
    print(f"[head_v2] salvato {out_dir / HEAD_FILE} | epoca {best_epoch} | probe AUC {best_auc:.4f} "
          f"(epoca 0, congelato: {auc0:.4f})")
    return hist


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="train_head_v2")
    ap.add_argument("--smoke", action="store_true", help="CPU, 3 epochs, 2000 train rows, temp output dir")
    a = ap.parse_args(argv)
    if a.smoke:
        import tempfile
        out = Path(tempfile.mkdtemp(prefix="head_v2_smoke_"))
        train(out_dir=out, recipe={**RECIPE, "epochs": 3, "batch_size": 512}, device="cpu", max_train_rows=2000,
              max_probe_queries=50)
        print(f"[head_v2] smoke ok -> {out}")
        return
    train()


if __name__ == "__main__":
    main()
