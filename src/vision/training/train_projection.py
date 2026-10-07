"""Train the projection head on cached positive pairs (backbone frozen).

Works on pre-extracted vectors (`pairs.npz`), no backbone forward. InfoNCE loss: pulls the anchor (full plan)
towards its positives (degraded views), pushes away the other plans in the batch.
Fit on `train` rows, val-loss on `valid` rows drives early stopping; the best weights go to `head.file`.
With `training.selection=probe_partial` the epoch is chosen on the partial-retrieval probe AUC
(`retrieval_probe.py`) instead of val-loss; `head.file` must then differ from `head.pt`.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW

from src.vision.data.vision_damage import check_head_damage
from src.vision.models.projection_head import ProjectionHead, info_nce
from src.vision.training.retrieval_probe import load_probe, probe_auc
from src.vision.utils.config import load_vision_config


def _train_epoch(head, optim, anchors, positives, idx, batch_size, tau, V, gen, device):
    """One training epoch over `idx` (random positive among the V views)."""
    head.train()
    perm = idx[torch.randperm(idx.numel(), device=device, generator=gen)]
    total = 0.0
    for start in range(0, perm.numel(), batch_size):
        b = perm[start:start + batch_size]
        v = torch.randint(0, V, (b.numel(),), device=device, generator=gen)
        loss = info_nce(head(anchors[b]), head(positives[b, v]), tau)
        optim.zero_grad()
        loss.backward()
        optim.step()
        total += loss.item() * b.numel()
    return total / perm.numel()


@torch.no_grad()
def _val_loss(head, anchors, positives, idx, batch_size, tau):
    """Deterministic val-loss: fixed order, positive v=0, training batch size (same number of negatives)."""
    head.eval()
    total, seen = 0.0, 0
    for start in range(0, idx.numel(), batch_size):
        b = idx[start:start + batch_size]
        if b.numel() < 2:                       # InfoNCE needs >=2 (in-batch negatives)
            continue
        total += info_nce(head(anchors[b]), head(positives[b, 0]), tau).item() * b.numel()
        seen += b.numel()
    return total / max(seen, 1)


def train(config) -> None:
    save_dir = Path(config.retrieval.save_dir)
    data = np.load(save_dir / "pairs.npz")
    device = config.model.device

    anchors   = torch.from_numpy(data["anchors"]).float().to(device)      # [M, D]
    positives = torch.from_numpy(data["positives"]).float().to(device)    # [M, V, D]
    M, V, D = positives.shape

    # train rows fit, valid rows early-stop; all train if no split
    if "splits" in data:
        splits = data["splits"].astype(str)
        train_idx = torch.from_numpy(np.where(splits == "train")[0]).to(device)
        valid_idx = torch.from_numpy(np.where(splits == "valid")[0]).to(device)
    else:
        train_idx = torch.arange(M, device=device)
        valid_idx = torch.empty(0, dtype=torch.long, device=device)
    has_val = valid_idx.numel() >= 2
    print(f"[train] coppie: M={M} V={V} D={D} | train={train_idx.numel()} valid={valid_idx.numel()} | device={device}")

    tcfg = config.training
    selection = str(tcfg.get("selection") or "val_loss")
    head_file = str(config.head.get("file") or "head.pt")
    if selection not in ("val_loss", "probe_partial"):
        raise ValueError(f"training.selection='{selection}' (attesi: val_loss | probe_partial)")
    if selection == "probe_partial" and head_file == "head.pt":
        # head.pt is the val-loss reference: never overwrite it
        raise ValueError("selection=probe_partial richiede head.file diverso da head.pt "
                         "(es. head.file=head_probe.pt)")
    probe = load_probe(save_dir, device) if selection == "probe_partial" else None
    if probe is not None:
        print(f"[train] selezione su probe partial: {probe['q_raw'].shape[1]} query x "
              f"f={probe['fractions']} | gallery {probe['gallery'].shape[0]}")

    # pairs, probe and config must share the damage; files without it are `random`
    damage = check_head_damage(tcfg.get("damage", "random"))
    pairs_damage = str(data["damage"]) if "damage" in data else "random"
    probe_damage = str(probe["meta"].get("strategy", "random")) if probe is not None else damage
    if pairs_damage != damage or probe_damage != damage:
        raise ValueError(f"training.damage={damage!r}, ma pairs.npz e' {pairs_damage!r} e la probe "
                         f"{probe_damage!r}: ricostruiscile con lo stesso danno (niente REUSE_PROBE)")
    print(f"[train] danno di coppie e probe: {damage}")

    # seeded init
    torch.manual_seed(int(tcfg.seed))
    head = ProjectionHead(D, config.head.hidden_dim, config.head.out_dim).to(device)
    optim = AdamW(head.parameters(), lr=float(tcfg.lr), weight_decay=float(tcfg.weight_decay))

    batch_size = int(tcfg.batch_size)
    tau = float(tcfg.temperature)
    max_epochs = int(tcfg.epochs)
    patience = int(tcfg.get("patience", 0))            # 0 = no early stopping
    gen = torch.Generator(device=device).manual_seed(int(tcfg.seed))

    wb = _wandb_init(config, {"D": D, "V": V, "train": int(train_idx.numel()), "valid": int(valid_idx.numel())})

    best_val, best_state, wait = float("inf"), None, 0
    best_auc, best_epoch, history = float("-inf"), None, []
    # probe mode also keeps the val-loss-best checkpoint of the same run (no early stop on it)
    vl_best, vl_state, vl_epoch = float("inf"), None, None
    for epoch in range(max_epochs):
        tr = _train_epoch(head, optim, anchors, positives, train_idx, batch_size, tau, V, gen, device)
        vl = _val_loss(head, anchors, positives, valid_idx, batch_size, tau) if has_val else None
        auc, auc_f = probe_auc(head, probe) if probe is not None else (None, None)

        msg = f"[train] epoch {epoch+1:02d}/{max_epochs} | loss {tr:.4f}"
        if has_val:
            msg += f" | val {vl:.4f}"
        if probe is not None:
            msg += f" | probe AUC {auc:.4f} (" + " ".join(f"f{f}={v:.4f}" for f, v in auc_f.items()) + ")"
        print(msg)
        history.append({"epoch": epoch + 1, "train_loss": tr, "val_loss": vl,
                        "probe_auc": auc, "probe_mrr": auc_f})
        if wb is not None:
            wb.log({"epoch": epoch + 1, "train_loss": tr, **({"val_loss": vl} if has_val else {}),
                    **({"probe_auc": auc} if probe is not None else {})})

        if probe is not None:                          # selection on the probe
            if has_val and vl < vl_best - 1e-4:
                vl_best, vl_state, vl_epoch = vl, deepcopy(head.state_dict()), epoch + 1
            if auc > best_auc + 1e-4:
                best_auc, best_epoch, best_state, wait = auc, epoch + 1, deepcopy(head.state_dict()), 0
            else:
                wait += 1
                if patience and wait >= patience:
                    print(f"[train] early stop a epoch {epoch+1} (probe AUC massima {best_auc:.4f})")
                    break
        elif has_val:                                  # early stopping on val-loss
            if vl < best_val - 1e-4:
                best_val, best_state, wait = vl, deepcopy(head.state_dict()), 0
            else:
                wait += 1
                if patience and wait >= patience:
                    print(f"[train] early stop a epoch {epoch+1} (val minima {best_val:.4f})")
                    break

    state = best_state if best_state is not None else head.state_dict()
    torch.save(state, save_dir / head_file)
    if probe is not None:
        print(f"[train] salvato {save_dir/head_file} | epoca {best_epoch} | probe AUC {best_auc:.4f}")
        vl_file = None
        if vl_state is not None:
            vl_file = f"{Path(head_file).stem}_valloss.pt"
            torch.save(vl_state, save_dir / vl_file)
            print(f"[train] salvato {save_dir/vl_file} | epoca {vl_epoch} (val-loss minima, "
                  f"stessa run: termine di confronto del criterio)")
        hist_path = save_dir / f"{Path(head_file).stem}_history.json"
        hist_path.write_text(json.dumps({"selection": selection, "best_epoch": best_epoch,
                                         "best_probe_auc": best_auc, "probe_meta": probe["meta"],
                                         "valloss_epoch": vl_epoch, "valloss_file": vl_file,
                                         "history": history}, indent=1))
    else:
        print(f"[train] salvato {save_dir/head_file}" + (f" | best val {best_val:.4f}" if has_val else ""))
    if wb is not None:
        wb.summary["best_val_loss"] = best_val if has_val else None
        if probe is not None:
            wb.summary["best_probe_auc"] = best_auc
        wb.finish()


def _wandb_init(config, extra):
    """Init wandb if `wandb.enabled`, else None."""
    wb = config.get("wandb", {})
    if not bool(wb.get("enabled", False)):
        return None
    import wandb
    tcfg = config.training
    wandb.init(
        project=wb.get("project", "cvcs-head"),
        mode=wb.get("mode", "online"),
        name=f"{config.model.name}_{config.model.variant}",
        config={
            "model": config.model.name, "variant": config.model.variant,
            "lr": float(tcfg.lr), "temperature": float(tcfg.temperature),
            "batch_size": int(tcfg.batch_size), "epochs": int(tcfg.epochs),
            "patience": int(tcfg.get("patience", 0)),
            "hidden_dim": config.head.hidden_dim, "out_dim": config.head.out_dim,
            **extra,
        },
    )
    return wandb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/vision_retrieval.yaml")
    args, overrides = parser.parse_known_args()
    train(load_vision_config(args.config, overrides))


if __name__ == "__main__":
    main()
