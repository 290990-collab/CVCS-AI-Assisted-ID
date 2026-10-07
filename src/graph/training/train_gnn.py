"""Self-supervised (InfoNCE) training of a graph encoder (GCN/GAT/GraphSAGE).

No labelled relevance pairs exist: two augmented views of each graph are positives, the
other plans in the batch negatives. Unlike the vision branch (head on cached vectors of a
frozen backbone), the network trains here and every epoch runs forward on the graphs.

Fit on `train`, early stopping on `valid`, best weights saved to `<save_dir>/encoder.pt`.
Feature normalisation statistics come from train only, saved to `<save_dir>/geom_stats.npz`.

Selection criterion is nDCG, not val-loss: weights are chosen by a retrieval probe on valid
(`RetrievalProbe`), because the two diverge (InfoNCE also pushes apart architecturally similar
plans in the same batch, which evaluation counts as relevant). Val-loss is still logged as a
convergence diagnostic; `--probe-every 0` restores the legacy val-loss criterion.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch_geometric.loader import DataLoader

from src.graph.evaluation.retrieval_probe import SELECTION_CRITERIA, RetrievalProbe
from src.graph.evaluation.robustness_probe import DEFAULT_PARTIAL_SEED, RobustnessProbe
from src.graph.graph_builder import NODE_FEATURE_DIM, NUM_RELATION_TYPES
from src.graph.graph_dataset import DEFAULT_ROOT, RplanGraphDataset
from src.graph.models import build_graph_encoder
from src.graph.training.augment import AugmentParams, two_views
from src.graph.transforms import (
    LOST_MARKER_DIM,
    build_node_transform,
    compute_geometry_stats,
    save_geometry_stats,
)


def info_nce(za: torch.Tensor, zp: torch.Tensor, temperature: float) -> torch.Tensor:
    """Symmetric InfoNCE/NT-Xent with in-batch negatives on L2-normalised view embeddings (positive of row i is zp[i])."""
    logits = (za @ zp.t()) / temperature
    labels = torch.arange(za.size(0), device=za.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def _encoder_kwargs(cfg) -> dict:
    """Encoder constructor kwargs, including encoder-specific ones."""
    kwargs = dict(
        # lost-marker adds one column
        in_dim=NODE_FEATURE_DIM + (LOST_MARKER_DIM if getattr(cfg, "lost_marker", False) else 0),
        hidden_dim=cfg.hidden_dim,
        out_dim=cfg.out_dim,
        num_layers=cfg.num_layers,
        pooling=cfg.pooling,
        dropout=cfg.dropout,
        # getattr: older checkpoints/entrypoints lack the field
        raw_skip=getattr(cfg, "raw_skip", False),
    )
    if cfg.encoder == "gat":
        kwargs.update(heads=cfg.heads, edge_dim=NUM_RELATION_TYPES, attn_dropout=cfg.attn_dropout)
    elif cfg.encoder == "sage":
        kwargs.update(aggr=cfg.aggr)
    return kwargs


def _train_epoch(encoder, optim, loader, aug, geom_stats, cfg, gen, device) -> float:
    """One training epoch: two views per batch, InfoNCE, backprop."""
    encoder.train()
    total, seen = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        if batch.num_graphs < 2:                 # InfoNCE needs >= 2 graphs
            continue
        view_a, view_b = two_views(batch, aug, gen, geom_stats)
        loss = info_nce(encoder(view_a), encoder(view_b), cfg.temperature)

        optim.zero_grad()
        loss.backward()
        optim.step()

        total += loss.item() * batch.num_graphs
        seen += batch.num_graphs
    return total / max(seen, 1)


@torch.no_grad()
def _val_loss(encoder, loader, aug, geom_stats, cfg, device) -> float:
    """Val-loss with deterministic augmentation (generator reseeded each call); diagnostic only."""
    encoder.eval()
    gen = torch.Generator(device=device).manual_seed(cfg.seed)
    total, seen = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        if batch.num_graphs < 2:
            continue
        view_a, view_b = two_views(batch, aug, gen, geom_stats)
        total += info_nce(encoder(view_a), encoder(view_b), cfg.temperature).item() * batch.num_graphs
        seen += batch.num_graphs
    return total / max(seen, 1)


class _ShadowSelection:
    """Legacy selection rule (full-probe nDCG, patience, epoch cap) run in shadow during a longer robustness-selected training.

    Mirrors the selection block of `train` (`> best + 1e-4`, `wait`, stop at `wait >= patience`);
    ignores epochs past `max_epoch` or its own stop, so the same run also yields the legacy checkpoint.
    """

    def __init__(self, patience: int, max_epoch: int):
        self.patience, self.max_epoch = patience, max_epoch
        self.best_score, self.best_state, self.best_epoch, self.best_scores = -float("inf"), None, 0, None
        self.wait, self.stopped, self.last_epoch = 0, False, 0

    def update(self, epoch: int, score: float, scores, encoder) -> None:
        """`epoch` is 1-based; the state dict is copied only on improvement."""
        if self.stopped or epoch > self.max_epoch:
            return
        self.last_epoch = epoch
        if score > self.best_score + 1e-4:
            self.best_score, self.best_state, self.wait = score, deepcopy(encoder.state_dict()), 0
            self.best_epoch, self.best_scores = epoch, scores
        else:
            self.wait += 1
            if self.patience and self.wait >= self.patience:
                self.stopped = True
                print(f"[train] shadow: early stop a epoch {epoch} "
                      f"(best {self.best_score:.4f} a epoch {self.best_epoch})")

    @property
    def complete(self) -> bool:
        """True once the legacy rule has stopped or hit its cap."""
        return self.stopped or self.last_epoch >= self.max_epoch


def train(cfg) -> None:
    device = cfg.device
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # statistics from train only, then a shared transform
    stats = None
    if cfg.normalize:
        stats = compute_geometry_stats(RplanGraphDataset(split="train"))
        save_geometry_stats(stats, save_dir / "geom_stats.npz")
    transform = build_node_transform(
        normalize=cfg.normalize, drop_self_loops=cfg.drop_self_loops, stats=stats,
        lost_marker=getattr(cfg, "lost_marker", False),
    )

    train_set = RplanGraphDataset(split="train", transform=transform)
    valid_set = RplanGraphDataset(split="valid", transform=transform)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=cfg.batch_size, shuffle=False)
    print(f"[train] {cfg.encoder} | train={len(train_set)} valid={len(valid_set)} | device={device}")

    # flip/rot act on raw [0,1] coordinates but features are z-scored: mean/std de-normalise and re-normalise
    aug = AugmentParams.from_cfg(cfg)
    geom_stats = None
    if stats is not None:
        geom_stats = (
            torch.as_tensor(stats["mean"], device=device).view(1, -1),
            torch.as_tensor(stats["std"], device=device).view(1, -1).clamp_min(1e-6),
        )
    elif aug.flip_prob > 0.0 or aug.rot_prob > 0.0:
        print("[train] ⚠️  flip/rot richiesti con --no-normalize: le feature sono "
              "gia' grezze, le simmetrie si applicano direttamente")
    print(f"[train] augmentation: node_drop={aug.node_drop} edge_drop={aug.edge_drop} "
          f"feat_mask={aug.feat_mask} (per-cella) geom_jitter={aug.geom_jitter} "
          f"flip={aug.flip_prob} rot={aug.rot_prob} | raw_skip={getattr(cfg, 'raw_skip', False)}")
    if aug.pair_mode == "asym_partial":
        print(f"[train] coppie ASIMMETRICHE (opzione D): intero (solo flip/rot) <-> "
              f"stanze rimosse f~U[{aug.partial_frac_min}, {aug.partial_frac_max}]; "
              "node_drop/edge_drop/feat_mask/jitter NON applicati | "
              f"lost_marker={aug.lost_marker}")

    # retrieval probe: expensive to build (per-axis ground truth from .mat), built once; only the encoder changes per epoch
    probe = None
    if cfg.probe_every > 0:
        probe = RetrievalProbe.build(
            valid_set,
            num_queries=cfg.probe_queries,
            gallery_size=cfg.probe_gallery,
            k=cfg.probe_k,
            seed=cfg.seed,
            batch_size=cfg.batch_size,
        )
        print(
            f"[train] sonda retrieval: gallery {len(probe)} (valid) | "
            f"{probe.num_queries} query | nDCG@{cfg.probe_k} | "
            f"criterio '{cfg.select_criterion}' | ogni {cfg.probe_every} epoche"
        )
    else:
        print("[train] sonda disattivata: selezione sulla val-loss (criterio legacy)")

    # robustness probe (selection_probe=partial) decides the epoch; the full probe is still built for logging and shadow selection
    rprobe, shadow = None, None
    if getattr(cfg, "selection_probe", "full") == "partial":
        rprobe = RobustnessProbe.build(
            RplanGraphDataset(transform=transform), transform,
            eval_config_path=cfg.probe_eval_config,
            num_queries=cfg.probe_partial_queries,
            seed=DEFAULT_PARTIAL_SEED,
            lost_marker=getattr(cfg, "lost_marker", False),
            batch_size=cfg.batch_size,
        )
        print(f"[train] sonda robustezza: {rprobe.info} | selezione su AUC self-recovery")
        if cfg.shadow_patience > 0:
            shadow = _ShadowSelection(cfg.shadow_patience, cfg.shadow_epochs)
            print(f"[train] selezione in ombra: '{cfg.select_criterion}' patience "
                  f"{cfg.shadow_patience}, tetto {cfg.shadow_epochs} -> {save_dir}_selfull")
    probe_history = []

    encoder = build_graph_encoder(cfg.encoder, **_encoder_kwargs(cfg)).to(device)
    optim = AdamW(encoder.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    gen = torch.Generator(device=device).manual_seed(cfg.seed)

    wb = _wandb_init(cfg, {"train": len(train_set), "valid": len(valid_set)})

    # selection score: higher is better (probe nDCG; negated val-loss in the legacy fallback)
    best_score, best_state, best_epoch, best_scores = -float("inf"), None, 0, None
    wait = 0
    epochs_run = 0
    for epoch in range(cfg.epochs):
        epochs_run = epoch + 1
        tr = _train_epoch(encoder, optim, train_loader, aug, geom_stats, cfg, gen, device)
        vl = _val_loss(encoder, valid_loader, aug, geom_stats, cfg, device)
        line = f"[train] epoch {epoch+1:02d}/{cfg.epochs} | loss {tr:.4f} | val {vl:.4f}"
        logs = {"epoch": epoch + 1, "train_loss": tr, "val_loss": vl}

        # None when the probe does not run: no decision, patience does not advance
        scores, score = None, None
        if probe is None:
            score = -vl
        elif (epoch + 1) % cfg.probe_every == 0:
            scores = probe.score(encoder, device)
            score = RetrievalProbe.selection_value(scores, cfg.select_criterion)
            line += f" | {RetrievalProbe.format_scores(scores)}"
            logs.update({f"probe/{k}": v for k, v in scores.items()})

        # in `partial` mode robustness is the selection score; the full probe is for log/shadow
        rscores = None
        if rprobe is not None and scores is not None:
            t0 = time.perf_counter()
            rscores = rprobe.score(encoder, device)
            score = rscores["auc"]
            line += f" | AUC {rscores['auc']:.4f} ({time.perf_counter() - t0:.1f}s)"
            logs.update({f"robust/{k}": v for k, v in rscores.items()})
        if rprobe is not None:
            probe_history.append({"epoch": epoch + 1, "train_loss": tr, "val_loss": vl,
                                  "probe": scores, "robust": rscores})

        print(line, flush=True)
        if wb is not None:
            wb.log(logs)
        if score is None:
            continue

        if shadow is not None:
            shadow.update(epoch + 1, RetrievalProbe.selection_value(scores, cfg.select_criterion),
                          scores, encoder)

        if score > best_score + 1e-4:
            best_score, best_state, wait = score, deepcopy(encoder.state_dict()), 0
            best_epoch, best_scores = epoch + 1, scores
        else:
            wait += 1
            if cfg.patience and wait >= cfg.patience:
                print(f"[train] early stop a epoch {epoch+1} (best {best_score:.4f} a epoch {best_epoch})")
                break

    criterion = f"ndcg@{cfg.probe_k}:{cfg.select_criterion}" if probe else "neg_val_loss"
    if rprobe is not None:
        criterion = f"self_rr_auc@{rprobe.info['max_rank']}:random"
    if best_state is None:
        # no selection eval ever ran (typically probe_every > epochs): final weights are saved but not selected
        print(f"[train] ⚠️  nessuna valutazione di selezione eseguita "
              f"(probe_every={cfg.probe_every} > epochs={cfg.epochs}?): "
              "salvo i pesi finali, non selezionati")
        best_score, best_epoch = float("nan"), cfg.epochs
    torch.save(best_state if best_state is not None else encoder.state_dict(),
               save_dir / "encoder.pt")
    print(f"[train] salvato {save_dir/'encoder.pt'} | best {criterion} {best_score:.4f} (epoch {best_epoch})")

    extra = None
    if rprobe is not None:
        # `probe_scores_at_best` stays the full probe (read by 03_train_gnn.sh); robustness has its own keys
        extra = {
            "selection_probe": "partial",
            "robust_probe_at_best": _robust_at(probe_history, best_epoch),
            "probe_partial": rprobe.info,
            "budget_binding": best_epoch > 0.9 * cfg.epochs,
            "epochs_run": epochs_run,
        }
        (save_dir / "probe_history.json").write_text(json.dumps(probe_history, indent=2))
    _save_summary(save_dir, cfg, criterion, best_score, best_epoch, best_scores, extra=extra)
    if shadow is not None:
        _save_shadow(save_dir, cfg, shadow, probe_history, rprobe.info)
    if wb is not None:
        wb.summary.update({"best_score": best_score, "best_epoch": best_epoch,
                           "selection_criterion": criterion})
        if best_scores is not None:
            wb.summary.update({f"best_{k}": v for k, v in best_scores.items()})
        wb.finish()


def _robust_at(probe_history: list[dict], epoch: int) -> dict | None:
    """Robustness scores logged at `epoch`, None if not measured."""
    return next((h["robust"] for h in probe_history if h["epoch"] == epoch), None)


def _save_shadow(save_dir: Path, cfg, shadow: "_ShadowSelection", probe_history, info) -> None:
    """Write the legacy-rule checkpoint to `<save_dir>_selfull` (encoder.pt, geom_stats.npz, training_summary.json)."""
    out = Path(f"{save_dir}_selfull")
    out.mkdir(parents=True, exist_ok=True)
    if not shadow.complete:
        print(f"[train] ⚠️  shadow INCOMPLETA: training fermato a epoch {shadow.last_epoch} "
              f"prima dello stop/tetto ({cfg.shadow_epochs}) della regola storica")
    torch.save(shadow.best_state, out / "encoder.pt")
    stats = save_dir / "geom_stats.npz"
    if stats.exists():
        shutil.copyfile(stats, out / "geom_stats.npz")
    shadow_cfg = argparse.Namespace(**{**vars(cfg), "variant": f"{cfg.variant}_selfull",
                                       "epochs": cfg.shadow_epochs})
    extra = {
        "selection_probe": "full",
        "shadow_of": cfg.variant,
        "shadow_patience": cfg.shadow_patience,
        "shadow_epochs": cfg.shadow_epochs,
        "robust_probe_at_best": _robust_at(probe_history, shadow.best_epoch),
        "probe_partial": info,
        "budget_binding": shadow.best_epoch > 0.9 * cfg.shadow_epochs,
        "epochs_run": shadow.last_epoch,
    }
    _save_summary(out, shadow_cfg, f"ndcg@{cfg.probe_k}:{cfg.select_criterion}",
                  shadow.best_score, shadow.best_epoch, shadow.best_scores, extra=extra)
    print(f"[train] shadow salvato {out/'encoder.pt'} | best ndcg@{cfg.probe_k}:"
          f"{cfg.select_criterion} {shadow.best_score:.4f} (epoch {shadow.best_epoch})")


def _save_summary(save_dir: Path, cfg, criterion, best_score, best_epoch, best_scores,
                  extra: dict | None = None) -> None:
    """Write `training_summary.json` next to the checkpoint; `extra` keys are appended."""
    summary = {
        "encoder": cfg.encoder,
        "variant": cfg.variant,
        "selection_criterion": criterion,
        "best_score": best_score,
        "best_epoch": best_epoch,
        "epochs_max": cfg.epochs,
        "probe_scores_at_best": best_scores,
        "pair_mode": getattr(cfg, "pair_mode", "symmetric"),
        "lost_marker": getattr(cfg, "lost_marker", False),
        "hyperparams": {
            "lr": cfg.lr, "weight_decay": cfg.weight_decay, "temperature": cfg.temperature,
            "batch_size": cfg.batch_size, "hidden_dim": cfg.hidden_dim, "out_dim": cfg.out_dim,
            "num_layers": cfg.num_layers, "pooling": cfg.pooling, "dropout": cfg.dropout,
            "node_drop": cfg.node_drop, "edge_drop": cfg.edge_drop, "feat_mask": cfg.feat_mask,
        },
    }
    if extra is not None:
        summary.update(extra)
    (save_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))


def _wandb_init(cfg, extra):
    """Init wandb if cfg.wandb, else None."""
    if not cfg.wandb:
        return None
    import wandb
    wandb.init(
        project=cfg.wandb_project,
        mode=cfg.wandb_mode,
        name=f"{cfg.encoder}_{cfg.variant}",
        config={**vars(cfg), **extra},
    )
    return wandb


def _default_save_dir(encoder: str, variant: str) -> str:
    """Save dir namespaced by encoder/variant."""
    return str(DEFAULT_ROOT.parent / encoder / variant)


def parse_args():
    p = argparse.ArgumentParser(description="Training InfoNCE di un encoder di grafo.")
    p.add_argument("--encoder", default="sage", choices=["gcn", "gat", "sage"])
    p.add_argument("--variant", default="base", help="etichetta per namespacing salvataggi")
    p.add_argument("--hidden-dim", type=int, default=128, dest="hidden_dim")
    p.add_argument("--out-dim", type=int, default=128, dest="out_dim")
    p.add_argument("--num-layers", type=int, default=2, dest="num_layers")
    p.add_argument("--pooling", default="add", choices=["add", "mean", "max", "mean_max"])
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--raw-skip", action="store_true", dest="raw_skip",
                   help="concatena l'add-pool delle feature grezze prima della proiezione "
                        "(composizione/geometria garantite, capacita' libera per la topologia)")
    # negative pair for ablation: YAML flags precede variant flags and argparse keeps the last
    p.add_argument("--no-raw-skip", action="store_false", dest="raw_skip")
    p.add_argument("--heads", type=int, default=4, help="solo GAT")
    p.add_argument("--attn-dropout", type=float, default=0.0, dest="attn_dropout", help="solo GAT")
    p.add_argument("--aggr", default="mean", help="solo SAGE (mean|max|add|lstm)")
    p.add_argument("--no-normalize", action="store_false", dest="normalize")
    p.add_argument("--keep-self-loops", action="store_false", dest="drop_self_loops")
    p.add_argument("--node-drop", type=float, default=0.1, dest="node_drop",
                   help="⚠️ invarianza FALSA per il full retrieval (cambia la composizione), "
                        "vera per il partial: dosare sapendolo")
    p.add_argument("--edge-drop", type=float, default=0.1, dest="edge_drop")
    p.add_argument("--feat-mask", type=float, default=0.1, dest="feat_mask",
                   help="probabilita' per CELLA (nodo, colonna), non piu' per colonna globale")
    p.add_argument("--geom-jitter", type=float, default=0.0, dest="geom_jitter",
                   help="dev.std del rumore sulle 6 colonne geometriche (unita' z-score)")
    p.add_argument("--flip-prob", type=float, default=0.0, dest="flip_prob",
                   help="probabilita' di riflettere la pianta (invarianza VERA su tutti gli assi)")
    p.add_argument("--rot-prob", type=float, default=0.0, dest="rot_prob",
                   help="probabilita' di ruotarla di 90/180/270 gradi (invarianza VERA)")
    # pair shape: symmetric, or asym_partial (whole graph vs graph with rooms removed)
    p.add_argument("--pair-mode", default="symmetric", dest="pair_mode",
                   choices=["symmetric", "asym_partial"])
    p.add_argument("--partial-frac-min", type=float, default=0.25, dest="partial_frac_min")
    p.add_argument("--partial-frac-max", type=float, default=0.75, dest="partial_frac_max")
    # lost-neighbours marker: in_dim 19 -> 20
    p.add_argument("--lost-marker", action="store_true", dest="lost_marker",
                   help="appende a ogni stanza il numero di vicini tolti (solo asym_partial); "
                        "cambia in_dim: l'eval deve ricevere lo stesso flag")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5, dest="weight_decay")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256, dest="batch_size")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--patience", type=int, default=8,
                   help="valutazioni-di-selezione senza miglioramento prima dello stop "
                        "(0 = off). Con --probe-every N conta le SONDE, non le epoche.")
    p.add_argument("--seed", type=int, default=42)
    # checkpoint selection (retrieval probe on valid)
    p.add_argument("--probe-every", type=int, default=1, dest="probe_every",
                   help="ogni quante epoche girare la sonda (0 = off -> selezione sulla val-loss)")
    p.add_argument("--probe-queries", type=int, default=500, dest="probe_queries")
    p.add_argument("--probe-gallery", type=int, default=5000, dest="probe_gallery",
                   help="grafi nella gallery della sonda (0 = tutto il valid)")
    p.add_argument("--probe-k", type=int, default=10, dest="probe_k",
                   help="profondita' dell'nDCG di selezione")
    p.add_argument("--select-criterion", default="mean", dest="select_criterion",
                   choices=list(SELECTION_CRITERIA),
                   help="asse su cui selezionare i pesi, o 'mean' per la media dei tre")
    # robustness selection: self-recovery AUC on valid queries disjoint from evaluation, shared gallery
    p.add_argument("--selection-probe", default="full", dest="selection_probe",
                   choices=["full", "partial"],
                   help="full = nDCG della sonda full (storico); partial = AUC del self-recovery")
    p.add_argument("--probe-partial-queries", type=int, default=2000, dest="probe_partial_queries")
    p.add_argument("--probe-eval-config", default="configs/graph_retrieval.yaml",
                   dest="probe_eval_config",
                   help="config di valutazione da cui ricostruire gallery e query escluse")
    p.add_argument("--shadow-patience", type=int, default=0, dest="shadow_patience",
                   help="patience della regola storica in ombra -> <save_dir>_selfull (0 = spenta)")
    p.add_argument("--shadow-epochs", type=int, default=150, dest="shadow_epochs",
                   help="tetto di epoche della regola storica in ombra")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save-dir", default=None, dest="save_dir")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default="cvcs-graph", dest="wandb_project")
    p.add_argument("--wandb-mode", default="online", dest="wandb_mode")
    args = p.parse_args()
    if args.lost_marker and args.pair_mode != "asym_partial":
        p.error("--lost-marker richiede --pair-mode asym_partial")
    if args.shadow_patience > 0 and args.selection_probe != "partial":
        p.error("--shadow-patience richiede --selection-probe partial")
    if args.selection_probe == "partial" and args.probe_every == 0:
        p.error("--selection-probe partial richiede la sonda (--probe-every > 0)")
    if args.shadow_patience > 0 and args.probe_every != 1:
        p.error("--shadow-patience richiede --probe-every 1 (la patience storica conta le epoche)")
    if args.save_dir is None:
        args.save_dir = _default_save_dir(args.encoder, args.variant)
    return args


if __name__ == "__main__":
    train(parse_args())
