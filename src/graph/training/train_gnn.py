# src/graph/training/train_gnn.py

"""
Training self-supervised (InfoNCE) di un encoder di grafo (GCN/GAT/GraphSAGE).

Non esistono coppie di rilevanza etichettate, quindi il segnale e' contrastivo:
per ogni grafo si generano DUE viste aumentate (node/edge drop, feature mask =
controparte grafo del masking partial); il modello deve avvicinare le due viste
della stessa pianta (positivi) e allontanare le altre piante del batch (negativi).

Differenza chiave col ramo vision: qui la rete **si allena**, quindi ogni epoca
fa forward sui grafi (sono piccoli, 4-8 nodi: veloce). Nel vision invece la head
girava su vettori cachati col backbone frozen.

Disciplina train/valid: fit sullo split `train`, misure sullo split `valid` ->
**early stopping**, si salvano i pesi migliori in `<save_dir>/encoder.pt`. La
normalizzazione delle feature usa statistiche calcolate SOLO dal train (no
leakage), salvate in `<save_dir>/geom_stats.npz`.

Criterio di selezione: nDCG, non val-loss
-----------------------------------------
I pesi migliori si scelgono con una **sonda di retrieval** sul valid
(`RetrievalProbe`), non con la val-loss InfoNCE. Motivo: le due grandezze
divergono. Nel benchmark del 17 lug 2026 la classifica per val-loss era
l'inverso di quella per retrieval (GAT: loss migliore 2.08, nDCG peggiore; GCN:
loss peggiore 2.73, nDCG migliore), perche' InfoNCE allontana anche le piante
architettonicamente simili che finiscono nello stesso batch — proprio quelle che
la valutazione considera rilevanti. Selezionare sulla loss significava quindi
premiare il comportamento sbagliato.

La val-loss continua a essere calcolata e stampata (diagnostica utile: dice se
il contrastivo sta convergendo), ma non decide piu' nulla. Con `--probe-every 0`
si torna al vecchio criterio (utile solo per confronto).
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch_geometric.loader import DataLoader

from src.graph.evaluation.retrieval_probe import SELECTION_CRITERIA, RetrievalProbe
from src.graph.graph_builder import NODE_FEATURE_DIM, NUM_RELATION_TYPES
from src.graph.graph_dataset import DEFAULT_ROOT, RplanGraphDataset
from src.graph.models import build_graph_encoder
from src.graph.training.augment import AugmentParams, two_views
from src.graph.transforms import (
    build_node_transform,
    compute_geometry_stats,
    save_geometry_stats,
)


def info_nce(za: torch.Tensor, zp: torch.Tensor, temperature: float) -> torch.Tensor:
    """InfoNCE/NT-Xent simmetrica con negativi in-batch.

    `za`, `zp` sono gli embedding L2-normalizzati delle due viste (stessa pianta).
    Per la riga i il positivo e' zp[i]; i negativi sono le altre righe del batch.
    (Gemella della info_nce del ramo vision: opera su vettori gia' L2-norm.)
    """
    logits = (za @ zp.t()) / temperature
    labels = torch.arange(za.size(0), device=za.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def _encoder_kwargs(cfg) -> dict:
    """Raccoglie i kwargs del costruttore dell'encoder, inclusi quelli specifici."""
    kwargs = dict(
        in_dim=NODE_FEATURE_DIM,
        hidden_dim=cfg.hidden_dim,
        out_dim=cfg.out_dim,
        num_layers=cfg.num_layers,
        pooling=cfg.pooling,
        dropout=cfg.dropout,
        # getattr: i checkpoint/entrypoint piu' vecchi non hanno questo campo.
        raw_skip=getattr(cfg, "raw_skip", False),
    )
    # Parametri specifici per encoder (ignorati dagli altri).
    if cfg.encoder == "gat":
        kwargs.update(heads=cfg.heads, edge_dim=NUM_RELATION_TYPES, attn_dropout=cfg.attn_dropout)
    elif cfg.encoder == "sage":
        kwargs.update(aggr=cfg.aggr)
    return kwargs


def _train_epoch(encoder, optim, loader, aug, geom_stats, cfg, gen, device) -> float:
    """Una epoca di training: due viste per batch -> InfoNCE -> backprop."""
    encoder.train()
    total, seen = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        if batch.num_graphs < 2:                 # InfoNCE ha bisogno di >=2 (negativi)
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
    """Val-loss con augmentation DETERMINISTICA (generator riseminato ogni volta),
    cosi' e' confrontabile tra epoche. Solo diagnostica: la selezione dei pesi la
    fa la sonda di retrieval (vedi docstring del modulo)."""
    encoder.eval()
    gen = torch.Generator(device=device).manual_seed(cfg.seed)  # fisso -> deterministico
    total, seen = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        if batch.num_graphs < 2:
            continue
        view_a, view_b = two_views(batch, aug, gen, geom_stats)
        total += info_nce(encoder(view_a), encoder(view_b), cfg.temperature).item() * batch.num_graphs
        seen += batch.num_graphs
    return total / max(seen, 1)


def train(cfg) -> None:
    device = cfg.device
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- adjustment: statistiche SOLO dal train, poi transform condivisa ---
    stats = None
    if cfg.normalize:
        stats = compute_geometry_stats(RplanGraphDataset(split="train"))
        save_geometry_stats(stats, save_dir / "geom_stats.npz")
    transform = build_node_transform(
        normalize=cfg.normalize, drop_self_loops=cfg.drop_self_loops, stats=stats
    )

    # --- dataset train/valid (grafi grezzi in cache, transform al volo) ---
    train_set = RplanGraphDataset(split="train", transform=transform)
    valid_set = RplanGraphDataset(split="valid", transform=transform)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=cfg.batch_size, shuffle=False)
    print(f"[train] {cfg.encoder} | train={len(train_set)} valid={len(valid_set)} | device={device}")

    # --- augmentation: intensita' + statistiche per le simmetrie geometriche ---
    # Riflessione e rotazione sono definite sulle coordinate GREZZE in [0,1],
    # ma le feature qui arrivano z-scorate: servono mean/std per de-normalizzare
    # e ri-normalizzare (le colonne non sono intercambiabili, std(cy)/std(cx)=1.24).
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

    # --- sonda di retrieval: il criterio con cui si scelgono i pesi ---
    # Costruirla e' caro (ground truth per-asse dai .mat) ma si fa UNA volta:
    # gallery e rilevanza sono fisse, a ogni epoca cambia solo l'encoder.
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

    # --- encoder + optimizer ---
    encoder = build_graph_encoder(cfg.encoder, **_encoder_kwargs(cfg)).to(device)
    optim = AdamW(encoder.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    gen = torch.Generator(device=device).manual_seed(cfg.seed)

    wb = _wandb_init(cfg, {"train": len(train_set), "valid": len(valid_set)})

    # --- loop con early stopping sul valid ---
    # Il punteggio di selezione e' sempre "piu' alto = meglio": con la sonda e'
    # l'nDCG, nel fallback legacy e' la val-loss NEGATA (minimizzarla = massimizzare -loss).
    best_score, best_state, best_epoch, best_scores = -float("inf"), None, 0, None
    wait = 0
    for epoch in range(cfg.epochs):
        tr = _train_epoch(encoder, optim, train_loader, aug, geom_stats, cfg, gen, device)
        vl = _val_loss(encoder, valid_loader, aug, geom_stats, cfg, device)
        line = f"[train] epoch {epoch+1:02d}/{cfg.epochs} | loss {tr:.4f} | val {vl:.4f}"
        logs = {"epoch": epoch + 1, "train_loss": tr, "val_loss": vl}

        # Punteggio di selezione: None nelle epoche in cui la sonda non gira
        # (nessuna decisione presa, il contatore di pazienza non avanza).
        scores, score = None, None
        if probe is None:
            score = -vl
        elif (epoch + 1) % cfg.probe_every == 0:
            scores = probe.score(encoder, device)
            score = RetrievalProbe.selection_value(scores, cfg.select_criterion)
            line += f" | {RetrievalProbe.format_scores(scores)}"
            logs.update({f"probe/{k}": v for k, v in scores.items()})

        print(line, flush=True)
        if wb is not None:
            wb.log(logs)
        if score is None:
            continue

        if score > best_score + 1e-4:
            best_score, best_state, wait = score, deepcopy(encoder.state_dict()), 0
            best_epoch, best_scores = epoch + 1, scores
        else:
            wait += 1
            if cfg.patience and wait >= cfg.patience:
                print(f"[train] early stop a epoch {epoch+1} (best {best_score:.4f} a epoch {best_epoch})")
                break

    criterion = f"ndcg@{cfg.probe_k}:{cfg.select_criterion}" if probe else "neg_val_loss"
    if best_state is None:
        # Nessuna valutazione di selezione e' mai avvenuta (tipico: probe_every
        # > epochs). Si salvano i pesi finali, ma NON sono stati scelti: dirlo,
        # altrimenti il checkpoint sembra selezionato e non lo e'.
        print(f"[train] ⚠️  nessuna valutazione di selezione eseguita "
              f"(probe_every={cfg.probe_every} > epochs={cfg.epochs}?): "
              "salvo i pesi finali, non selezionati")
        best_score, best_epoch = float("nan"), cfg.epochs
    torch.save(best_state if best_state is not None else encoder.state_dict(),
               save_dir / "encoder.pt")
    print(f"[train] salvato {save_dir/'encoder.pt'} | best {criterion} {best_score:.4f} (epoch {best_epoch})")

    _save_summary(save_dir, cfg, criterion, best_score, best_epoch, best_scores)
    if wb is not None:
        wb.summary.update({"best_score": best_score, "best_epoch": best_epoch,
                           "selection_criterion": criterion})
        if best_scores is not None:
            wb.summary.update({f"best_{k}": v for k, v in best_scores.items()})
        wb.finish()


def _save_summary(save_dir: Path, cfg, criterion, best_score, best_epoch, best_scores) -> None:
    """Scrive `training_summary.json` accanto al checkpoint.

    Serve allo sweep OFAT: confrontare due configurazioni = leggere due file,
    senza ripescare i numeri dai log di slurm.
    """
    summary = {
        "encoder": cfg.encoder,
        "variant": cfg.variant,
        "selection_criterion": criterion,
        "best_score": best_score,
        "best_epoch": best_epoch,
        "epochs_max": cfg.epochs,
        "probe_scores_at_best": best_scores,
        "hyperparams": {
            "lr": cfg.lr, "weight_decay": cfg.weight_decay, "temperature": cfg.temperature,
            "batch_size": cfg.batch_size, "hidden_dim": cfg.hidden_dim, "out_dim": cfg.out_dim,
            "num_layers": cfg.num_layers, "pooling": cfg.pooling, "dropout": cfg.dropout,
            "node_drop": cfg.node_drop, "edge_drop": cfg.edge_drop, "feat_mask": cfg.feat_mask,
        },
    }
    (save_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))


def _wandb_init(cfg, extra):
    """Inizializza wandb se cfg.wandb, altrimenti None (nessuna dipendenza)."""
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
    """Cartella di salvataggio namespaced per encoder/variante (simmetrica al vision)."""
    return str(DEFAULT_ROOT.parent / encoder / variant)


def parse_args():
    p = argparse.ArgumentParser(description="Training InfoNCE di un encoder di grafo.")
    # modello
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
    # Coppia negativa: serve all'ablation, perche' i flag del YAML vengono emessi
    # prima di quelli della variante e argparse fa vincere l'ultimo della riga.
    p.add_argument("--no-raw-skip", action="store_false", dest="raw_skip")
    p.add_argument("--heads", type=int, default=4, help="solo GAT")
    p.add_argument("--attn-dropout", type=float, default=0.0, dest="attn_dropout", help="solo GAT")
    p.add_argument("--aggr", default="mean", help="solo SAGE (mean|max|add|lstm)")
    # adjustment
    p.add_argument("--no-normalize", action="store_false", dest="normalize")
    p.add_argument("--keep-self-loops", action="store_false", dest="drop_self_loops")
    # augmentation
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
    # ottimizzazione
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5, dest="weight_decay")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256, dest="batch_size")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--patience", type=int, default=8,
                   help="valutazioni-di-selezione senza miglioramento prima dello stop "
                        "(0 = off). Con --probe-every N conta le SONDE, non le epoche.")
    p.add_argument("--seed", type=int, default=42)
    # selezione del checkpoint (sonda di retrieval sul valid)
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
    # infra
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save-dir", default=None, dest="save_dir")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default="cvcs-graph", dest="wandb_project")
    p.add_argument("--wandb-mode", default="online", dest="wandb_mode")
    args = p.parse_args()
    if args.save_dir is None:
        args.save_dir = _default_save_dir(args.encoder, args.variant)
    return args


if __name__ == "__main__":
    train(parse_args())
