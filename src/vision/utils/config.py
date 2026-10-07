from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def load_vision_config(
    main_path: str = "configs/vision_retrieval.yaml",
    overrides: list[str] | None = None,
) -> DictConfig:
    """
    Load the main retrieval config and graft the selected encoder's kwargs into model.kwargs.

    Files: configs/vision_retrieval.yaml (model.name + pipeline params),
    configs/vision_models/<name>.yaml (encoder kwargs).
    overrides: dotlist, e.g. ["model.variant=gem"]; applied before and after the graft.
    """
    main_path = Path(main_path)
    if not main_path.exists():
        raise FileNotFoundError(f"Config principale non trovata: {main_path}")

    cfg = OmegaConf.load(main_path)

    override_cfg = (
        OmegaConf.from_dotlist(list(overrides)) if overrides else None
    )

    # overrides first, so `model.name=...` selects the right preset
    if override_cfg is not None:
        cfg = OmegaConf.merge(cfg, override_cfg)

    name       = cfg.model.name
    model_dir  = Path(cfg.model.config_dir)
    model_path = model_dir / f"{name}.yaml"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Preset del modello '{name}' non trovato: {model_path}. "
            f"Atteso un file {model_dir}/<name>.yaml con i kwargs dell'encoder."
        )

    model_cfg = OmegaConf.load(model_path)

    cfg.model.kwargs = model_cfg

    # again after the graft: `model.kwargs.*` overrides win over the preset
    if override_cfg is not None:
        cfg = OmegaConf.merge(cfg, override_cfg)

    return cfg


def transform_tag(cfg: DictConfig) -> str:
    """
    Tag of the active head/whitening, to namespace logs, per-query files and plots.

    E.g. "raw", "whiten", "whiten768", "head", "head+whiten-train". `whitening.fit_split=train`
    adds a `-train` suffix; transductive (`all`) has none, keeping existing file names.
    """
    parts = []
    head = cfg.get("head")
    if head is not None and head.get("enabled"):
        # checkpoint stem in the tag: head.pt -> "head", head_probe.pt -> "head-probe"
        stem = Path(head.get("file") or "head.pt").stem
        parts.append("head" if stem == "head" else stem.replace("_", "-"))
    if cfg.whitening.enabled:
        dim = cfg.whitening.get("dim")
        tag = f"whiten{dim}" if dim else "whiten"
        fit_split = cfg.whitening.get("fit_split") or "all"
        if fit_split != "all":
            tag = f"{tag}-{fit_split}"
        parts.append(tag)
    qh = cfg.get("query_head")
    if qh is not None and qh.get("enabled"):
        # query-only head: head_v2.pt -> "qhead-v2"
        parts.append("q" + Path(qh.get("file") or "head_v2.pt").stem.replace("_", "-"))
    return "+".join(parts) or "raw"
