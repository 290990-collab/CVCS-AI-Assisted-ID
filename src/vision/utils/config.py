# src/vision/utils/config.py

from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def load_vision_config(
    main_path: str = "configs/vision_retrieval.yaml",
    overrides: list[str] | None = None,
) -> DictConfig:
    """
    Carica la config principale del retrieval vision e vi innesta i kwargs
    dell'encoder selezionato --> unisce 2 YAML in 1 oggetto di configurazione (DictConfig)

    Layout dei file:
      configs/vision_retrieval.yaml      -> sceglie model.name + parametri pipeline
      configs/vision_models/<name>.yaml  -> kwargs specifici di quell'encoder

    Args:
        main_path: percorso del file di config principale.
        overrides: lista di override in stile dotlist (es. ["model.variant=gem",
                   "model.kwargs.image_size=384"]) applicati DOPO l'innesto del
                   preset. Servono per gli sweep del benchmark senza editare i YAML.

    Returns:
        OmegaConf DictConfig con i kwargs del modello caricati in model.kwargs.
    """
    main_path = Path(main_path)
    if not main_path.exists():
        raise FileNotFoundError(f"Config principale non trovata: {main_path}")

    cfg = OmegaConf.load(main_path)

    override_cfg = (
        OmegaConf.from_dotlist(list(overrides)) if overrides else None
    )

    # Applica gli override PRIMA di scegliere il preset: così un override CLI
    # come `model.name=siglip2` seleziona il preset giusto. Senza questo, il
    # preset veniva scelto leggendo il default del file (sempre dinov2) e ogni
    # encoder finiva per caricare i kwargs di DINOv2.
    if override_cfg is not None:
        cfg = OmegaConf.merge(cfg, override_cfg)

    # carica il preset dell'encoder selezionato (configs/vision_models/<name>.yaml)
    name       = cfg.model.name
    model_dir  = Path(cfg.model.config_dir)
    model_path = model_dir / f"{name}.yaml"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Preset del modello '{name}' non trovato: {model_path}. "
            f"Atteso un file {model_dir}/<name>.yaml con i kwargs dell'encoder."
        )

    model_cfg = OmegaConf.load(model_path)

    # innesta i kwargs dell'encoder nella config principale
    cfg.model.kwargs = model_cfg

    # Riapplica gli override DOPO l'innesto, così quelli su `model.kwargs.*`
    # (es. image_size, extraction_layer) vincono sul preset.
    if override_cfg is not None:
        cfg = OmegaConf.merge(cfg, override_cfg)

    return cfg


def transform_tag(cfg: DictConfig) -> str:
    """
    Etichetta del contributo attivo (head/whitening), usata per namespacing di
    log e visualizzazioni così le combinazioni non si sovrascrivono.
    Esempi: "raw", "whiten", "whiten768", "head", "head+whiten".
    """
    parts = []
    head = cfg.get("head")
    if head is not None and head.get("enabled"):
        parts.append("head")
    if cfg.whitening.enabled:
        dim = cfg.whitening.get("dim")
        parts.append(f"whiten{dim}" if dim else "whiten")
    return "+".join(parts) or "raw"
