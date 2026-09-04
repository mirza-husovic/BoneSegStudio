"""Model registry.

Every model the application can serve is described by a :class:`ModelSpec`
and registered in :data:`MODEL_REGISTRY`. The UI populates its model
dropdown from the registry and the pipeline never hard-codes an
architecture, so adding a second UNet, a YOLO detector or SAM2 later only
requires:

  1. a new ``ModelSpec`` entry (with a new ``family``), and
  2. a builder function registered in ``_BUILDERS`` for that family.

Nothing else in the codebase has to change.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

from boneseg.config import DEFAULT_MODEL_PATH, PROJECT_ROOT
from boneseg.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    """Static description of a registered model."""

    key: str                    # registry id, used in config / UI values
    display_name: str           # human-readable name for the UI
    family: str                 # builder id: "smp-unet" (future: "yolo", "sam2")
    weights_path: Path
    encoder: str = "efficientnet-b3"
    in_channels: int = 3
    classes: int = 1
    patch: int = 512            # native training patch size
    description: str = ""


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "model367b3": ModelSpec(
        key="model367b3",
        display_name="model367b3 — UNet EfficientNet-B3 (legacy, FINAL4 only)",
        family="smp-unet",
        weights_path=DEFAULT_MODEL_PATH,
        encoder="efficientnet-b3",
        description=(
            "Bone-outline UNet trained on UNET_DATASET_FINAL4 (367 pairs), "
            "patch 512 / stride 256, BCE+Dice, best epoch 66; test std Dice "
            "0.587 / ROI-d50 0.625 (no-TTA, 36 imgs), clDice@5 0.788."
        ),
    ),
    "cropmodel": ModelSpec(
        key="cropmodel",
        display_name="cropmodel — UNet EffNet-B3 (crop experiment, epoch 60)",
        family="smp-unet",
        weights_path=PROJECT_ROOT / "models" / "cropmodel" / "best_bone_model.pth",
        encoder="efficientnet-b3",
        description=(
            "Isti recept kao model367b3, treniran na UNET_DATASET_CROP "
            "(393 izrezanih grobova: 367 FINAL4 + 26 Jana). NAPOMENA: treniran "
            "SAMO na izrezima fokusiranog groba -> najbolje radi na uskom kadru "
            "oko jednog groba; na punom kadru s vise susjednih kostura moze "
            "odstupati (nije vidio takve prizore u treningu). Eval (crop->paste-back, "
            "GT-box): final4 std Dice 0.621 / ROI-d50 0.633."
        ),
    ),
    "gapmodel": ModelSpec(
        key="gapmodel",
        display_name="gapmodel — UNet EffNet-B3 (gap-weight + clDice loss)",
        family="smp-unet",
        weights_path=PROJECT_ROOT / "models" / "gapmodel" / "best_bone_model.pth",
        encoder="efficientnet-b3",
        description=(
            "Isti UNet/dataset kao cropmodel (UNET_DATASET_CROP), ali treniran s "
            "kombiniranim lossom: 0.5*Dice + 0.5*weightedBCE(gap-mape) + 0.5*clDice. "
            "Cilj: RAZDVAJANJE bliskih kostiju (jaci BCE u uskim prorezima) + "
            "ZATVARANJE kontura (clDice topoloski). best epoha 28 / 44 epoha. "
            "Test std Dice 0.608 / ROI-d50 0.620 (blago nizi std od cropa ~0.62 -- "
            "ocekivan trade jer se mijenjao cilj; prava presuda je VIZUALNA: "
            "razdvaja li kraljeske i zatvara li konture)."
        ),
    ),
    "colab_final4jana": ModelSpec(
        key="colab_final4jana",
        display_name="colab — UNet EffNet-B3 (FINAL4+Jana, full-frame) [PRODUCTION]",
        family="smp-unet",
        weights_path=PROJECT_ROOT / "models" / "colab_final4jana" / "best_bone_model.pth",
        encoder="efficientnet-b3",
        description=(
            "PRODUKCIJSKI model. Isti recept kao model367b3, ali treniran na "
            "FINAL4 + Jana (pun kadar, bez cropa). Radi ispravno na punom kadru i "
            "generalizira na Janu koju model367b3 nikad nije vidio. "
            "Eval (pun kadar, TTA): final4 std Dice 0.599 / ROI-d50 0.634; "
            "jana std Dice 0.773; svih 40 std 0.618 / ROI-d50 0.650 (najbolji "
            "ukupni ROI-d50 od svih modela)."
        ),
    ),
    "resnet34": ModelSpec(
        key="resnet34",
        display_name="resnet34 — UNet ResNet34 (FINAL4, backbone eksperiment)",
        family="smp-unet",
        # Weights live in the training folder on the Desktop (a symlink under
        # models/ would need admin/dev-mode on Windows), so point at the
        # absolute path directly — same style as TRAINING_STAGING_DIR.
        weights_path=Path.home() / "Desktop" / "modeli" / "Resnet model" / "best_bone_model.pth",
        encoder="resnet34",
        description=(
            "Zamjena enkodera: ResNet34 umjesto EfficientNet-B3, isti recept, "
            "treniran SAMO na FINAL4 (bez Jane). best epoha 63. "
            "Eval (final4, TTA): std Dice 0.593 / ROI-d50 0.633 — prakticki "
            "izjednacen s model367b3 (0.587 / 0.625) i colabom na final4; backbone "
            "ne mice strop. Dodan za usporedbu/vizualnu presudu, nije produkcijski."
        ),
    ),
}


def get_model_spec(key: str) -> ModelSpec:
    """Look up a spec by key, with a helpful error for unknown keys."""
    try:
        return MODEL_REGISTRY[key]
    except KeyError:
        known = ", ".join(sorted(MODEL_REGISTRY))
        raise KeyError(f"Unknown model '{key}'. Registered models: {known}")


# ---------------------------------------------------------------------------
# Builders (one per model family)
# ---------------------------------------------------------------------------
def _build_smp_unet(spec: ModelSpec, device: torch.device) -> torch.nn.Module:
    """Reproduce predict.py's load_model(): smp.Unet + strict state_dict."""
    import segmentation_models_pytorch as smp  # deferred: heavy import

    model = smp.Unet(
        encoder_name=spec.encoder,
        encoder_weights=None,       # weights come from the checkpoint
        in_channels=spec.in_channels,
        classes=spec.classes,
    )
    # The checkpoint is a plain state_dict, so weights_only=True is safe and
    # avoids the arbitrary-code-execution risk of the pickle default.
    state = torch.load(str(spec.weights_path), map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model


_BUILDERS: dict[str, Callable[[ModelSpec, torch.device], torch.nn.Module]] = {
    "smp-unet": _build_smp_unet,
    # Future: "yolo": _build_yolo, "sam2": _build_sam2, ...
}


def load_model(spec: ModelSpec, device: torch.device) -> torch.nn.Module:
    """Build the architecture, load weights, move to device, set eval mode.

    Raises
    ------
    FileNotFoundError
        If the weights file is missing (surfaced to the UI as a clear message).
    """
    if not spec.weights_path.exists():
        raise FileNotFoundError(
            f"Model weights not found: {spec.weights_path}\n"
            f"Check the path in boneseg/config.py (DEFAULT_MODEL_PATH)."
        )
    builder = _BUILDERS.get(spec.family)
    if builder is None:
        raise ValueError(f"No builder registered for model family '{spec.family}'")

    logger.info("Loading model '%s' from %s on %s", spec.key, spec.weights_path, device)
    model = builder(spec, device)
    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model '%s' loaded (%.2fM parameters)", spec.key, n_params / 1e6)
    return model
