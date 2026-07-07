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

from boneseg.config import DEFAULT_MODEL_PATH
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
        display_name="model367b3 — UNet EfficientNet-B3 (production)",
        family="smp-unet",
        weights_path=DEFAULT_MODEL_PATH,
        encoder="efficientnet-b3",
        description=(
            "Bone-outline UNet trained on UNET_DATASET_FINAL4 (367 pairs), "
            "patch 512 / stride 256, BCE+Dice, best epoch 66; test std Dice "
            "0.587 / ROI-d50 0.625 (no-TTA, 36 imgs), clDice@5 0.788."
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
