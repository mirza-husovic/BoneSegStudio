"""SAM2 promptable segmentation — point/box prompts for annotating features
the bone-UNet was never trained for (walls, stones, cut lines, any feature).

Deliberately dumb: no caching policy lives here (the caller — ``Studio`` in
``webui/server.py`` — decides when the encoded image is stale, since it
already tracks image identity for the mask-editing pipeline). This class
only knows "load the model" and "encode this array" / "predict from these
prompts", exactly mirroring the point-in/mask-out shape of the official
SAM2 API so a config/checkpoint swap (e.g. to sam2.1_hiera_large if the
base_plus boundary quality ever falls short) is a one-line change.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from boneseg.logging_setup import get_logger

logger = get_logger(__name__)

try:
    import torch

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    HAS_SAM2 = True
except Exception:  # pragma: no cover - optional dependency
    HAS_SAM2 = False


class SamEngine:
    """Lazy-loaded SAM2 image predictor. One encoded image at a time."""

    def __init__(self, checkpoint_path: Path, config_name: str):
        self.checkpoint_path = checkpoint_path
        self.config_name = config_name
        self._predictor = None
        self._device: str | None = None

    @property
    def loaded(self) -> bool:
        return self._predictor is not None

    def _ensure_loaded(self) -> None:
        if self._predictor is not None:
            return
        if not HAS_SAM2:
            raise RuntimeError(
                "sam2 is not installed (pip install sam2) — promptable "
                "segmentation is unavailable.")
        if not self.checkpoint_path.is_file():
            raise RuntimeError(
                f"SAM2 checkpoint not found: {self.checkpoint_path}")
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        model = build_sam2(self.config_name, str(self.checkpoint_path),
                            device=self._device)
        self._predictor = SAM2ImagePredictor(model)
        logger.info("SAM2 loaded on %s (%s)", self._device, self.checkpoint_path.name)

    def set_image(self, rgb: np.ndarray) -> None:
        """Encode an image (the expensive step, ~1s) — call once per photo,
        not once per prompt."""
        self._ensure_loaded()
        with torch.inference_mode(), torch.autocast(self._device, dtype=torch.bfloat16):
            self._predictor.set_image(rgb)

    def predict(
        self,
        points: np.ndarray | None = None,
        labels: np.ndarray | None = None,
        box: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float]:
        """Return (mask [bool, encoded-image shape], confidence score) for
        the best of SAM2's 3 candidate masks."""
        self._ensure_loaded()
        with torch.inference_mode(), torch.autocast(self._device, dtype=torch.bfloat16):
            masks, scores, _ = self._predictor.predict(
                point_coords=points, point_labels=labels, box=box,
                multimask_output=True)
        best = int(np.argmax(scores))
        return masks[best].astype(bool), float(scores[best])
