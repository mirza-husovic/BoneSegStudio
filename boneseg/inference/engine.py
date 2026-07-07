"""Sliding-window inference engine.

``predict_full`` is a line-for-line port of the validated function in
``model330b3/predict.py``: reflect-padding so windows cover the edges,
overlap-averaging of sigmoid probabilities, optional 4-way flip TTA and
mixed precision on CUDA. Do not "improve" this function without re-running
the test-set evaluation — its behavior matches the training-time eval.

:class:`InferenceEngine` wraps the function with lazy model loading and
device management so the UI can create it once and reuse it across images.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
import torch

from boneseg.config import InferenceSettings
from boneseg.inference.device import DeviceInfo, detect_device
from boneseg.logging_setup import get_logger
from boneseg.models import ModelSpec, get_model_spec, load_model

logger = get_logger(__name__)

# 4-way TTA: identity + h-flip + v-flip + 180° rotation.
# Each entry is (forward-op, inverse-op) on (B,C,H,W) tensors. The same
# torch.flip dims work on both input and output because the spatial axes are
# in identical positions for the (B,3,H,W) input and the (B,1,H,W) output.
_TensorOp = Callable[[torch.Tensor], torch.Tensor]
TTA_TRANSFORMS: list[tuple[_TensorOp, _TensorOp]] = [
    (lambda x: x,                          lambda y: y),
    (lambda x: torch.flip(x, dims=[3]),    lambda y: torch.flip(y, dims=[3])),   # h-flip
    (lambda x: torch.flip(x, dims=[2]),    lambda y: torch.flip(y, dims=[2])),   # v-flip
    (lambda x: torch.flip(x, dims=[2, 3]), lambda y: torch.flip(y, dims=[2, 3])),  # 180°
]


@torch.no_grad()
def predict_full(
    model: torch.nn.Module,
    img_rgb: np.ndarray,
    patch: int,
    stride: int,
    device: torch.device,
    use_amp: bool,
    tta: bool = False,
    batch_size: int = 8,
    progress_cb: Callable[[float], None] | None = None,
) -> np.ndarray:
    """Predict a full-resolution probability map for an RGB uint8 image.

    Tiles are run through the network in batches of ``batch_size``. Because
    the model is in eval mode (BatchNorm uses running stats), every tile's
    output is independent of its batch companions, so the result is
    numerically equivalent to the original one-tile-at-a-time loop — the
    batching only amortizes per-call GPU launch overhead (large speedup on
    many-tile orthophotos).

    Parameters
    ----------
    progress_cb
        Optional callback receiving completion in [0, 1] — used by the UI
        progress bar; the algorithm is unchanged when it is None. Raising
        from the callback aborts the prediction (used for cancellation).

    Returns
    -------
    np.ndarray
        float32 probability map with the same H×W as the input.
    """
    H, W = img_rgb.shape[:2]
    pad_h = max(0, patch - H)
    pad_w = max(0, patch - W)
    # also pad so (H+pad - patch) is a multiple of stride -> windows cover edges
    if (H + pad_h - patch) % stride:
        pad_h += stride - ((H + pad_h - patch) % stride)
    if (W + pad_w - patch) % stride:
        pad_w += stride - ((W + pad_w - patch) % stride)
    img_pad = np.pad(img_rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    Hp, Wp = img_pad.shape[:2]

    psum = np.zeros((Hp, Wp), dtype=np.float32)
    cnt = np.zeros((Hp, Wp), dtype=np.float32)

    amp_on = use_amp and device.type == "cuda"
    transforms = TTA_TRANSFORMS if tta else [TTA_TRANSFORMS[0]]
    n_tta = len(transforms)
    batch_size = max(1, int(batch_size))

    positions = [
        (y, x)
        for y in range(0, Hp - patch + 1, stride)
        for x in range(0, Wp - patch + 1, stride)
    ]
    n_tiles = len(positions)
    done = 0

    for start in range(0, n_tiles, batch_size):
        chunk = positions[start:start + batch_size]
        tiles = np.stack([img_pad[y:y + patch, x:x + patch] for y, x in chunk])
        t = (
            torch.from_numpy(tiles)
            .permute(0, 3, 1, 2)
            .float()
            .to(device)
            / 255.0
        )
        accum: torch.Tensor | None = None
        for fwd, inv in transforms:
            t_in = fwd(t)
            if amp_on:
                with torch.amp.autocast("cuda"):
                    logits = model(t_in)
            else:
                logits = model(t_in)
            prob_t = torch.sigmoid(inv(logits))
            accum = prob_t if accum is None else accum + prob_t
        probs = (accum / n_tta)[:, 0].float().cpu().numpy()
        for (y, x), prob in zip(chunk, probs):
            psum[y:y + patch, x:x + patch] += prob
            cnt[y:y + patch, x:x + patch] += 1.0

        done += len(chunk)
        if progress_cb is not None:
            progress_cb(done / n_tiles)

    prob_full = psum / np.maximum(cnt, 1e-6)
    return prob_full[:H, :W]


class InferenceEngine:
    """Holds one loaded model on the detected device.

    The model is loaded lazily on first use and cached; switching to another
    registered model reloads only when the key actually changes.
    """

    def __init__(self, model_key: str) -> None:
        self.device_info: DeviceInfo = detect_device()
        self._model_key = model_key
        self._model: torch.nn.Module | None = None

    # -- properties ---------------------------------------------------------
    @property
    def spec(self) -> ModelSpec:
        return get_model_spec(self._model_key)

    @property
    def device(self) -> torch.device:
        return self.device_info.device

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # -- lifecycle ----------------------------------------------------------
    def load(self) -> None:
        """Load weights if not already resident."""
        if self._model is None:
            t0 = time.time()
            self._model = load_model(self.spec, self.device)
            logger.info("Model ready in %.1fs", time.time() - t0)

    def set_model(self, model_key: str) -> None:
        """Switch to another registered model (reload on next predict)."""
        if model_key != self._model_key:
            logger.info("Switching model: %s -> %s", self._model_key, model_key)
            self._model_key = model_key
            self._model = None

    # -- inference ----------------------------------------------------------
    def predict(
        self,
        img_rgb: np.ndarray,
        settings: InferenceSettings,
        progress_cb: Callable[[float], None] | None = None,
    ) -> np.ndarray:
        """Run sliding-window inference; returns float32 probability map."""
        self.load()
        assert self._model is not None
        return predict_full(
            self._model,
            img_rgb,
            patch=settings.patch,
            stride=settings.stride,
            device=self.device,
            use_amp=settings.use_amp,
            tta=settings.use_tta,
            batch_size=settings.batch_size,
            progress_cb=progress_cb,
        )
