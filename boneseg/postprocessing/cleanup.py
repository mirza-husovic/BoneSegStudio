"""Probability-map binarization and mask cleanup.

Identical to predict.py's ``postprocess()``: threshold the sigmoid map, then
drop connected components smaller than ``min_size`` pixels (speckle noise).
"""

from __future__ import annotations

import inspect

import numpy as np
from skimage.measure import label
from skimage.morphology import remove_small_objects

# scikit-image 0.26 renamed ``min_size`` (removes objects strictly smaller)
# to ``max_size`` (removes objects smaller-than-or-equal). Detect which the
# installed version exposes so we reproduce predict.py's exact behavior
# (``min_size=N`` drops components with < N pixels) on either version.
_USE_MAX_SIZE = "max_size" in inspect.signature(remove_small_objects).parameters


def _remove_small_objects(binary: np.ndarray, min_size: int) -> np.ndarray:
    """Drop components with fewer than ``min_size`` pixels (version-agnostic)."""
    if _USE_MAX_SIZE:
        # max_size removes objects with <= max_size px; min_size-1 reproduces
        # "strictly smaller than min_size".
        return remove_small_objects(binary, max_size=min_size - 1)
    return remove_small_objects(binary, min_size=min_size)


def threshold_and_clean(prob: np.ndarray, threshold: float, min_size: int) -> np.ndarray:
    """Binarize a float probability map and remove tiny blobs.

    Returns a uint8 {0, 1} mask with the same shape as ``prob``.
    """
    binary = prob >= threshold
    if min_size > 0:
        binary = _remove_small_objects(binary, min_size)
    return binary.astype(np.uint8)


def count_components(mask01: np.ndarray) -> int:
    """Number of 8-connected foreground components (for the info panel)."""
    if mask01.sum() == 0:
        return 0
    return int(label(mask01, connectivity=2).max())


def remove_component_at(mask01: np.ndarray, row: int, col: int) -> tuple[np.ndarray, bool]:
    """Delete the whole connected component under (row, col), if any.

    One-click equivalent of a GIS vertex tool's "select feature, delete" —
    operates on the 8-connected blob rather than individual vertices, which
    is enough to drop a stray false-positive component. Returns a NEW array
    (input is never mutated) and whether anything was actually removed.
    """
    if not (0 <= row < mask01.shape[0] and 0 <= col < mask01.shape[1]):
        return mask01, False
    labels = label(mask01, connectivity=2)
    lbl = labels[row, col]
    if lbl == 0:
        return mask01, False
    new_mask = mask01.copy()
    new_mask[labels == lbl] = 0
    return new_mask, True
