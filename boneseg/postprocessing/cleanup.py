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


def _erode(binary: np.ndarray, px: int) -> np.ndarray:
    """Shave ``px`` pixels off the mask boundary (thin the predicted band).

    Thinning the prediction at inference (a) severs the thin bridge where two
    close bones' outline bands touch, so their centerlines split into two, and
    (b) brings the band closer to the ~6px annotation width. The centerline
    itself is width-invariant, so this only changes the vector where a merge is
    actually broken; elsewhere it just cleans up the mask. Too much erosion
    snaps genuinely thin bones, so this is a small, opt-in amount (0 = off).
    """
    if px <= 0:
        return binary
    from scipy import ndimage as ndi  # deferred: keep import cost off the hot path
    return ndi.binary_erosion(binary, iterations=int(px))


def threshold_and_clean(
    prob: np.ndarray, threshold: float, min_size: int, erode_px: int = 0
) -> np.ndarray:
    """Binarize a float probability map, optionally thin it, and drop tiny blobs.

    Returns a uint8 {0, 1} mask with the same shape as ``prob``.
    """
    binary = prob >= threshold
    binary = _erode(binary, erode_px)
    if min_size > 0:
        binary = _remove_small_objects(binary, min_size)
    return binary.astype(np.uint8)


def adaptive_threshold_and_clean(
    prob: np.ndarray,
    block_size: int,
    offset: float,
    floor: float,
    min_size: int,
    erode_px: int = 0,
) -> np.ndarray:
    """Binarize with a LOCAL (adaptive) threshold, then remove tiny blobs.

    Instead of one global cutoff, each pixel is kept when it exceeds the mean
    of its local ``block_size`` neighborhood (minus ``offset``) AND clears an
    absolute ``floor``. Because two touching bones have a slight probability
    dip in the seam between them, the local threshold carves that dip out —
    splitting outlines that a global 0.5 fuses into one thick band — while the
    ``floor`` keeps texture in low-probability background (soil, stone) from
    lighting up everywhere. Trade-off: more false positives on confusing
    backgrounds than the global threshold; easy to delete in the editor.

    Returns a uint8 {0, 1} mask with the same shape as ``prob``.
    """
    from skimage.filters import threshold_local  # deferred: heavy import

    prob = prob.astype(np.float32, copy=False)
    block = max(3, int(block_size))
    if block % 2 == 0:                 # skimage requires an odd window
        block += 1
    local_t = threshold_local(prob, block_size=block, offset=float(offset))
    binary = (prob > local_t) & (prob >= float(floor))
    binary = _erode(binary, erode_px)
    if min_size > 0:
        binary = _remove_small_objects(binary, min_size)
    return binary.astype(np.uint8)


def count_components(mask01: np.ndarray) -> int:
    """Number of 8-connected foreground components (for the info panel).

    Uses OpenCV's labeler: on a 65 MP grave mask skimage's took ~490 ms per
    edit, cv2 takes ~75 ms for the same answer.
    """
    if mask01.sum() == 0:
        return 0
    try:
        import cv2
        n, _ = cv2.connectedComponents(mask01.astype(np.uint8), connectivity=8)
        return int(n - 1)                      # label 0 is the background
    except Exception:  # pragma: no cover - opencv is a hard dependency
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


def carve_mask_by_centerlines(
    mask01: np.ndarray,
    removed_skel: np.ndarray,
    kept_skel: np.ndarray,
    max_radius_px: float = 20.0,
) -> np.ndarray:
    """Erase the mask band that belonged to DELETED centerlines only.

    Every foreground pixel is assigned to its NEAREST centerline (a Voronoi
    partition of the blob by its own lines); a pixel closer to a deleted line
    than to any surviving one — and within ``max_radius_px`` of it — is
    dropped.

    A dense grave predicts as ONE merged blob covering many bones, so the
    whole-component rule alone can never remove anything there: deleting a
    centerline left its mask behind, the mask view showed a bone the skeleton
    view no longer had, and the next rebuild derived the deleted line straight
    back out of it. Carving by nearest line removes exactly the erased bone's
    band and leaves its neighbours — and their outlines, DXF and pixel stats —
    untouched. The radius keeps far-away fragments (which may simply never have
    had a centerline) out of it.

    Returns a NEW array; the input is never mutated.
    """
    removed = np.asarray(removed_skel) > 0
    if not removed.any():
        return mask01
    from scipy import ndimage as ndi  # deferred: keep import cost off the hot path

    d_removed = ndi.distance_transform_edt(~removed)
    kept = np.asarray(kept_skel) > 0
    if kept.any():
        d_kept = ndi.distance_transform_edt(~kept)
    else:
        # Nothing survives: every band within reach of a deleted line goes.
        d_kept = np.full(mask01.shape, np.inf, dtype=np.float64)

    drop = (mask01 > 0) & (d_removed < d_kept) & (d_removed <= float(max_radius_px))
    out = mask01.copy()
    out[drop] = 0
    return out
