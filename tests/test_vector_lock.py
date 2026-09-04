"""Hand-drawn centerlines must survive mask edits — and take their mask with them.

Covers the two halves of the "my edits come back" bug:

  1. deleting a centerline inside a MERGED blob must carve away that bone's
     band only (``carve_mask_by_centerlines``), so the mask view agrees with
     the skeleton view;
  2. a later mask edit must not re-derive the skeleton from the mask
     (``BonePipeline.apply_manual_mask_locked``), which used to throw the whole
     hand drawing away and resurrect every deleted line.

Run:
    python tests/test_vector_lock.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from boneseg.config import PostprocessSettings  # noqa: E402
from boneseg.pipeline import BonePipeline, PipelineResult  # noqa: E402
from boneseg.postprocessing import (  # noqa: E402
    carve_mask_by_centerlines,
    rasterize_polylines,
)

H, W = 120, 200
ROW_A, ROW_B = 56, 64          # two bones 8 px apart -> one merged band


def _merged_mask() -> np.ndarray:
    """Two horizontal bones whose 7 px bands touch: one connected component."""
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[ROW_A - 3:ROW_A + 4, 20:180] = 1
    mask[ROW_B - 3:ROW_B + 4, 20:180] = 1
    return mask


def _line(row: int) -> list[tuple[float, float]]:
    return [(float(x), float(row)) for x in range(20, 180)]


def test_carve_splits_a_merged_blob() -> None:
    mask = _merged_mask()
    kept = rasterize_polylines([_line(ROW_B)], mask.shape)
    removed = rasterize_polylines([_line(ROW_A)], mask.shape)

    out = carve_mask_by_centerlines(mask, removed_skel=removed, kept_skel=kept > 0)

    assert out[ROW_A, 100] == 0, "deleted bone's band should be gone"
    assert out[ROW_B, 100] == 1, "surviving bone's band must stay"
    assert out[ROW_B - 2:ROW_B + 3, 20:180].all(), "surviving band must stay whole"
    assert mask[ROW_A, 100] == 1, "input must not be mutated"
    print(f"  carve: fg {int(mask.sum())} -> {int(out.sum())} px")


def _result(mask: np.ndarray, lines: list) -> PipelineResult:
    return PipelineResult(
        source_path=Path("synthetic.png"),
        image=np.zeros((H, W, 3), dtype=np.uint8),
        georef=None,
        prob=np.zeros((H, W), dtype=np.float32),
        mask=mask,
        skeleton=rasterize_polylines(lines, mask.shape),
        polylines_px=lines,
        polylines_out=[[(x, -y) for x, y in ln] for ln in lines],
        rings_out=[],
        n_components=1,
        fg_pixels=int(mask.sum()),
        inference_seconds=0.0,
        postprocess_seconds=0.0,
    )


def test_locked_apply_keeps_the_drawing() -> None:
    pipeline = BonePipeline.__new__(BonePipeline)   # no model needed here
    pp = PostprocessSettings()

    mask = _merged_mask()
    lines = [_line(ROW_A), _line(ROW_B)]
    result = _result(mask, lines)

    # The user erases the mask over bone A and paints a new blob elsewhere.
    mask_rem = np.zeros((H, W), dtype=bool)
    mask_rem[ROW_A - 3:ROW_A + 4, 20:180] = True
    mask_add = np.zeros((H, W), dtype=bool)
    mask_add[20:27, 40:160] = True
    edited = mask.copy()
    edited[mask_rem] = 0
    edited[mask_add] = 1

    out = pipeline.apply_manual_mask_locked(
        result, edited, pp, mask_add=mask_add, mask_rem=mask_rem)

    kept = [ln for ln in out.polylines_px if abs(ln[0][1] - ROW_B) < 1]
    assert len(kept) == 1, "the untouched bone must keep exactly its own line"
    assert kept[0] == lines[1], "an untouched line must survive byte-for-byte"
    assert not any(abs(ln[0][1] - ROW_A) < 1 for ln in out.polylines_px), \
        "the line over erased mask must be dropped"
    assert any(min(y for _, y in ln) < 40 for ln in out.polylines_px), \
        "the newly painted region must contribute a line"
    assert out.stats["edited"] == "vector"
    print(f"  locked apply: {len(lines)} lines in -> {len(out.polylines_px)} out")


def main() -> int:
    for fn in (test_carve_splits_a_merged_blob, test_locked_apply_keeps_the_drawing):
        print(f"[ ] {fn.__name__}")
        fn()
        print(f"[x] {fn.__name__} passed")
    print("\nAll vector-lock tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
