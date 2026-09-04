"""End-to-end editor regression: hand-drawn centerlines must not come back.

Drives the real FastAPI app in-process (no model: the flow starts from a blank
canvas) through the sequence that used to destroy a session's work —

    paint mask -> Apply edits -> delete a centerline -> Apply edits
    -> touch the mask again -> Apply edits

and asserts that the deleted line stays deleted, its mask goes with it, and
the untouched lines are still there afterwards.

Run:
    python tests/test_editor_flow.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from boneseg.webui.server import create_app  # noqa: E402

H, W = 200, 320
ROW_A, ROW_B = 90, 110       # two bones, 20 px apart


def _png(arr_rgba: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr_rgba, "RGBA").save(buf, "PNG")
    return buf.getvalue()


def _near(line: list, row: int, tol: float = 5.0) -> bool:
    """Whether a polyline runs along ``row`` (mean Y within ``tol``)."""
    return abs(sum(y for _, y in line) / len(line) - row) <= tol


def _painted(rows: list[tuple[int, int]], base: np.ndarray | None = None) -> bytes:
    """An edit-res mask PNG with a 9 px band on each requested row range."""
    rgba = base if base is not None else np.zeros((H, W, 4), dtype=np.uint8)
    for row, half in rows:
        rgba[row - half:row + half + 1, 40:280] = 255
    return _png(rgba)


def main() -> int:
    client = TestClient(create_app())

    # A plain photo, opened but never inferred — the editor starts blank.
    img = np.full((H, W, 3), 60, dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "PNG")
    r = client.post("/api/open", files={"file": ("flow.png", buf.getvalue(), "image/png")})
    r.raise_for_status()
    assert r.json()["edit"]["width"] == W, "edit resolution should match this small image"
    client.post("/api/blank_canvas").raise_for_status()
    print("[1/5] blank canvas ready")

    def mask_version() -> int:
        m = client.get("/api/image/mask.png")
        m.raise_for_status()
        return int(m.headers["X-Mask-Version"])

    def apply_mask(png: bytes) -> dict:
        v = mask_version()          # serving the mask arms the diff basis
        r = client.post("/api/apply_mask",
                        files={"file": ("mask.png", png, "image/png")},
                        data={"mask_version": str(v), "prune_branch_px": "0",
                              "min_skeleton_px": "0"})
        r.raise_for_status()
        return r.json()

    # 1. Paint two bones and apply -> the server derives their centerlines
    #    (a painted band also keeps its end spurs: they are hand-made too).
    apply_mask(_painted([(ROW_A, 4), (ROW_B, 4)]))
    vec = client.get("/api/vectors").json()
    on_a = [ln for ln in vec["polylines"] if _near(ln, ROW_A)]
    on_b = [ln for ln in vec["polylines"] if _near(ln, ROW_B)]
    assert on_a and on_b, f"expected centerlines on both bones, got {len(vec['polylines'])}"
    print(f"[2/5] painted 2 bones -> {len(on_a)} + {len(on_b)} centerlines")

    # 2. Delete the first bone's centerlines by hand (skeleton-view eraser).
    keep = on_b
    sum_ = client.post("/api/set_vectors", json={"polylines": keep}).json()
    assert sum_["result"]["vectors_locked"] is True, "hand edits must lock the vectors"
    assert sum_["result"]["n_centerlines"] == len(keep)
    print("[3/5] deleted one centerline; vectors locked")

    # The mask under it must be gone too — mask view has to agree with the
    # skeleton view (this is what used to show a 'ghost' bone).
    m = client.get("/api/image/mask.png")
    served = np.asarray(Image.open(io.BytesIO(m.content)).convert("RGBA"))[..., 3] > 10
    assert not served[ROW_A, 160], "mask under the deleted centerline must be carved away"
    assert served[ROW_B, 160], "the surviving bone's mask must stay"
    print("[4/5] mask carved under the deleted line, neighbour intact")

    # 3. Touch the mask again (paint one more bone) and apply. The deleted
    #    line must NOT come back and the hand-kept lines must be untouched.
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    rgba[..., :3] = 255
    rgba[..., 3] = np.where(served, 255, 0)
    rgba[26:35, 40:280] = 255                       # a third bone, painted by hand
    sum_ = apply_mask(_png(rgba))
    vec2 = client.get("/api/vectors").json()
    rows = sorted({round(sum(y for _, y in ln) / len(ln)) for ln in vec2["polylines"]})
    assert not any(_near(ln, ROW_A) for ln in vec2["polylines"]), (
        f"the deleted centerline came back after a mask edit (rows={rows})")
    for ln in keep:
        assert ln in vec2["polylines"], f"a hand-kept line was rewritten (rows={rows})"
    assert any(_near(ln, 30) for ln in vec2["polylines"]), (
        f"the newly painted bone got no centerline (rows={rows})")
    assert sum_["result"]["vectors_locked"] is True, "the lock must hold across applies"
    print("[5/5] mask edit kept the drawing (no resurrection)")

    print("\nEditor flow OK — hand-drawn centerlines survive mask edits.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
