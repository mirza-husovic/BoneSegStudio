"""The full-resolution edit window: a 1 px stroke must stay 1 px.

The overview canvas is capped at MAX_EDIT_DIM, so on a big photo one canvas
pixel covers several real ones and a painted stroke grows on the way in (NEAREST
upscale) and again on the way back (INTER_AREA + "any covered pixel survives").
/api/edit_view hands the browser a REGION instead, at scale 1.0, where painting
is pixel-exact — and where a 2 px seam between two bones can actually be hit.

Run:
    python tests/test_fullres_window.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from boneseg.config import MAX_EDIT_DIM  # noqa: E402
from boneseg.webui.server import create_app  # noqa: E402

W, H = 5000, 3000                      # bigger than MAX_EDIT_DIM -> overview shrinks
RECT = (1200, 900, 800, 600)           # the window we edit in


def _png(rgba: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, "PNG")
    return buf.getvalue()


def main() -> int:
    app = create_app()
    client = TestClient(app)
    state = app.state.studio

    img = np.full((H, W, 3), 60, dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "PNG")
    r = client.post("/api/open", files={"file": ("big.png", buf.getvalue(), "image/png")})
    r.raise_for_status()
    overview = r.json()["edit"]
    assert overview["width"] == MAX_EDIT_DIM, "the overview should fill the cap"
    assert overview["scale"] < 1.0
    print(f"[1/4] overview canvas {overview['width']}x{overview['height']} "
          f"(scale {overview['scale']:.3f}: 1 canvas px = "
          f"{1/overview['scale']:.2f} real px)")

    client.post("/api/blank_canvas").raise_for_status()

    x, y, w, h = RECT
    view = client.post("/api/edit_view", json={"x": x, "y": y, "w": w, "h": h}).json()["edit"]
    assert (view["width"], view["height"]) == (w, h), view
    assert view["scale"] == 1.0, "a window this small must not be scaled at all"
    assert (view["x"], view["y"]) == (x, y)
    print(f"[2/4] edit window {w}x{h} at ({x}, {y}), scale {view['scale']}")

    # Paint a 1 px line inside the window and apply it.
    m = client.get("/api/image/mask.png")
    m.raise_for_status()
    served = np.asarray(Image.open(io.BytesIO(m.content)).convert("RGBA"))
    assert served.shape[:2] == (h, w), served.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., :3] = 255
    rgba[300, 100:700, 3] = 255                      # exactly one row
    r = client.post("/api/apply_mask",
                    files={"file": ("mask.png", _png(rgba), "image/png")},
                    data={"mask_version": m.headers["X-Mask-Version"],
                          "prune_branch_px": "0", "min_skeleton_px": "0"})
    r.raise_for_status()

    mask = state.result.mask > 0
    col = mask[:, x + 400]
    rows = np.flatnonzero(col)
    assert rows.size == 1, f"a 1 px stroke became {rows.size} px thick in the mask"
    assert rows[0] == y + 300, f"stroke landed at row {rows[0]}, expected {y + 300}"
    ys, xs = np.nonzero(mask)
    assert xs.min() == x + 100 and xs.max() == x + 699, (xs.min(), xs.max())
    print(f"[3/4] 1 px stroke stored as {rows.size} px at row {rows[0]} "
          f"(exact, no thickening)")

    # Vectors round-trip through the window's coordinate frame.
    vec = client.get("/api/vectors").json()["polylines"]
    assert vec, "the painted stroke should have a centerline"
    line = max(vec, key=len)
    assert 0 <= line[0][0] <= w and 0 <= line[0][1] <= h, f"not in window coords: {line[0]}"
    client.post("/api/set_vectors", json={"polylines": vec}).raise_for_status()
    full = state.result.polylines_px[0]
    assert x <= full[0][0] <= x + w and y <= full[0][1] <= y + h, \
        f"set_vectors did not map back to full-res: {full[0]}"
    back = client.get("/api/vectors").json()["polylines"]
    assert back[0][0] == vec[0][0], "window -> full -> window must be lossless"
    print("[4/4] vectors round-trip window -> full-res -> window unchanged")

    # And the overview still works after all that.
    whole = client.post("/api/edit_view", json={"whole": True}).json()["edit"]
    assert (whole["x"], whole["y"]) == (0, 0) and whole["width"] == MAX_EDIT_DIM
    print("\nFull-res window OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
